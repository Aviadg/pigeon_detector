import cv2
import numpy as np
import argparse
from datetime import datetime
import time
import os
import threading
import queue
import RPi.GPIO as GPIO

class FrameGrabber(threading.Thread):
    """Thread class for grabbing frames from RTSP stream in background"""
    def __init__(self, rtsp_url, frame_queue, max_queue_size=2):
        super().__init__()
        self.rtsp_url = rtsp_url
        self.frame_queue = frame_queue
        self.max_queue_size = max_queue_size
        self.running = True
        self.daemon = True  # Thread will exit when main program exits
        self.total_frames = 0
        self.reconnect_count = 0
        
    def run(self):
        # Connection options for RTSP
        os.environ["OPENCV_FFMPEG_CAPTURE_OPTIONS"] = "rtsp_transport;tcp|analyzeduration;1000000|buffer_size;1000000|max_delay;500000"
        
        while self.running:
            try:
                # Create a new connection each time for freshness
                self.reconnect_count += 1
                print(f"Connecting to camera (attempt #{self.reconnect_count}): {self.rtsp_url}")
                
                cap = cv2.VideoCapture(self.rtsp_url, cv2.CAP_FFMPEG)
                
                if not cap.isOpened():
                    print("Failed to open RTSP stream, retrying in 3 seconds...")
                    time.sleep(3)
                    continue
                
                # Try different buffer settings
                cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
                
                print(f"Successfully connected to {self.rtsp_url}")
                
                # Only read limited number of frames before reconnecting
                frame_count = 0
                max_frames_per_connection = 50  # Reconnect after this many frames
                
                while self.running and frame_count < max_frames_per_connection:
                    # Clear any buffered frames
                    for _ in range(2):
                        cap.grab()
                        
                    # Get a fresh frame
                    ret, frame = cap.read()
                    
                    if not ret or frame is None or frame.size == 0:
                        print("Failed to grab valid frame, reconnecting...")
                        break
                    
                    # Put frame in queue, replacing old frame if queue is full
                    if self.frame_queue.full():
                        try:
                            self.frame_queue.get_nowait()  # Discard oldest frame
                        except queue.Empty:
                            pass
                    
                    self.frame_queue.put(frame)
                    frame_count += 1
                    self.total_frames += 1
                    
                    # Small sleep to prevent CPU overuse
                    time.sleep(0.01)
                
                print(f"Periodic reconnect after {frame_count} frames")
                # Release this capture and create a new one in next loop
                cap.release()
                
            except Exception as e:
                print(f"Error in frame grabber: {str(e)}")
                time.sleep(3)  # Wait before retrying

class YOLOBirdMonitor:
    def __init__(self, rtsp_url, model_path, threshold=0.3, 
                 save_frames=False, cooldown_period=60, bird_class_id=14):
        # Store configuration
        self.rtsp_url = rtsp_url
        self.model_path = model_path
        self.threshold = threshold
        self.save_frames = save_frames
        self.cooldown_period = cooldown_period
        self.bird_class_id = bird_class_id
        self.frame_counter = 0
        
        # Detection tracking
        self.consecutive_detections = 0
        self.last_detection_time = 0
        self.in_cooldown = False
        
        # Setup GPIO
        self.led_pin = 21
        GPIO.setmode(GPIO.BCM)
        GPIO.setup(self.led_pin, GPIO.OUT)
        GPIO.output(self.led_pin, GPIO.LOW)
        
        # Create results directory if saving frames
        if self.save_frames:
            self.results_dir = "results"
            os.makedirs(self.results_dir, exist_ok=True)
        
        # Setup frame queue and grabber thread
        self.frame_queue = queue.Queue(maxsize=2)
        self.grabber = None
        
        # Load model flag - defer actual loading until first use
        self.model = None
        
    def __del__(self):
        GPIO.cleanup()
        if self.grabber is not None:
            self.grabber.running = False
            self.grabber.join(timeout=1.0)
    
    def load_model(self):
        """Load the YOLO model - only when needed"""
        if self.model is not None:
            return

        try:
            from ultralytics import YOLO
            
            # Load the model
            print(f"Loading YOLO model from {self.model_path}...")
            self.model = YOLO(self.model_path)
            
            # Set confidence threshold
            self.model.conf = self.threshold
            
            print(f"YOLO model loaded successfully (threshold: {self.threshold})")
            
        except Exception as e:
            print(f"Failed to load YOLO model: {str(e)}")
            raise
    
    def start_frame_grabber(self):
        """Start the background frame grabber thread"""
        if self.grabber is None or not self.grabber.is_alive():
            self.grabber = FrameGrabber(self.rtsp_url, self.frame_queue)
            self.grabber.start()
            print("Frame grabber thread started")
            
            # Wait for first frame
            print("Waiting for first frame...")
            start_wait = time.time()
            while self.frame_queue.empty():
                time.sleep(0.1)
                if time.time() - start_wait > 30:  # Wait up to 30 seconds
                    print("Timeout waiting for first frame")
                    return False
            
            print("Received first frame, stream is active")
            return True
        return True
    
    def save_frame(self, frame, confidence):
        """Save the processed frame if enabled"""
        if not self.save_frames:
            return None
            
        is_bird = confidence >= self.threshold
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"frame_{timestamp}_{self.frame_counter:04d}_{'bird' if is_bird else 'nobird'}_{confidence:.2%}.jpg"
        filepath = os.path.join(self.results_dir, filename)
        cv2.imwrite(filepath, frame)
        self.frame_counter += 1
        return filepath
    
    def handle_detection(self, confidence):
        """Handle bird detection logic and GPIO control"""
        current_time = time.time()
        
        # Check if we're in cooldown
        if self.in_cooldown:
            if current_time - self.last_detection_time >= self.cooldown_period:
                self.in_cooldown = False
                self.consecutive_detections = 0
                GPIO.output(self.led_pin, GPIO.LOW)
                print("Cooldown period ended")
            return False
            
        # Process detection
        if confidence >= self.threshold:
            self.consecutive_detections += 1
            print(f"Bird detected! Confidence: {confidence:.2%}, Consecutive detections: {self.consecutive_detections}/3")
            
            if self.consecutive_detections >= 3:
                print("🐦 CONFIRMED BIRD DETECTION! 🐦")
                GPIO.output(self.led_pin, GPIO.HIGH)
                self.last_detection_time = current_time
                self.in_cooldown = True
                return True
        else:
            # Gradually reduce consecutive detections instead of resetting to zero
            # This makes the system more robust to occasional misdetections
            self.consecutive_detections = max(0, self.consecutive_detections - 1)
            
        return False
    
    def detect_birds(self, frame):
        """Run YOLO detection on a frame and look for birds"""
        start_time = time.time()
        
        # Ensure model is loaded
        if self.model is None:
            self.load_model()
        
        # Run detection
        results = self.model(frame, verbose=False)
        
        # Process results for bird detection
        max_confidence = 0.0
        result = results[0]  # Get first result
        
        # Check for bird detections (bird_class_id)
        for box in result.boxes:
            cls = int(box.cls.item())
            conf = box.conf.item()
            
            if cls == self.bird_class_id and conf > max_confidence:
                max_confidence = conf
        
        # Get annotated frame
        annotated_frame = result.plot()
        
        inference_time = time.time() - start_time
        return annotated_frame, max_confidence, inference_time
    
    def run(self):
        """Main monitoring loop"""
        print(f"Starting YOLO bird monitoring with threaded frame grabber...")
        system_start_time = time.time()
        processed_frames = 0
        last_status_time = time.time()
        
        # Start the frame grabber thread
        if not self.start_frame_grabber():
            print("Failed to start frame grabber, exiting")
            return
        
        while True:
            try:
                # Check if grabber is running
                if self.grabber is None or not self.grabber.is_alive():
                    print("Frame grabber thread died, restarting...")
                    if not self.start_frame_grabber():
                        print("Failed to restart frame grabber, waiting...")
                        time.sleep(5)
                        continue
                
                # Try to get a frame
                try:
                    if self.frame_queue.empty():
                        print("Frame queue empty, waiting...")
                        time.sleep(0.5)
                        continue
                    
                    frame = self.frame_queue.get(timeout=1.0)
                except queue.Empty:
                    print("Timeout getting frame from queue")
                    continue
                
                current_time = time.time()
                
                # Detect birds in the frame
                annotated_frame, confidence, inference_time = self.detect_birds(frame)
                
                # Handle detection result
                confirmed_detection = self.handle_detection(confidence)
                
                # Save the frame if enabled
                if self.save_frames or confirmed_detection:
                    self.save_frame(annotated_frame, confidence)
                
                # Update processed frame count
                processed_frames += 1
                
                # Print status every 10 seconds
                if current_time - last_status_time >= 10.0:
                    elapsed_time = current_time - system_start_time
                    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                    
                    print(f"\n{'=' * 50}")
                    print(f"Status at: {timestamp}")
                    print(f"Uptime: {elapsed_time:.1f} seconds")
                    print(f"Frames acquired: {self.grabber.total_frames}")
                    print(f"Frames processed: {processed_frames}")
                    if elapsed_time > 0:
                        print(f"Processing rate: {processed_frames/elapsed_time:.2f} fps")
                    print(f"Last detection confidence: {confidence:.2%}")
                    print(f"Last processing time: {inference_time:.3f} seconds")
                    print(f"Consecutive detections: {self.consecutive_detections}")
                    print(f"In cooldown: {self.in_cooldown}")
                    if self.in_cooldown:
                        cooldown_remaining = self.cooldown_period - (current_time - self.last_detection_time)
                        print(f"Cooldown remaining: {cooldown_remaining:.1f} seconds")
                    print(f"{'=' * 50}")
                    
                    last_status_time = current_time
                
                # Sleep to maintain approximately 1 fps 
                # (adjusted for processing time to keep consistent pace)
                processing_time = time.time() - current_time
                sleep_time = max(0.01, 1.0 - processing_time)
                time.sleep(sleep_time)
                
            except KeyboardInterrupt:
                print("\nStopping monitoring...")
                break
            except Exception as e:
                print(f"Error in main loop: {str(e)}")
                time.sleep(5)
                continue
        
        # Clean up
        if self.grabber is not None:
            self.grabber.running = False
            self.grabber.join(timeout=1.0)

def install_dependencies():
    """Install required dependencies if not already installed"""
    try:
        import ultralytics
        print("Ultralytics already installed")
    except ImportError:
        print("Installing Ultralytics YOLO...")
        os.system("pip install ultralytics")
    
    try:
        import cv2
        print("OpenCV already installed")
    except ImportError:
        print("Installing OpenCV...")
        os.system("pip install opencv-python")

def main():
    parser = argparse.ArgumentParser(description='YOLO Bird Detection from RTSP Stream')
    parser.add_argument('--rtsp-url', type=str, default=os.getenv('RTSP_URL'),
                      help='RTSP URL for camera stream')
    parser.add_argument('--model-path', type=str, default='yolov8n.pt',
                      help='Path to Ultralytics YOLO model')
    parser.add_argument('--threshold', type=float, default=float(os.getenv('DETECTION_THRESHOLD', '0.3')),
                      help='Detection threshold (default: 0.3)')
    parser.add_argument('--save-frames', type=bool, default=os.getenv('SAVE_FRAMES', 'false').lower() == 'true',
                      help='Save processed frames')
    parser.add_argument('--cooldown-period', type=int, default=int(os.getenv('COOLDOWN_PERIOD', '60')),
                      help='Cooldown period in seconds after confirmed detection')
    parser.add_argument('--bird-class-id', type=int, default=int(os.getenv('BIRD_CLASS_ID', '14')),
                      help='Class ID for birds in the model (default 14 for COCO)')

    args = parser.parse_args()
    
    # Check for dependencies
    install_dependencies()
    
    # Set higher priority for process
    try:
        os.nice(-10)  # Lower value means higher priority
    except:
        pass
    
    # Create and run the monitor
    monitor = YOLOBirdMonitor(
        rtsp_url=args.rtsp_url,
        model_path=args.model_path,
        threshold=args.threshold,
        save_frames=args.save_frames,
        cooldown_period=args.cooldown_period,
        bird_class_id=args.bird_class_id
    )
    monitor.run()

if __name__ == "__main__":
    main()