import cv2
import numpy as np
import argparse
from datetime import datetime
import time
import os
import RPi.GPIO as GPIO

class YOLOBirdMonitor:
    def __init__(self, rtsp_url, model_path, threshold=0.5, 
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
        
        # Initialize video capture
        self.cap = None
        self.last_detection_check = 0  # Track when we last ran detection
        self.reconnect_count = 0
        self.last_frame_time = 0
        
        # Load model flag - defer actual loading until first use
        self.model = None
        
    def __del__(self):
        GPIO.cleanup()
        if self.cap is not None:
            self.cap.release()
    
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
        
    def connect_camera(self):
        """Connect to the RTSP camera stream with improved settings"""
        if self.cap is not None:
            self.cap.release()
            
        # Completely reset the connection
        self.reconnect_count += 1
        print(f"Connecting to camera (attempt #{self.reconnect_count}): {self.rtsp_url}")
        
        # Try different connection methods based on prior success
        if self.reconnect_count % 3 == 0:
            # Every third attempt, try with FFMPEG backend
            print("Trying FFMPEG backend...")
            self.cap = cv2.VideoCapture(self.rtsp_url, cv2.CAP_FFMPEG)
        elif self.reconnect_count % 3 == 1:
            # First and every third attempt thereafter, try with default backend
            print("Trying default backend...")
            self.cap = cv2.VideoCapture(self.rtsp_url)
        else:
            # Second and every third attempt thereafter, try with GSTREAMER backend
            print("Trying GSTREAMER backend...")
            self.cap = cv2.VideoCapture(self.rtsp_url, cv2.CAP_GSTREAMER)
        
        if not self.cap.isOpened():
            print("Failed to open camera connection!")
            return False
        
        # Try to disable buffering
        self.cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        
        # Try to limit frame rate
        self.cap.set(cv2.CAP_PROP_FPS, 1)
        
        # Try to reset position
        self.cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
        
        # Read a test frame to confirm connection
        ret, frame = self.cap.read()
        if not ret or frame is None or frame.size == 0:
            print("Could open camera but failed to read frame!")
            self.cap.release()
            self.cap = None
            return False
            
        print(f"Successfully connected to camera. Frame size: {frame.shape}")
        return True

    def get_new_frame(self):
        """Get a new frame from the camera with improved handling"""
        # Check if we have a valid camera connection
        if self.cap is None or not self.cap.isOpened():
            if not self.connect_camera():
                time.sleep(2)  # Wait before retrying
                return None
        
        # Clear any buffered frames
        for _ in range(10):  # Drop more frames to ensure we get a fresh one
            self.cap.grab()
        
        # Try to get a new frame
        ret, frame = self.cap.read()
        
        # Validate the frame
        if not ret or frame is None or frame.size == 0:
            print("Failed to grab valid frame from camera")
            # Force reconnect
            self.cap.release()
            self.cap = None
            return None
        
        # Check for frame similarity (stuck frame detection)
        current_time = time.time()
        if hasattr(self, 'last_frame') and self.last_frame is not None:
            # Calculate rough similarity using mean of absolute difference
            diff = cv2.absdiff(frame, self.last_frame)
            mean_diff = np.mean(diff)
            
            # If frames are too similar and it's been more than 2 seconds
            if mean_diff < 0.1 and (current_time - self.last_frame_time) > 2.0:
                print(f"WARNING: Detected potentially stuck frame (diff: {mean_diff:.4f})")
                
                # Try reconnecting if frames appear stuck
                if mean_diff < 0.05:
                    print("Frame appears stuck, reconnecting...")
                    self.cap.release()
                    self.cap = None
                    return None
        
        # Update frame tracking
        self.last_frame = frame.copy()
        self.last_frame_time = current_time
        
        return frame
    
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
            print(f"Bird detected! Consecutive detections: {self.consecutive_detections}/3")
            
            if self.consecutive_detections >= 3:
                print("Confirmed bird detection!")
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
        print(f"Starting YOLO bird monitoring (one detection per second)...")
        system_start_time = time.time()
        frame_count = 0
        
        while True:
            try:
                # Get a new frame with improved handling
                frame = self.get_new_frame()
                
                if frame is None:
                    # If we couldn't get a valid frame, retry after a delay
                    print("Failed to get valid frame, retrying...")
                    time.sleep(1)
                    continue
                
                # Update frame count
                frame_count += 1
                current_time = time.time()
                
                # Only run detection once per second
                if current_time - self.last_detection_check >= 1.0:
                    # Detect birds in the frame
                    annotated_frame, confidence, inference_time = self.detect_birds(frame)
                    
                    # Handle detection result
                    confirmed_detection = self.handle_detection(confidence)
                    
                    # Save the frame if enabled
                    self.save_frame(annotated_frame, confidence)
                    
                    # Update last detection time
                    self.last_detection_check = current_time
                    
                    # Print status
                    elapsed_time = current_time - system_start_time
                    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                    print(f"\nTimestamp: {timestamp}")
                    print(f"Processing time: {inference_time:.3f} seconds")
                    print(f"Confidence: {confidence:.2%}")
                    print(f"Consecutive detections: {self.consecutive_detections}")
                    print(f"In cooldown: {self.in_cooldown}")
                    print(f"Frames processed: {frame_count}")
                    if elapsed_time > 0:
                        print(f"Average frame rate: {frame_count/elapsed_time:.2f} fps")
                    print("-" * 50)
                
                # Sleep to prevent CPU overuse and maintain ~1 fps processing rate
                # Use adaptive sleep to maintain close to 1 fps
                processing_time = time.time() - current_time
                sleep_time = max(0, 1.0 - processing_time)
                if sleep_time > 0:
                    time.sleep(sleep_time)
                
            except KeyboardInterrupt:
                print("\nStopping monitoring...")
                break
            except Exception as e:
                print(f"Error: {str(e)}")
                print("Waiting 5 seconds before retry...")
                time.sleep(5)
                continue
        
        if self.cap is not None:
            self.cap.release()

def main():
    parser = argparse.ArgumentParser(description='YOLO Bird Detection from RTSP Stream')
    parser.add_argument('--rtsp-url', type=str, default=os.getenv('RTSP_URL'),
                      help='RTSP URL for camera stream')
    parser.add_argument('--model-path', type=str, default='yolov8n.pt',
                      help='Path to Ultralytics YOLO model')
    parser.add_argument('--threshold', type=float, default=float(os.getenv('DETECTION_THRESHOLD', '0.5')),
                      help='Detection threshold')
    parser.add_argument('--save-frames', type=bool, default=os.getenv('SAVE_FRAMES', 'false').lower() == 'true',
                      help='Save processed frames')
    parser.add_argument('--cooldown-period', type=int, default=int(os.getenv('COOLDOWN_PERIOD', '60')),
                      help='Cooldown period in seconds after confirmed detection')
    parser.add_argument('--bird-class-id', type=int, default=int(os.getenv('BIRD_CLASS_ID', '14')),
                      help='Class ID for birds in the model (default 14 for COCO)')

    args = parser.parse_args()
    
    # Add TCP transport if not specified in URL
    if args.rtsp_url and 'rtsp://' in args.rtsp_url and '?' not in args.rtsp_url:
        args.rtsp_url += '?tcp'
    
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