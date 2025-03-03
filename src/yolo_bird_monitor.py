import cv2
import numpy as np
import argparse
from datetime import datetime
import time
import os
import RPi.GPIO as GPIO
import glob

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
        
        # Load YOLO model
        self.load_model()
        
    def __del__(self):
        GPIO.cleanup()
        if self.cap is not None:
            self.cap.release()
    
    def load_model(self):
        """Load the YOLO model - prefer pre-converted NCNN models"""
        try:
            from ultralytics import YOLO
            
            # Check if we should use a pre-converted model
            ncnn_models = []
            model_dir = os.path.dirname(self.model_path)
            model_base = os.path.splitext(os.path.basename(self.model_path))[0]
            
            # Look for NCNN models
            ncnn_pattern = os.path.join(model_dir, f"{model_base}_ncnn_model*")
            ncnn_models = glob.glob(ncnn_pattern)
            
            # If no exact match, look for any NCNN model
            if not ncnn_models:
                ncnn_models = glob.glob(os.path.join(model_dir, "*_ncnn_model*"))
            
            # Use NCNN model if available
            if ncnn_models:
                # Get the directory path of the first NCNN model
                ncnn_path = os.path.dirname(ncnn_models[0])
                if not ncnn_path:  # If in current directory
                    ncnn_path = os.path.splitext(ncnn_models[0])[0]
                else:
                    ncnn_path = os.path.join(ncnn_path, os.path.splitext(os.path.basename(ncnn_models[0]))[0])
                    
                print(f"Using pre-converted NCNN model: {ncnn_path}")
                self.model = YOLO(ncnn_path)
            else:
                # No pre-converted model found, use original one
                print(f"No pre-converted NCNN model found. Using original model: {self.model_path}")
                self.model = YOLO(self.model_path)
            
            # Set confidence threshold
            self.model.conf = self.threshold
            
            print(f"YOLO model loaded successfully (threshold: {self.threshold})")
            
        except Exception as e:
            print(f"Failed to load YOLO model: {str(e)}")
            raise
        
    def connect_camera(self):
        """Connect to the RTSP camera stream"""
        if self.cap is not None:
            self.cap.release()
        
        # Enhanced camera connection with optimized settings
        self.cap = cv2.VideoCapture(self.rtsp_url, cv2.CAP_FFMPEG)
        
        if not self.cap.isOpened():
            raise Exception("Failed to connect to RTSP stream")
        
        # Optimize for latency
        self.cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        self.cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
        self.cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc('H', '2', '6', '4'))
        self.cap.set(cv2.CAP_PROP_FPS, 30)
        
        print(f"Connected to camera: {self.rtsp_url}")

    def flush_buffer(self):
        """Flush the buffer to get the latest frame"""
        for _ in range(5):  # Clear several frames to get the most recent
            self.cap.grab()
    
    def save_frame(self, frame, confidence):
        """Save the processed frame if enabled"""
        if not self.save_frames:
            return None
            
        is_bird = confidence >= self.threshold
        filename = f"frame_{self.frame_counter:04d}_{'bird' if is_bird else 'nobird'}_{confidence:.2%}.jpg"
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
                if self.cap is None or not self.cap.isOpened():
                    print("Connecting to camera...")
                    self.connect_camera()
                
                # Get current frame with optimized buffer management
                self.flush_buffer()
                ret, frame = self.cap.read()
                
                if not ret or frame is None or frame.size == 0:
                    print("Failed to grab valid frame, attempting to reconnect...")
                    time.sleep(1)
                    self.connect_camera()
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
                    print(f"Average frame rate: {frame_count/elapsed_time:.2f} fps")
                    print("-" * 50)
                
                # Small delay to prevent CPU overuse
                time.sleep(0.05)
                
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

def install_dependencies():
    """Install required dependencies if not already installed"""
    try:
        import ultralytics
        print("Ultralytics already installed")
    except ImportError:
        print("Installing Ultralytics YOLO...")
        os.system("pip install ultralytics")
    
    # Install NCNN dependencies if not present
    try:
        import ncnn
        print("NCNN already installed")
    except ImportError:
        print("Installing NCNN dependencies...")
        os.system("pip install ncnn")

def main():
    parser = argparse.ArgumentParser(description='YOLO Bird Detection from RTSP Stream')
    parser.add_argument('--rtsp-url', type=str, default=os.getenv('RTSP_URL'),
                      help='RTSP URL for camera stream')
    parser.add_argument('--model-path', type=str, default='models/yolov8n.pt',
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