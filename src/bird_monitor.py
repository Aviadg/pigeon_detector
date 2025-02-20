import cv2
import numpy as np
from PIL import Image
import tflite_runtime.interpreter as tflite
import argparse
from datetime import datetime
import time
import os
import RPi.GPIO as GPIO

class RTSPBirdMonitor:
    def __init__(self, rtsp_url, model_path, threshold=0.5, save_frames=False, cooldown_period=60):
        # Initialize TFLite interpreter
        self.interpreter = tflite.Interpreter(model_path=model_path)
        self.interpreter.allocate_tensors()
        
        # Get input and output details
        self.input_details = self.interpreter.get_input_details()
        self.output_details = self.interpreter.get_output_details()
        
        # Get input shape
        self.input_shape = self.input_details[0]['shape']
        self.img_size = self.input_shape[1]
        
        # Store configuration
        self.rtsp_url = rtsp_url
        self.threshold = threshold
        self.save_frames = save_frames
        self.cooldown_period = cooldown_period
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
        
    def __del__(self):
        GPIO.cleanup()
        
    def connect_camera(self):
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

    def flush_buffer(self):
        """Flush the buffer to get the latest frame"""
        for _ in range(5):  # Increased from 2 to 5 for better buffer clearing
            self.cap.grab()
            
    def preprocess_frame(self, frame):
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        pil_image = Image.fromarray(rgb_frame)
        pil_image = pil_image.resize((self.img_size, self.img_size))
        processed_pil = pil_image.copy()
        img_array = np.array(pil_image, dtype=np.float32)
        img_array = np.expand_dims(img_array, axis=0)
        return img_array, processed_pil
    
    def save_processed_frame(self, pil_image, confidence):
        if not self.save_frames:
            return None
            
        is_bird = confidence >= self.threshold
        filename = f"frame_{self.frame_counter:04d}_{'bird' if is_bird else 'nobird'}_{confidence:.2%}.jpg"
        filepath = os.path.join(self.results_dir, filename)
        pil_image.save(filepath)
        self.frame_counter += 1
        return filepath
    
    def handle_detection(self, confidence):
        current_time = time.time()
        
        # Check if we're in cooldown
        if self.in_cooldown:
            if current_time - self.last_detection_time >= self.cooldown_period:
                self.in_cooldown = False
                self.consecutive_detections = 0
                GPIO.output(self.led_pin, GPIO.LOW)
            return False
            
        # Process detection
        if confidence >= self.threshold:
            self.consecutive_detections += 1
            if self.consecutive_detections >= 3:
                print("Confirmed bird detection!")
                GPIO.output(self.led_pin, GPIO.HIGH)
                self.last_detection_time = current_time
                self.in_cooldown = True
                return True
        else:
            self.consecutive_detections = 0
            
        return False
    
    def predict_frame(self, frame):
        processed_frame, processed_pil = self.preprocess_frame(frame)
        
        self.interpreter.set_tensor(self.input_details[0]['index'], processed_frame)
        self.interpreter.invoke()
        
        prediction = self.interpreter.get_tensor(self.output_details[0]['index'])
        confidence = float(prediction[0][0])
        
        self.save_processed_frame(processed_pil, confidence)
        
        return confidence
    
    def run(self):
        print(f"Starting RTSP monitoring (Image size: {self.img_size}x{self.img_size})...")
        system_start_time = time.time()
        frame_count = 0
        
        while True:
            try:
                if self.cap is None or not self.cap.isOpened():
                    print("Reconnecting to camera...")
                    self.connect_camera()
                
                # Get current frame with optimized buffer management
                self.flush_buffer()
                ret, frame = self.cap.read()
                
                if not ret or frame is None or frame.size == 0:
                    print("Failed to grab valid frame, attempting to reconnect...")
                    self.connect_camera()
                    continue
                
                # Update frame count and calculate timing
                current_time = time.time()
                frame_count += 1
                elapsed_time = current_time - system_start_time
                
                # Only run detection once per second
                if current_time - self.last_detection_check >= 1.0:
                    start_process = time.time()
                    confidence = self.predict_frame(frame)
                    process_time = time.time() - start_process
                    
                    confirmed_detection = self.handle_detection(confidence)
                    self.last_detection_check = current_time
                    
                    # Print status
                    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                    print(f"\nTimestamp: {timestamp}")
                    print(f"Processing time: {process_time:.3f} seconds")
                    print(f"Confidence: {confidence:.2%}")
                    print(f"Consecutive detections: {self.consecutive_detections}")
                    print(f"In cooldown: {self.in_cooldown}")
                    print(f"Average FPS: {frame_count/elapsed_time:.2f}")
                    print("-" * 50)
                
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
    parser = argparse.ArgumentParser(description='Bird Detection from RTSP Stream')
    parser.add_argument('--rtsp-url', type=str, default=os.getenv('RTSP_URL'),
                      help='RTSP URL for camera stream')
    parser.add_argument('--model-path', type=str, default='models/model_rpi.tflite',
                      help='Path to TFLite model file')
    parser.add_argument('--threshold', type=float, default=float(os.getenv('DETECTION_THRESHOLD', '0.5')),
                      help='Detection threshold')
    parser.add_argument('--save-frames', type=bool, default=os.getenv('SAVE_FRAMES', 'false').lower() == 'true',
                      help='Save processed frames')
    parser.add_argument('--cooldown-period', type=int, default=int(os.getenv('COOLDOWN_PERIOD', '60')),
                      help='Cooldown period in seconds after confirmed detection')

    args = parser.parse_args()
    
    # Add TCP transport if not specified in URL
    if 'rtsp://' in args.rtsp_url and '?' not in args.rtsp_url:
        args.rtsp_url += '?tcp'
    
    monitor = RTSPBirdMonitor(
        rtsp_url=args.rtsp_url,
        model_path=args.model_path,
        threshold=args.threshold,
        save_frames=args.save_frames,
        cooldown_period=args.cooldown_period
    )
    monitor.run()

if __name__ == "__main__":
    main()