import os
import glob
from ultralytics import YOLO
import time

def convert_yolo_models():
    """
    Find and convert all YOLO models (.pt, .pth, .torchscript) to NCNN format
    """
    # Look for models in the models directory
    model_files = []
    for ext in ['.pt', '.pth', '.torchscript']:
        model_files.extend(glob.glob(f'models/*{ext}'))
    
    if not model_files:
        print("No YOLO models found to convert. Using default yolov8n.pt")
        # Use default model if none found
        try:
            model = YOLO('yolov8n.pt')
            print("Converting default yolov8n model to NCNN format...")
            start_time = time.time()
            
            # Export to NCNN format
            ncnn_path = 'models/yolov8n_ncnn_model'
            model.export(format="ncnn", imgsz=640)
            
            # Move the exported files to the models directory
            if not os.path.exists('models'):
                os.makedirs('models')
                
            # YOLO exports to current directory, so we need to move files
            exported_files = glob.glob('*_ncnn_model.*')
            for file in exported_files:
                target_file = os.path.join('models', os.path.basename(file))
                os.rename(file, target_file)
                
            elapsed_time = time.time() - start_time
            print(f"Default model converted successfully in {elapsed_time:.2f} seconds")
        except Exception as e:
            print(f"Failed to convert default model: {str(e)}")
        return
        
    # Convert all found models
    for model_path in model_files:
        try:
            model_name = os.path.basename(model_path)
            print(f"Converting model {model_name} to NCNN format...")
            start_time = time.time()
            
            # Load original model
            model = YOLO(model_path)
            
            # Export to NCNN format
            ncnn_path = f"{os.path.splitext(model_path)[0]}_ncnn_model"
            model.export(format="ncnn", imgsz=640)
            
            elapsed_time = time.time() - start_time
            print(f"Model {model_name} converted successfully in {elapsed_time:.2f} seconds")
            
        except Exception as e:
            print(f"Failed to convert model {model_path}: {str(e)}")

if __name__ == "__main__":
    convert_yolo_models()