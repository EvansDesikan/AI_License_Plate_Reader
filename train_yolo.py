from ultralytics import YOLO
import multiprocessing

if __name__ == '__main__':
    # 1. Load the model
    model = YOLO('yolov8n.pt') 

    # 2. Train it
    results = model.train(
        data='data.yaml', 
        epochs=20,        
        imgsz=640,        
        batch=8,          
        name='plate_model',
        workers=2  # Reduced to 2 for better stability on Windows
    )

    print("🎉 Training Finished!")