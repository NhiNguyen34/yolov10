import os
import cv2
import argparse
from ultralytics import YOLOv10

def main(config_path, epochs):
    
    # Change dir to yolov10
    os.chdir('yolov10')
    
    # Install independences
    os.system('pip install .')
    
    # Set HOME dir
    HOME = os.getcwd()
    print(HOME)
    
    # Disable wandb
    os.environ['WANDB_MODE'] = 'disabled'
    
    # Get weights
    os.system(f"wget -P {HOME}/weights -q https://github.com/THU-MIG/yolov10/releases/download/v1.1/yolov10n.pt")

    model = YOLOv10(f'{HOME}/weights/yolov10n.pt')
    
    results = model.train(data=config_path, epochs=epochs, imgsz=640)
    
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train YOLOv10")
    parser.add_argument('config', type=str, help='Path configuration')
    parser.add_argument('epochs', type=str, help='Epochs configuration')

    
    args = parser.parse_args()
    
    main(args.config, args.epochs)
    
    