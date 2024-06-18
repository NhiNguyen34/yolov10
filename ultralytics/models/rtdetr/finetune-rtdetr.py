from ultralytics import RTDETR
import argparse
import os

def main(rtdetr_w, data_yaml, epochs):
    
    os.environ['WANDB_MODE'] = 'disabled'

    # Load model
    model = RTDETR(rtdetr_w)
    results = model.train(data=data_yaml, epochs=epochs, imgsz=640)
    
    # Save checkpoint
    model.save('rtdetr-l-visdrone.pt')
    

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train YOLOv10")
    parser.add_argument('rtdetr_w', type=str, help='rtdetr_w')
    parser.add_argument('data_yaml', type=str, help='data_yaml')
    parser.add_argument('epochs', type=int, help='epochs')

    args = parser.parse_args()
    
    main(args.rtdetr_w, args.data_yaml, args.epochs)
    