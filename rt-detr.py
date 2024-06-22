import torch
import torch.distributed as dist
import torch.multiprocessing as mp
from ultralytics import RTDETR

def setup(rank, world_size):
    # Initializes the distributed backend which will take care of sychronizing nodes/GPUs
    dist.init_process_group("nccl", rank=rank, world_size=world_size)
    torch.cuda.set_device(rank)

def cleanup():
    dist.destroy_process_group()

def train(rank, world_size, model_path="rtdetr-l.pt", data="coco8.yaml", epochs=100, imgsz=640):
    setup(rank, world_size)
    
    # Load model and move it to the current GPU
    model = RTDETR(model_path).to(rank)
    model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[rank])

    # Train the model
    results = model.module.train(data=data, epochs=epochs, imgsz=imgsz)

    cleanup()

def run_inference(model, image_path, device):
    # Ensure the model is in evaluation mode
    model.eval()

    # Move the model to the appropriate device
    model.to(device)

    # Run inference
    results = model(image_path)
    return results

if __name__ == "__main__":
    # Number of GPUs available
    world_size = torch.cuda.device_count()

    if world_size > 1:
        # Multi-GPU training
        mp.spawn(train,
                 args=(world_size,),
                 nprocs=world_size,
                 join=True)
    else:
        # Single-GPU training
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        model = RTDETR("rtdetr-l.pt").to(device)

        # Display model information (optional)
        model.info()

        # Train the model (optional, reduce epochs for initial tests)
        train_model = False
        if train_model:
            model.train(data="coco8.yaml", epochs=10, imgsz=640)  # Use fewer epochs for testing

        # Run inference
        inference_results = run_inference(model, "path/to/bus.jpg", device)
        print(inference_results)
