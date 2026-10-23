import torch
from torch.utils.data import DataLoader
from models.raspi_yolo import RaspiYOLO
from losses.box_loss import CIoULoss
from losses.class_loss import FocalLoss
import yaml

def main(cfg_path='configs/raspi.yaml'):
    with open(cfg_path) as f:
        cfg = yaml.safe_load(f)

    model = RaspiYOLO(
        num_classes=cfg['model']['num_classes'],
        width_mult=cfg['model']['width_mult']
    )
    # Move to GPU if available for training; on RPi stay on CPU or use quantized
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.to(device)

    # TODO: implement your Dataset that returns (img, targets)
    # train_loader = DataLoader(...)

    optimizer = torch.optim.AdamW(model.parameters(), lr=cfg['train']['lr'])
    box_loss = CIoULoss()
    cls_loss = FocalLoss()

    model.train()
    for epoch in range(cfg['train']['epochs']):
        for imgs, targets in train_loader:
            imgs = imgs.to(device)
            preds = model(imgs, mode='detect')
            # Compute losses on decoded boxes / classes
            loss = ... 
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        print(f'Epoch {epoch} loss: {loss.item():.4f}')

    torch.save(model.state_dict(), 'weights/raspi_yolo_last.pt')

if __name__ == '__main__':
    main()