import cv2
import torch
import time
import numpy as np
from models.raspi_yolo import RaspiYOLO
import yaml

def letterbox(im, new_shape=(320, 320)):
    # standard letterbox implementation
    ...

def main():
    with open('configs/raspi.yaml') as f:
        cfg = yaml.safe_load(f)

    model = RaspiYOLO(num_classes=cfg['model']['num_classes'])
    model.load_state_dict(torch.load('weights/raspi_yolo_last.pt', map_location='cpu'))
    model.eval()
    model.reset_memory()

    # RPi Camera or USB
    cap = cv2.VideoCapture(0)   # or use picamera2
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        img = letterbox(frame, cfg['model']['input_size'])
        tensor = torch.from_numpy(img).permute(2, 0, 1).float() / 255.0
        tensor = tensor.unsqueeze(0)

        t0 = time.time()
        with torch.no_grad():
            det_outs, temporal_pred, _ = model(tensor, mode='temporal')
        fps = 1.0 / (time.time() - t0 + 1e-6)

        # Decode det_outs → boxes, scores, classes (implement NMS)
        # Draw boxes + show LSTM memory status
        cv2.putText(frame, f'FPS: {fps:.1f}', (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        cv2.imshow('RaspiYOLO + LSTM', frame)
        if cv2.waitKey(1) == 27:
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == '__main__':
    main()