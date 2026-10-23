import torch
from models.raspi_yolo import RaspiYOLO

model = RaspiYOLO(...)
model.load_state_dict(torch.load('weights/raspi_yolo_last.pt'))
model.eval()

# ONNX (good with onnxruntime on RPi)
dummy = torch.randn(1, 3, 320, 320)
torch.onnx.export(model, dummy, 'raspi_yolo.onnx',
                  input_names=['images'],
                  output_names=['dets'],
                  opset_version=12,
                  dynamic_axes={'images': {0: 'batch'}})

# Optional: further convert to TFLite via onnx-tf or use ultralytics export if you switch base
print('Exported. On RPi prefer onnxruntime or TFLite interpreter.')