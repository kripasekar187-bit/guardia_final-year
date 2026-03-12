from ultralytics import YOLO

try:
    model = YOLO('yolov8n-pose.pt')
    results = model.export(format='tflite', optimize=False)
except Exception as e:
    import traceback
    traceback.print_exc()
