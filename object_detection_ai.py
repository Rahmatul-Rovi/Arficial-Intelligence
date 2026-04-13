from ultralytics import YOLO

# ১. Pre-trained AI model load 
model = YOLO('yolov8n.pt') 

results = model.predict(source='https://ultralytics.com/images/bus.jpg', save=True)

for result in results:
    boxes = result.boxes
    for box in boxes:
        cls = int(box.cls[0])
        name = model.names[cls]
        conf = float(box.conf[0])
        print(f"AI Detected: {name} (Confidence: {conf:.2f})")

print("\nResult saved 'runs/detect/predict' folder-e.")