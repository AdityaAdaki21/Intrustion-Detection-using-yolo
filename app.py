from flask import Flask, request, jsonify, render_template, Response
import cv2
import numpy as np
from ultralytics import YOLO
from PIL import Image

app = Flask(__name__)

# Load YOLOv8 model (ensure the model file is in the project directory)
model = YOLO("yolov8n.pt")

camera = None  # Initialize camera as None

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/detect', methods=['POST'])
def detect():
    if 'image' not in request.files:
        return jsonify({'error': 'No image provided'}), 400

    file = request.files['image']
    img = Image.open(file.stream)
    img = np.array(img)
    img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)

    results = model(img)
    detections = []
    for det in results[0].boxes:
        detection = {
            'class': int(det.cls[0]),
            'confidence': float(det.conf[0]),
            'box': [float(x) for x in det.xyxy[0]]
        }
        detections.append(detection)

    return jsonify(detections)

def generate_video_feed():
    global camera
    if camera is None:
        camera = cv2.VideoCapture(0)  # Initialize the camera here

    while True:
        ret, frame = camera.read()  # Use 'camera' here for consistency
        if not ret:
            print("Failed to capture image from camera.")
            break

        # Convert the frame from BGR (OpenCV format) to RGB (YOLOv8 format)
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        # Run YOLO detection on the frame
        results = model(frame_rgb, conf=0.3)  # Lower confidence threshold if needed

        # Draw bounding boxes and labels on the frame
        for det in results[0].boxes:
            x1, y1, x2, y2 = map(int, det.xyxy[0])
            cls = int(det.cls[0])
            conf = float(det.conf[0])

            # Draw the bounding box
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(frame, f"Class: {cls}, Conf: {conf:.2f}", (x1, y1 - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

        # Encode frame as JPEG
        ret, buffer = cv2.imencode('.jpg', frame)
        if not ret:
            print("Failed to encode frame.")
            continue
        frame = buffer.tobytes()

        # Yield frame as a multipart response for live video feed
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')

@app.route('/video_feed')
def video_feed():
    return Response(generate_video_feed(), mimetype='multipart/x-mixed-replace; boundary=frame')

@app.teardown_appcontext
def close_camera(exception):
    global camera
    if camera is not None:
        camera.release()  # Release the camera when the application context is torn down
        camera = None

if __name__ == '__main__':
    app.run(debug=True)
