from flask import Flask, Response
import cv2
import torch
import pyttsx3
import threading
import time

app = Flask(__name__)

# Load YOLOv5s pretrained model
model = torch.hub.load('ultralytics/yolov5', 'custom', path='yolov5s.pt')

camera = None

# Voice engine
engine = pyttsx3.init()

last_spoken = ""
last_time = 0

CONFIDENCE_THRESHOLD = 0.6

# ✅ ONLY THESE OBJECTS WILL BE SHOWN & SPOKEN
ALLOWED_CLASSES = [
    "person",
    "bottle",
    "cup",
    "cell phone",
    "laptop",
    "chair",
    "keyboard",
    "mouse",
    "book",
    "pen"
]

def speak_async(text):
    def run():
        engine.say(text)
        engine.runAndWait()
    threading.Thread(target=run, daemon=True).start()

def generate_frames():
    global camera, last_spoken, last_time

    while camera is not None:
        success, frame = camera.read()
        if not success:
            break

        results = model(frame)
        detections = results.pandas().xyxy[0]

        for _, row in detections.iterrows():
            label = row['name']
            conf = row['confidence']

            # ❌ FILTER WRONG OBJECTS
            if label not in ALLOWED_CLASSES:
                continue

            if conf > CONFIDENCE_THRESHOLD:
                now = time.time()
                if label != last_spoken or now - last_time > 4:
                    speak_async(f"{label} detected")
                    last_spoken = label
                    last_time = now

        frame = results.render()[0]

        ret, buffer = cv2.imencode('.jpg', frame)
        frame = buffer.tobytes()

        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')

@app.route('/')
def index():
    return """
<!DOCTYPE html>
<html>
<head>
<title>Object Detection with Voice</title>
<style>
body {
    background: linear-gradient(120deg,#141e30,#243b55);
    color: white;
    text-align: center;
    font-family: Arial;
}
img {
    border: 4px solid #00ffcc;
    margin-top: 20px;
    display: none;
}
button {
    padding: 12px 25px;
    font-size: 18px;
    margin: 15px;
    border-radius: 8px;
    cursor: pointer;
}
</style>

<script>
function startCamera() {
    fetch('/start').then(() => {
        let img = document.getElementById("video");
        img.src = "/video";
        img.style.display = "block";
    });
}

function stopCamera() {
    fetch('/stop').then(() => {
        let img = document.getElementById("video");
        img.src = "";
        img.style.display = "none";
    });
}
</script>
</head>

<body>
<h1>Object Detection</h1>

<img id="video" width="640" height="480">

<br>
<button onclick="startCamera()">Start Camera</button>
<button onclick="stopCamera()">Stop Camera</button>

</body>
</html>
"""

@app.route('/start')
def start():
    global camera
    if camera is None:
        camera = cv2.VideoCapture(0)
    return "Camera Started"

@app.route('/stop')
def stop():
    global camera
    if camera is not None:
        camera.release()
        camera = None
    return "Camera Stopped"

@app.route('/video')
def video():
    return Response(generate_frames(),
                    mimetype='multipart/x-mixed-replace; boundary=frame')

if __name__ == "__main__":
    app.run(debug=True)
