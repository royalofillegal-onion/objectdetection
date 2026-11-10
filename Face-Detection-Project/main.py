import base64
from functools import lru_cache
import os

import cv2
import numpy as np
from flask import Flask, render_template_string, request


BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(BASE_DIR, "yolo11n.pt")
MAX_UPLOAD_SIZE = 10 * 1024 * 1024
ALLOWED_EXTENSIONS = {"jpg", "jpeg", "png"}

app = Flask(__name__)
app.config["MAX_CONTENT_LENGTH"] = MAX_UPLOAD_SIZE

face_cascade = cv2.CascadeClassifier(
    os.path.join(cv2.data.haarcascades, "haarcascade_frontalface_default.xml")
)

if face_cascade.empty():
    raise RuntimeError("OpenCV's frontal-face cascade could not be loaded.")


PAGE = """
<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>Face and Object Detection</title>
  <style>
    body { font-family: Arial, sans-serif; background: #f3f5f9; color: #20243a; margin: 0; }
    main { max-width: 760px; margin: 48px auto; padding: 32px; background: white;
           border-radius: 14px; box-shadow: 0 8px 28px #17213a16; }
    h1 { color: #333399; margin-top: 0; }
    p { line-height: 1.5; }
    form { display: flex; flex-wrap: wrap; gap: 12px; align-items: center; margin: 24px 0; }
    button, .download { background: #333399; color: white; border: 0; border-radius: 6px;
                        padding: 12px 18px; font-size: 15px; text-decoration: none; cursor: pointer; }
    img { display: block; max-width: 100%; max-height: 600px; margin: 20px auto; }
    .error { padding: 12px; background: #fff0f0; color: #a00000; border-radius: 6px; }
    .counts { font-weight: bold; }
    .hint { color: #606778; font-size: 14px; }
  </style>
</head>
<body>
  <main>
    <h1>AI Face &amp; Object Detection</h1>
    <p>Upload an image to detect faces and objects. On a phone, you can choose to take a photo.</p>
    <form method="post" enctype="multipart/form-data">
      <input type="file" name="image" accept="image/jpeg,image/png" capture="environment" required>
      <button type="submit">Detect</button>
    </form>
    <p class="hint">JPEG or PNG, up to 10 MB. Live webcam streaming is not available in this hosted version.</p>
    {% if error %}<p class="error">{{ error }}</p>{% endif %}
    {% if image_data %}
      <p class="counts">Faces: {{ face_count }} &nbsp;|&nbsp; Objects: {{ object_count }}</p>
      <img src="data:image/jpeg;base64,{{ image_data }}" alt="Image with detected faces and objects">
      <a class="download" href="data:image/jpeg;base64,{{ image_data }}" download="detection-result.jpg">Download result</a>
    {% endif %}
  </main>
</body>
</html>
"""


@lru_cache(maxsize=1)
def get_model():
    from ultralytics import YOLO

    return YOLO(MODEL_PATH)


def detect_faces(frame: np.ndarray) -> int:
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    gray = cv2.equalizeHist(gray)
    faces = face_cascade.detectMultiScale(
        gray,
        scaleFactor=1.1,
        minNeighbors=5,
        minSize=(35, 35),
    )

    for x, y, width, height in faces:
        cv2.rectangle(frame, (x, y), (x + width, y + height), (0, 255, 0), 2)
        cv2.putText(
            frame,
            "HUMAN FACE",
            (x, max(y - 10, 20)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (0, 255, 0),
            2,
        )

    return len(faces)


def detect_objects(frame: np.ndarray) -> int:
    results = get_model().predict(source=frame, conf=0.35, imgsz=640, verbose=False)
    object_count = 0

    for result in results:
        if result.boxes is None:
            continue

        for box in result.boxes:
            x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())
            class_id = int(box.cls[0])
            confidence = float(box.conf[0])
            object_name = get_model().names[class_id]
            label = f"{object_name} {confidence * 100:.1f}%"

            cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 0, 0), 2)
            cv2.putText(
                frame,
                label,
                (x1, max(y1 - 10, 20)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (255, 0, 0),
                2,
            )
            object_count += 1

    return object_count


@app.route("/", methods=["GET", "POST"])
def index():
    if request.method == "GET":
        return render_template_string(PAGE)

    uploaded_file = request.files.get("image")
    if uploaded_file is None or not uploaded_file.filename:
        return render_template_string(PAGE, error="Choose an image to upload."), 400

    extension = uploaded_file.filename.rsplit(".", 1)[-1].lower()
    if extension not in ALLOWED_EXTENSIONS:
        return render_template_string(
            PAGE, error="Upload a JPEG or PNG image."
        ), 400

    image_bytes = uploaded_file.read()
    frame = cv2.imdecode(np.frombuffer(image_bytes, dtype=np.uint8), cv2.IMREAD_COLOR)
    if frame is None:
        return render_template_string(
            PAGE, error="The uploaded file is not a valid image."
        ), 400

    face_count = detect_faces(frame)
    object_count = detect_objects(frame)
    success, encoded_image = cv2.imencode(".jpg", frame)
    if not success:
        raise RuntimeError("OpenCV could not encode the detection result.")

    image_data = base64.b64encode(encoded_image).decode("ascii")
    return render_template_string(
        PAGE,
        image_data=image_data,
        face_count=face_count,
        object_count=object_count,
    )


@app.errorhandler(413)
def upload_too_large(_error):
    return render_template_string(
        PAGE, error="The image exceeds the 10 MB upload limit."
    ), 413


if __name__ == "__main__":
    app.run(
        host="0.0.0.0",
        port=int(os.environ.get("PORT", "10000")),
        debug=False,
    )
