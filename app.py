from flask import Flask, render_template, request
from ultralytics import YOLO
import os

app = Flask(__name__)

UPLOAD_FOLDER = "uploads"
app.config["UPLOAD_FOLDER"] = UPLOAD_FOLDER

os.makedirs(UPLOAD_FOLDER, exist_ok=True)

# Load YOLO model
model = YOLO("yolo11n.pt")


@app.route("/")
def home():
    return render_template("index.html")


@app.route("/detect", methods=["POST"])
def detect():

    if "image" not in request.files:
        return "No image selected"

    image = request.files["image"]

    if image.filename == "":
        return "No image selected"

    image_path = os.path.join(
        app.config["UPLOAD_FOLDER"],
        image.filename
    )

    image.save(image_path)

    # AI detection
    results = model(image_path)

    detections = []

    for result in results:
        for box in result.boxes:
            class_id = int(box.cls[0])
            confidence = float(box.conf[0])
            animal = result.names[class_id]

            detections.append({
                "animal": animal,
                "confidence": round(confidence * 100, 2)
            })

    return render_template(
        "result.html",
        detections=detections,
        image=image.filename
    )


if __name__ == "__main__":
    app.run(debug=False, use_reloader=False)