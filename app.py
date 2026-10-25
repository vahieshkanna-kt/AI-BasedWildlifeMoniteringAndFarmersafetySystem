import os
import sqlite3
from datetime import datetime
from flask import Flask, render_template, request, redirect, url_for, flash
from werkzeug.utils import secure_filename
from ultralytics import YOLO
import cv2
from twilio.rest import Client

app = Flask(__name__)
app.secret_key = "wildlife_safety_secret_key"

# Configuration
UPLOAD_FOLDER = 'static/uploads'
PROCESSED_FOLDER = 'static/processed'
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg'}

app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['PROCESSED_FOLDER'] = PROCESSED_FOLDER

os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(PROCESSED_FOLDER, exist_ok=True)

# Load Pretrained YOLOv8 Model
model = YOLO('yolov8n.pt')

# --- TWILIO SMS CREDENTIALS ---
TWILIO_ACCOUNT_SID = " YOUR_TWILIO_ACCOUNT_SID"
TWILIO_AUTH_TOKEN = "YOUR_TWILIO_AUTH_TOKEN"
TWILIO_PHONE_NUMBER = "+1"      # Your Twilio Trial Number
RECIPIENT_PHONE_NUMBER = "+91"   # Your verified personal mobile number

def send_sms_alert(animal_name, confidence):
    """Sends a real SMS text message to the farmer's mobile phone via Twilio."""
    if "YOUR_ACTUAL" in TWILIO_ACCOUNT_SID:
        print("Twilio credentials not configured. Skipping SMS.")
        return
    
    try:
        client = Client(TWILIO_ACCOUNT_SID, TWILIO_AUTH_TOKEN)
        msg_body = f"WILDLIFE ALERT: {animal_name} detected with {confidence}% confidence. Move to a safe location."
        
        message = client.messages.create(
            body=msg_body,
            from_=TWILIO_PHONE_NUMBER,
            to=RECIPIENT_PHONE_NUMBER
        )
        print(f"Twilio SMS sent successfully! SID: {message.sid}")
    except Exception as e:
        print(f"Failed to send Twilio SMS: {e}")

def init_db():
    conn = sqlite3.connect('wildlife.db')
    cursor = conn.cursor()
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS detections (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            animal_name TEXT NOT NULL,
            confidence REAL NOT NULL,
            image_filename TEXT NOT NULL,
            processed_filename TEXT NOT NULL,
            timestamp TEXT NOT NULL
        )
    ''')
    conn.commit()
    conn.close()

init_db()

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

@app.route('/')
def index():
    conn = sqlite3.connect('wildlife.db')
    cursor = conn.cursor()
    cursor.execute("SELECT COUNT(*) FROM detections")
    total_analyzed = cursor.fetchone()[0]
    
    cursor.execute("SELECT COUNT(*) FROM detections WHERE animal_name != 'No Animal Detected'")
    total_detections = cursor.fetchone()[0]
    
    cursor.execute("SELECT * FROM detections ORDER BY id DESC LIMIT 5")
    recent = cursor.fetchall()
    conn.close()
    
    return render_template('index.html', total_analyzed=total_analyzed, total_detections=total_detections, recent=recent)

@app.route('/detect', methods=['POST'])
def detect():
    if 'file' not in request.files:
        flash('No file part in the request.', 'danger')
        return redirect(url_for('index'))
    
    file = request.files['file']
    if file.filename == '':
        flash('No image selected for uploading.', 'warning')
        return redirect(url_for('index'))
    
    if file and allowed_file(file.filename):
        filename = secure_filename(file.filename)
        timestamp_prefix = datetime.now().strftime('%Y%m%d_%H%M%S_')
        safe_filename = timestamp_prefix + filename
        
        upload_path = os.path.join(app.config['UPLOAD_FOLDER'], safe_filename)
        file.save(upload_path)
        
        try:
            results = model(upload_path)
            res = results[0]
            
            detected_animals = []
            img = cv2.imread(upload_path)
            
            for box in res.boxes:
                cls_id = int(box.cls[0])
                conf = float(box.conf[0])
                class_name = model.names[cls_id]
                
                animal_classes = ['elephant', 'bear', 'horse', 'cow', 'sheep', 'dog', 'cat', 'bird', 'zebra', 'giraffe']
                
                if class_name in animal_classes or conf > 0.4:
                    detected_animals.append((class_name, conf))
                    
                    xyxy = box.xyxy[0].tolist()
                    x1, y1, x2, y2 = map(int, xyxy)
                    cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 2)
                    label = f"{class_name} {conf:.2f}"
                    cv2.putText(img, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
            
            processed_filename = 'proc_' + safe_filename
            processed_path = os.path.join(app.config['PROCESSED_FOLDER'], processed_filename)
            cv2.imwrite(processed_path, img)
            
            if detected_animals:
                top_animal, top_conf = max(detected_animals, key=lambda x: x[1])
                animal_display = top_animal.capitalize()
                confidence_display = round(top_conf * 100, 2)
                
                # TRIGGER TWILIO SMS TEXT MESSAGE TO PHONE
                send_sms_alert(animal_display, confidence_display)
            else:
                animal_display = "No Animal Detected"
                confidence_display = 0.0
                processed_filename = safe_filename
            
            conn = sqlite3.connect('wildlife.db')
            cursor = conn.cursor()
            cursor.execute('''
                INSERT INTO detections (animal_name, confidence, image_filename, processed_filename, timestamp)
                VALUES (?, ?, ?, ?, ?)
            ''', (animal_display, confidence_display, safe_filename, processed_filename, datetime.now().strftime('%Y-%m-%d %H:%M:%S')))
            conn.commit()
            conn.close()
            
            return render_template('result.html', 
                                   animal=animal_display, 
                                   confidence=confidence_display, 
                                   original=safe_filename, 
                                   processed=processed_filename)
            
        except Exception as e:
            flash(f'Model inference error: {str(e)}', 'danger')
            return redirect(url_for('index'))
            
    else:
        flash('Invalid file format. Please upload PNG, JPG, or JPEG images only.', 'danger')
        return redirect(url_for('index'))

@app.route('/history')
def history():
    conn = sqlite3.connect('wildlife.db')
    cursor = conn.cursor()
    cursor.execute("SELECT * FROM detections ORDER BY id DESC")
    records = cursor.fetchall()
    conn.close()
    return render_template('history.html', records=records)

@app.route('/about')
def about():
    return render_template('about.html')

if __name__ == '__main__':
    app.run(debug=False)