from flask import Flask, render_template, request, redirect, url_for, send_from_directory
import os
from datetime import datetime
import time

app = Flask(__name__)

# Configuration
UPLOAD_FOLDER = 'static/visitors'
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

# Store visitor data
visitors = []

@app.route('/')
def index():
    # Sort visitors by timestamp (newest first)
    sorted_visitors = sorted(visitors, key=lambda x: x['timestamp'], reverse=True)
    current_date = datetime.now().strftime("%Y-%m-%d")
    return render_template('index.html', 
                         visitors=sorted_visitors,
                         current_date=current_date,
                         now=datetime.now())

@app.route('/upload', methods=['POST'])
def upload_file():
    if 'image' not in request.files:
        return 'No image file', 400
    
    file = request.files['image']
    if file.filename == '':
        return 'No selected file', 400
    
    if file:
        # Save the file
        timestamp = request.form.get('timestamp', datetime.now().strftime("%Y%m%d_%H%M%S"))
        filename = f"visitor_{timestamp}.jpg"
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(filepath)
        
        # Add to visitors list
        detection_time = request.form.get('detection_time', datetime.now().strftime("%Y-%m-%d %H:%M:%S"))
        visitor_data = {
            'image': filename,
            'timestamp': timestamp,
            'time': detection_time,
            'date': datetime.now().strftime("%Y-%m-%d")
        }
        visitors.append(visitor_data)
        
        return 'File uploaded successfully', 200

@app.route('/static/visitors/<filename>')
def uploaded_file(filename):
    return send_from_directory(app.config['UPLOAD_FOLDER'], filename)

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)