from flask import Flask, render_template, request, jsonify
import os
import cv2
import numpy as np
import sqlite3
from datetime import datetime
import torch
import segmentation_models_pytorch as smp

app = Flask(__name__)

UPLOAD_FOLDER = 'static/uploads'
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

print("Инициализация нейросети U-Net (PyTorch)...")
model = smp.Unet(
    encoder_name="resnet34",
    encoder_weights="imagenet",
    in_channels=3,
    classes=1,
)
model.eval()
print("Нейросеть успешно загружена!")

def init_db():
    conn = sqlite3.connect('history.db')
    cursor = conn.cursor()
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS requests (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            timestamp TEXT,
            tumor_area TEXT,
            result_image TEXT
        )
    ''')
    conn.commit()
    conn.close()

init_db()

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/process', methods=['POST'])
def process_image():
    if 'image' not in request.files:
        return jsonify({'error': 'Нет файла с изображением'}), 400
    
    file = request.files['image']
    if file.filename == '':
        return jsonify({'error': 'Файл не выбран'}), 400

    img_bytes = file.read()
    np_img = np.frombuffer(img_bytes, np.uint8)
    img = cv2.imdecode(np_img, cv2.IMREAD_COLOR)
    h, w, _ = img.shape

    # --- ИНФЕРЕНС НЕЙРОСЕТИ U-NET ---
    img_resized = cv2.resize(img, (256, 256))
    tensor_img = torch.from_numpy(img_resized).permute(2, 0, 1).float() / 255.0
    tensor_img = tensor_img.unsqueeze(0)
    
    with torch.no_grad():
        pred_mask = model(tensor_img)
        pred_mask = torch.sigmoid(pred_mask) > 0.5
        pred_mask = pred_mask.squeeze().cpu().numpy().astype(np.uint8) * 255

    pred_mask_resized = cv2.resize(pred_mask, (w, h), interpolation=cv2.INTER_NEAREST)

    # --- АНАЛИЗ ИЗОБРАЖЕНИЯ ---
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    _, thresh = cv2.threshold(blurred, 150, 255, cv2.THRESH_BINARY)

    combined_mask = cv2.bitwise_and(thresh, thresh, mask=pred_mask_resized)
    if cv2.countNonZero(combined_mask) < 50:
        combined_mask = thresh

    contours, _ = cv2.findContours(combined_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    output_img = img.copy()
    total_tumor_pixels = 0
    valid_contours = []

    for cnt in contours:
        area = cv2.contourArea(cnt)
        x, y, cw, ch = cv2.boundingRect(cnt)
        is_touching_border = (x < 10 or y < 10 or (x + cw) > w - 10 or (y + ch) > h - 10)

        perimeter = cv2.arcLength(cnt, True)
        circularity = (4 * np.pi * area) / (perimeter ** 2) if perimeter > 0 else 0

        if 100 < area < 15000 and not is_touching_border and circularity > 0.2:
            valid_contours.append(cnt)
            total_tumor_pixels += int(area)

    # --- РАСЧЕТ ПЛОЩАДИ В КВАДРАТНЫХ САНТИМЕТРАХ ---
    fov_mm = 230.0 
    cm_per_pixel = (fov_mm / w) / 10.0 # размер одного пикселя в сантиметрах
    tumor_area_cm2 = total_tumor_pixels * (cm_per_pixel ** 2)

    if valid_contours:
        for cnt in valid_contours:
            cv2.drawContours(output_img, [cnt], -1, (0, 0, 255), 2)
            
            overlay = output_img.copy()
            cv2.fillPoly(overlay, [cnt], (0, 0, 255))
            alpha = 0.4
            cv2.addWeighted(overlay, alpha, output_img, 1 - alpha, 0, output_img)

    # Сохранение результата
    result_filename = f"result_{datetime.now().strftime('%Y%m%d_%H%M%S')}.jpg"
    result_path = os.path.join(app.config['UPLOAD_FOLDER'], result_filename)
    cv2.imwrite(result_path, output_img)

    conn = sqlite3.connect('history.db')
    cursor = conn.cursor()
    cursor.execute(
        'INSERT INTO requests (timestamp, tumor_area, result_image) VALUES (?, ?, ?)',
        (datetime.now().strftime('%Y-%m-%d %H:%M:%S'), f"{tumor_area_cm2:.2f} см²", result_path)
    )
    conn.commit()
    conn.close()

    return jsonify({
        'status': 'success',
        'area': f"{tumor_area_cm2:.2f} см² ({total_tumor_pixels} пикселей)",
        'image_url': f'/{result_path}'
    })

if __name__ == '__main__':
    app.run(debug=True, port=5000)