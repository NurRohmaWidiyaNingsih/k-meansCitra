import os
import numpy as np
from PIL import Image
from sklearn.cluster import KMeans
from flask import Flask, request, render_template, send_file, url_for
import io

# Inisialisasi Flask
app = Flask(__name__)

# Folder untuk menyimpan file
UPLOAD_FOLDER = 'static/uploads'
SEGMENTED_FOLDER = 'static/segmented'

# Membuat folder jika belum ada
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(SEGMENTED_FOLDER, exist_ok=True)

# Fungsi untuk segmentasi gambar
def segment_image(image_path, n_clusters=3):
    image = Image.open(image_path)
    image = image.resize((100, 100))  # Resize untuk mempercepat proses
    image_array = np.array(image)  # Ubah gambar ke array numpy
    image_array_reshaped = image_array.reshape((-1, 3))  # Format (R, G, B)

    # KMeans clustering
    kmeans = KMeans(n_clusters=n_clusters, random_state=42)
    kmeans.fit(image_array_reshaped)

    # Membuat gambar hasil segmentasi
    segmented_image = kmeans.cluster_centers_[kmeans.labels_]
    segmented_image = segmented_image.reshape(image_array.shape).astype(np.uint8)

    return segmented_image

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/upload', methods=['POST'])
def upload_image():
    if 'file' not in request.files:
        return 'No file part', 400

    file = request.files['file']
    n_clusters = request.form.get('n_clusters', 3)

    # Pastikan file diupload
    if file.filename == '':
        return 'No selected file', 400

    if file:
        # Simpan file yang diupload
        input_image_path = os.path.join(UPLOAD_FOLDER, file.filename)
        file.save(input_image_path)

        # Pastikan input cluster adalah angka
        try:
            n_clusters = int(n_clusters)
        except ValueError:
            return 'Please enter a valid number for clusters.', 400

        # Segmentasi gambar
        segmented_image = segment_image(input_image_path, n_clusters)

        # Simpan hasil segmentasi
        output_image_path = os.path.join(SEGMENTED_FOLDER, f"segmented_{file.filename}")
        Image.fromarray(segmented_image).save(output_image_path)

        # Dapatkan URL gambar hasil segmentasi
        segmented_image_url = url_for('static', filename=f'segmented/{f"segmented_{file.filename}"}')

        # Kembalikan halaman dengan gambar asli dan hasil segmentasi
        return render_template('index.html', original_image=file.filename, segmented_image_url=segmented_image_url)

if __name__ == '__main__':
    app.run(debug=True)
