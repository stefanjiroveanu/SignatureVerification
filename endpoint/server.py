from flask import Flask, request, jsonify
from training.recognition import compute_dataset, find_most_similar_signature, compute_phog
import numpy as np
import cv2
import base64
from flask_cors import CORS

app = Flask(__name__)
CORS(app)


@app.post("/api/image")
def find_ten_most_similar_signature():
    if 'file' not in request.files:
        return jsonify({'error': 'No file part'}), 400
    file = request.files['file']

    if file.filename == '':
        return jsonify({'error': 'No selected file'}), 400

    if file and allowed_file(file.filename):
        in_memory_file = np.fromstring(file.read(), np.uint8)
        img = cv2.imdecode(in_memory_file, cv2.IMREAD_GRAYSCALE)
        img = cv2.resize(img, (50, 25), interpolation=cv2.INTER_AREA)
        _, img = cv2.threshold(img, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

        processed_image = compute_phog(img)
        processed_image = np.array(processed_image).reshape(1, -1)
        dataset_features, dataset_labels, image_paths = compute_dataset()
        most_similar_images = find_most_similar_signature(processed_image, dataset_features, image_paths)
        encoded_images = []
        for image_path in most_similar_images:
            with open(image_path[0], "rb") as image_file:
                encoded_string = base64.b64encode(image_file.read()).decode('utf-8')
                encoded_images.append({"image": encoded_string, "similarity": image_path[1]})

        return jsonify(encoded_images)


def allowed_file(filename):
    return '.' in filename and \
        filename.rsplit('.', 1)[1].lower() in {'png', 'jpg', 'jpeg', 'gif'}


if __name__ == "__main__":
    app.run(debug=True, host='0.0.0.0', port=5000)
