from flask import Flask, request, jsonify, render_template, send_from_directory
from flask_cors import CORS
import os
from werkzeug.utils import secure_filename
from fabric_matcher import extract_features, store_features, get_top_matches, FABRIC_DIR

app = Flask(__name__, template_folder='templates')
CORS(app)  # Enable CORS for all routes
UPLOAD_FOLDER = FABRIC_DIR
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg'}

app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

@app.route('/')
def serve_ui():
    return render_template('index.html')

@app.route('/api/')
def index():
    return jsonify({
        'message': 'Welcome to the Fabric Image Matcher API',
        'endpoints': {
            '/api/upload': 'POST - Upload a new fabric image',
            '/api/match': 'POST - Match an image against the database'
        },
        'status': 'running'
    }), 200

@app.route('/api/upload', methods=['POST'])
def upload_image():
    if 'file' not in request.files:
        return jsonify({'error': 'No file part'}), 400
    file = request.files['file']
    if file.filename == '':
        return jsonify({'error': 'No selected file'}), 400
    if file and allowed_file(file.filename):
        filename = secure_filename(file.filename)
        file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(file_path)
        
        features = store_features(file_path)
        return jsonify({
            'message': 'Image uploaded and features stored',
            'image_path': file_path
        }), 200
    return jsonify({'error': 'Invalid file type'}), 400

@app.route('/api/match', methods=['POST'])
def match_image():
    if 'file' not in request.files:
        return jsonify({'error': 'No file part'}), 400
    file = request.files['file']
    if file.filename == '':
        return jsonify({'error': 'No selected file'}), 400
    if file and allowed_file(file.filename):
        filename = secure_filename(file.filename)
        file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(file_path)
        
        features = extract_features(file_path)
        matches = get_top_matches(features)
        
        results = [
            {'image_path': path, 'similarity': float(score)}
            for path, score in matches
        ]
        return jsonify({'matches': results}), 200
    return jsonify({'error': 'Invalid file type'}), 400

@app.route('/fabrics/<path:filename>')
def serve_fabrics(filename):
    return send_from_directory(app.config['UPLOAD_FOLDER'], filename)

if __name__ == '__main__':
    if not os.path.exists(UPLOAD_FOLDER):
        os.makedirs(UPLOAD_FOLDER)
    app.run(debug=True, host='0.0.0.0', port=5000)