# app.py
from flask import Flask, render_template, request, jsonify, make_response
import os
from werkzeug.utils import secure_filename
from algorithms.text_processor import process_files, compare_two_files, preprocess_text
from algorithms.file_handler import extract_text_from_file

app = Flask(__name__)
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16MB max file size
app.config['UPLOAD_FOLDER'] = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'uploads')
ALLOWED_EXTENSIONS = {'txt', 'pdf', 'doc', 'docx'}

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

def save_file(file):
    if file and allowed_file(file.filename):
        filename = secure_filename(file.filename)
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(filepath)
        return filepath
    return None

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/compare', methods=['POST'])
def compare():
    try:
        # Vérification de la présence des fichiers
        if 'files[]' not in request.files:
            return jsonify({'error': 'Aucun fichier téléchargé'}), 400
        
        files = request.files.getlist('files[]')
        if len(files) < 2:
            return jsonify({'error': 'Veuillez télécharger au moins 2 fichiers'}), 400

        # Vérification des fichiers vides
        non_empty_files = [f for f in files if f.filename]
        if len(non_empty_files) < 2:
            return jsonify({'error': 'Veuillez télécharger au moins 2 fichiers non vides'}), 400

        # Vérification des extensions
        invalid_files = [f.filename for f in non_empty_files if not allowed_file(f.filename)]
        if invalid_files:
            return jsonify({'error': f'Format(s) de fichier non autorisé(s): {", ".join(invalid_files)}'}), 400

        # Comparaison directe des fichiers sans sauvegarde
        if len(non_empty_files) == 2:
            result = compare_two_files(non_empty_files[0], non_empty_files[1])
        else:
            result = process_files(non_empty_files)

        return jsonify(result)

    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.errorhandler(413)
def request_entity_too_large(error):
    return jsonify({'error': 'La taille du fichier dépasse la limite autorisée (16MB)'}), 413

if __name__ == '__main__':
    os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)
    app.run(debug=True)
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port)
