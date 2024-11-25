from flask import Flask, request, jsonify, render_template
from joblib import load
import pandas as pd
import os

app = Flask(__name__)

# Increase maximum file upload size to 50 MB
app.config['MAX_CONTENT_LENGTH'] = 50 * 1024 * 1024  # 50 MB

# Create a directory named 'uploads' in the same directory as your Flask app if it doesn't exist
UPLOAD_FOLDER = os.path.join(os.getcwd(), 'uploads')
if not os.path.exists(UPLOAD_FOLDER):
    os.makedirs(UPLOAD_FOLDER)

# Load the trained model
loaded_model = load('gradient_boosting_model.joblib')

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    file = request.files['file']  # Get the uploaded file
    app.logger.info('File received')
    if file and file.filename.endswith('.csv'):
        file_path = os.path.join(UPLOAD_FOLDER, file.filename)
        file.save(file_path)
        app.logger.info(f'File saved at {file_path}')
        try:
            new_data = pd.read_csv(file_path)  # Load the file into a DataFrame
            app.logger.info('File loaded into DataFrame')
            predictions = loaded_model.predict(new_data)  # Make predictions
            app.logger.info(f'Predictions made: {predictions}')
            new_data['Predictions'] = predictions  # Add predictions to DataFrame
            response_html = new_data.to_html()  # Convert DataFrame to HTML
            app.logger.info('Returning predictions as HTML')
            return response_html  # Return predictions as HTML
        except Exception as e:
            app.logger.error(f'Error during prediction: {e}')
            return jsonify({'error': str(e)}), 500
    else:
        app.logger.error('Invalid file format. Please upload a CSV file.')
        return jsonify({'error': 'Invalid file format. Please upload a CSV file.'}), 400

if __name__ == '__main__':
    app.run(debug=True)
