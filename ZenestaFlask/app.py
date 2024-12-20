"""
This script runs the application using a development server.
It contains the definition of routes and views for the application.
"""

import json
from flask import Flask, request, jsonify
from flask_cors import cross_origin

from werkzeug.datastructures import FileStorage
from werkzeug.utils import secure_filename

from ml.model import PredictionModel

app = Flask(__name__)

model = PredictionModel('pretrained.h5')

# Make the WSGI interface available at the top level so wfastcgi can get it.
wsgi_app = app.wsgi_app

@app.route('/')
def hello() -> str:
    """Renders a sample page."""
    return "Hello World!"

@app.route('/setup')
def setup() -> json:
    try:
        PredictionModel.create_pretrained()
        return jsonify({ 'result': 'Success' })
    except Exception as err:
        return jsonify({ 'result': 'Error', 'error': err })

@app.route('/predict', methods=['POST'])
@cross_origin()
def predict() -> json:
    if request.method == 'POST':
        image_file: FileStorage = request.files['image']

        # file_name = secure_filename(image_file.filename)

        # image_file.save(f"./{file_name}")

        prediction_result: dict = model.predict(image_file)

        response_body: list = []
        for result in prediction_result:
            result_dict: dict = {}
            # result_dict['id'] = result[0]
            result_dict['object'] = result[1]
            result_dict['confidence'] = str(result[2])
            response_body.append(result_dict)

        return json.dumps({ 'prediction_result' : response_body });

if __name__ == '__main__':
    import os
    HOST = os.environ.get('SERVER_HOST', 'localhost')
    try:
        PORT = int(os.environ.get('SERVER_PORT', '6060'))
    except ValueError:
        PORT = 6060
    app.run(HOST, PORT)