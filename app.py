from flask import Flask, request, jsonify
from flask_cors import CORS
from flask_cors import cross_origin
import boto3
from io import BytesIO
from PIL import Image
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
import numpy as np
import os
import tempfile
import logging

app = Flask(__name__)
CORS(app)

@app.route('/')
def index():
    return('Bienvenido')

# Configurar logging para obtener más detalles de los errores
logging.basicConfig(level=logging.INFO)

# Configurar el cliente de S3 usando credenciales del entorno
s3 = boto3.client('s3',                  
    aws_access_key_id=os.getenv('AWS_ACCESS_KEY_ID'),
    aws_secret_access_key=os.getenv('AWS_SECRET_ACCESS_KEY'),
    region_name=os.getenv('AWS_REGION'),
    use_ssl=True 
)

BUCKET_NAME = 'b4rb3r'
MODEL_KEY = 'face_shape_model.h5'

def download_and_load_model():
    try:
        logging.info("Intentando descargar el modelo desde S3.")
        model_file = BytesIO()
        s3.download_fileobj(BUCKET_NAME, MODEL_KEY, model_file)
        model_file.seek(0)
        logging.info("Modelo descargado con éxito.")

        # Guardar el archivo temporalmente
        with tempfile.NamedTemporaryFile(delete=False, suffix='.h5') as temp_model_file:
            temp_model_file.write(model_file.getbuffer())
            temp_model_path = temp_model_file.name
            logging.info(f"Modelo guardado temporalmente en {temp_model_path}.")

        # Cargar el modelo en el formato .h5
        model = load_model(temp_model_path)
        logging.info("Modelo cargado con éxito.")
        return model
    except Exception as e:
        logging.error(f"Error al cargar el modelo: {e}", exc_info=True)
        raise

@app.route('/predict-camera', methods=['POST'])
@cross_origin()
def predict():
    # Cargar el modelo solo cuando se llama a la ruta de predicción
    try:
        model = download_and_load_model()
    except Exception as e:
        return jsonify({"error": "Error al cargar el modelo"}), 500
    
    # Definimos un diccionario para mapear los índices a nombres de rostros
    class_mapping = {
        0: 'Rostro cuadrado',
        1: 'Rostro ovalado',
        2: 'Rostro redondo',
        3: 'Rostro triangular'
    }
    
    try:
        file = request.files.get('file')
        if file is None:
            logging.error("No se ha proporcionado ningún archivo.")
            return jsonify({"error": "No se ha proporcionado ningún archivo"}), 400

        img = Image.open(file.stream).convert('RGB')
        
        # Preprocesar la imagen para Keras
        img = img.resize((224, 224))  # Redimensionar
        img_array = image.img_to_array(img)
        img_array = np.expand_dims(img_array, axis=0)
        img_array /= 255.0

        # Realizar la inferencia
        prediction = model.predict(img_array)
        predicted_class_index = np.argmax(prediction, axis=1)
        predicted_face_shape = class_mapping[predicted_class_index[0]]

        return jsonify({"face_shape": predicted_face_shape})
    
    except Exception as e:
        logging.error(f"Error durante la predicción: {e}", exc_info=True)
        return jsonify({"error": str(e)}), 500
    
@app.route('/predict-gallery', methods=['POST'])
@cross_origin()
def predict_galery():
    # Cargar el modelo solo cuando se llama a la ruta de predicción
    try:
        model = download_and_load_model()
    except Exception as e:
        return jsonify({"error": "Error al cargar el modelo"}), 500
    
    # Definimos un diccionario para mapear los índices a nombres de rostros
    class_mapping = {
        0: 'Rostro cuadrado',
        1: 'Rostro ovalado',
        2: 'Rostro redondo',
        3: 'Rostro triangular'
    }
    
    try:
        file = request.files.get('file')
        if file is None:
            logging.error("No se han proporcionado archivos.")
            return jsonify({"error": "No se han proporcionado archivos"}), 400

        img = Image.open(file.stream).convert('RGB')

        # Preprocesar la imagen para Keras
        img = img.resize((224, 224))  # Redimensionar
        img_array = image.img_to_array(img)
        img_array = np.expand_dims(img_array, axis=0)
        img_array /= 255.0

        # Realizar la inferencia
        prediction = model.predict(img_array)
        predicted_class_index = np.argmax(prediction, axis=1)
        predicted_face_shape = class_mapping[predicted_class_index[0]]

        return jsonify({"face_shape": predicted_face_shape})
    
    except Exception as e:
        logging.error(f"Error durante la predicción: {e}", exc_info=True)
        return jsonify({"error": str(e)}), 500

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=8080)