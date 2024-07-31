from flask import Flask, request, jsonify, render_template_string
from flask_cors import CORS
import pickle
import numpy as np

app = Flask(__name__)
CORS(app)  # Habilitar CORS para todas las rutas

# Cargar el modelo entrenado
with open('modelo_entrenado.pkl', 'rb') as f:
    model = pickle.load(f)

# Definir los rangos válidos
RANGOS = {
    'Temperatura(H)': (6, 60),
    'Humedad(%)': (20, 90),
    'Luz(Lux)': (1000, 7000),
    'pH_Suelo': (6.0, 7.5),
    'Contenido_Nutrientes(ppm)': (100, 250),
    'CO2': (0.0, 1.0)
}

def esta_dentro_de_rango(valor, rango):
    return rango[0] <= valor <= rango[1]

@app.route('/')
def index():
    # Cargar y devolver el contenido del archivo HTML
    with open('index.html', 'r') as file:
        html_content = file.read()
    return render_template_string(html_content)

@app.route('/predict', methods=['POST'])
def predict():
    data = request.get_json(force=True)
    
    # Verificar si todas las variables están dentro de los rangos válidos
    for variable, rango in RANGOS.items():
        valor = data.get(variable)
        if valor is None or not esta_dentro_de_rango(valor, rango):
            return jsonify({
                'prediccion': 0,
                'estado': 'muere'
            })
    
    # Realizar la predicción si todo está dentro del rango
    features = np.array([
        data['Temperatura(H)'],
        data['Humedad(%)'],
        data['Luz(Lux)'],
        data['pH_Suelo'],
        data['Contenido_Nutrientes(ppm)'],
        data['CO2']
    ]).reshape(1, -1)
    
    prediction = model.predict(features)[0]
    status = 'Planta Muere' if prediction > 50 else 'Planta Vive'
    
    return jsonify({
        'prediccion': prediction,
        'estado': status
    })

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)  # Asegúrate de estar ejecutando en el puerto correcto
