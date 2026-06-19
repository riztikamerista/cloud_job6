from flask import Flask, render_template, request
import pickle
import numpy as np
import pandas as pd

app = Flask(__name__)

# Load model dan scaler
# Berdasarkan dokumen, ada list model (Decision Tree dan SVC)
models = pickle.load(open('model.pkl', 'rb'))
scaler = pickle.load(open('scaler.pkl', 'rb'))

model_names = ['Decision Tree', 'SVC']

@app.route("/")
def home():
    return render_template('index.html', model_names=model_names)

@app.route('/predict', methods=['POST'])
def predict():
    if request.method == 'POST':
        # Mengambil data dari form
        pregnancies = int(request.form['pregnancies'])
        glucose = int(request.form['glucose'])
        bloodpressure = int(request.form['bloodpressure'])
        skinthickness = int(request.form['skinthickness'])
        insulin = int(request.form['insulin'])
        bmi = float(request.form['bmi'])
        diabetespedigree = float(request.form['diabetespedigreefunction'])
        age = int(request.form['age'])
        
        # Pilihan model dari dropdown
        selected_model_name = request.form['model']
        
        # Format data menjadi DataFrame sesuai dengan saat training
        data = pd.DataFrame([[pregnancies, glucose, bloodpressure, skinthickness, insulin, bmi, diabetespedigree, age]], 
                            columns=['Pregnancies', 'Glucose', 'BloodPressure', 'SkinThickness', 'Insulin', 'BMI', 'DiabetesPedigreeFunction', 'Age'])
        
        # Lakukan scaling
        data_scaled = scaler.transform(data)
        
        # Pilih model berdasarkan input
        model_index = model_names.index(selected_model_name)
        model = models[model_index]
        
        # Melakukan prediksi
        prediction = model.predict(data_scaled)
        
        # Menentukan output
        if prediction[0] == 1:
            output = "Diabetic"
        else:
            output = "Non-Diabetic"
            
        return render_template('index.html', prediction_text=output, model_names=model_names)

if __name__ == "__main__":
    app.run(debug=True)