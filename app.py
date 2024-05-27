from flask import Flask,render_template,url_for,request,jsonify
from flask_cors import cross_origin
import pandas as pd
import numpy as np
import datetime
import pickle

app = Flask(__name__, template_folder="template")
try:
    # Load the model from the file
    with open("cat.pkl", "rb") as file:
        model = pickle.load(file)
    print("Model loaded successfully.")
except FileNotFoundError:
    print("File not found. Please check the path and filename.")
except Exception as e:
    print(f"An error occurred while loading the model: {e}")
print("Model Loaded")

@app.route("/",methods=['GET'])
@cross_origin()
def home():
	return render_template("index.html")
@app.route("/predict",methods=['GET', 'POST'])
@cross_origin()
def predict():
    if request.method == "POST":
        # Extracting the form data
        state_ut_name = float(request.form['location'])  # This corresponds to STATE_UT_NAME
        min_temp = float(request.form['mintemp'])        # This corresponds to MinTemp
        max_temp = float(request.form['maxtemp'])        # This corresponds to MaxTemp
        humidity = float(request.form['humidity'])       # This corresponds to Humidity
        pressure = float(request.form['pressure'])       # This corresponds to Pressure

        # Extracting the date to get year, month, and day
        date = request.form['date']
        year = float(pd.to_datetime(date, format="%Y-%m-%d").year)
        month = float(pd.to_datetime(date, format="%Y-%m-%d").month)
        day = float(pd.to_datetime(date, format="%Y-%m-%d").day)
        # Create input list for prediction
        input_lst = [state_ut_name, min_temp, max_temp, humidity, pressure, year, month, day]

        # Reshape input_lst to match the model's expected input
        input_array = np.array(input_lst).reshape(1, -1)

        # Predict using the loaded model
        pred = model.predict(input_array)
        output = pred[0]

        # Render the appropriate template based on the prediction
        if output == 0:
            return render_template("after_sunny.html")
        else:
            return render_template("after_rainy.html")

    return render_template("predictor.html")

if __name__=='__main__':
	app.run(debug=True)