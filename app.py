import pickle
from flask import Flask,request,app,jsonify,url_for,render_template
import numpy as np
import pandas as pd

app = Flask(__name__)
model = pickle.load(open('regmodel.pkl','rb'))
scaler = pickle.load(open('scaling.pkl','rb'))

@app.route('/')
def home():
    return render_template('home.html')

@app.route('/predict_api',methods=['POST'])
def predict_api():
    data = request.json['data']
    new_data = scaler.transform(np.array(list(data.values())).reshape(1,-1))
    output = model.predict(new_data)
    return jsonify(output[0])

@app.route('/predict',methods=['POST'])
def predict():
    data = [float(x) for x in request.form.values()]
    x = scaler.transform(np.array(data).reshape(1,-1))
    y_pred = model.predict(x)[0]
    return render_template("home.html",prediction_text="The house price is {}".format(y_pred))

if __name__ == '__main__':
    app.run(debug=True)
    