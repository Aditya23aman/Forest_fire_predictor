from flask import Flask, request, jsonify, render_template
import numpy as np
import pandas as pd
# from sklearn.preprocessing import StandardScaler
import pickle

app = Flask(__name__)

#import model and scaler from pickle
model = pickle.load(open('models/ridgecv.pkl','rb'))
scalar = pickle.load(open('models/scaler.pkl', 'rb'))


#Route for homepage
@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predictdata',methods = ['GET','POST'])
def predict_datapoint():
    if request.method=='POST':
        Temperature=float(request.form.get('Temperature'))
        RH = float(request.form.get('RH'))
        Ws = float(request.form.get('Ws'))
        Rain = float(request.form.get('Rain'))
        FFMC = float(request.form.get('FFMC'))
        DC = float(request.form.get('DC'))
        ISI = float(request.form.get('ISI'))
        # BUI = float(request.form.get('BUI'))
        # FWI = float(request.form.get('FWI'))
        Classes = float(request.form.get('Classes'))
        Region = float(request.form.get('Region'))

        new_data_scaled = scalar.transform([[Temperature,RH,Ws,Rain,FFMC,DC,ISI,Classes,Region]])
        result = model.predict(new_data_scaled)

        return render_template('home.html',result=result[0])
    
    else:
        return render_template('home.html')


if __name__=='__main__':
    app.run(host="0.0.0.0")
        