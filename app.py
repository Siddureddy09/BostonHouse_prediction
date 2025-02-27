import pickle
from flask import Flask,request,app,jsonify,url_for,render_template
import numpy as np
import pandas as pd
import json

app=Flask(__name__)

# load model
model=pickle.load(open('regmodel.pkl','rb'))
sclar=pickle.load(open('scaling.pkl','rb'))

@app.route('/')
def home():
    return render_template('home.html')

@app.route('/predict_api',methods=['POST'])
def predict_api():
    data=request.json['data']
    print(data) #it is dictionary
    print(np.array(list(data.values())).reshape(1,-1)) # reshape(1,-1) -> reshape to 2 dim (it is single feature)
    new_data=sclar.transform(np.array(list(data.values())).reshape(1,-1))
    output=model.predict(new_data)
    print(output[0]) # as the output os 2 dimensional array
    return jsonify(output[0][0])

@app.route('/predict',methods=['POST'])
def predict():
    data=[ float(x) for x in request.form.values()]
    new_data=sclar.transform(np.array(list(data)).reshape(1,-1))
    output=model.predict(new_data)[0][0]
    return render_template('home.html',prediction_text="The House price prediction in Boston is :  {}".format(output))

if __name__=="__main__":
    app.run(debug=True)