import numpy as np
from flask import Flask, request,render_template
import os
from sklearn.externals import joblib
import pickle
import flask

app = Flask(__name__)


filename = 'finalized_model.sav'
model = joblib.load(filename)
with open('vectorizer.pickle', 'rb') as handle:
	vectorizer = pickle.load(handle)

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict',methods=['GET','POST'])
def predict():
    temp=request.get_data(as_text=True)
    new=['it is a book']
    message=vectorizer.transform(new)
    pred = model.predict(message)
    if pred == 1:
        return "SPAM"
    else:
        return "HAM"
    return str(pred)
    
if __name__=='__main__':
    app.run(debug=True)
    