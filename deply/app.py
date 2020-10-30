from flask import Flask
from flask import request
from .predict import model_predict
from flask import render_template

app = Flask(__name__)

@app.route('/')
def hello_world():
    return render_template('home.html')


@app.route('/predict',methods=['POST'])
def predict():
    if request.method == 'POST':
        # check if the post request has the file part
        if 'file' not in request.files:
            file = request.files['file']
    
    prediction_text = model_predict(file)
    render_template('predict', prediction_text=prediction_text)
