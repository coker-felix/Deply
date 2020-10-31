from flask import Flask
from flask import request
from deply.predict import model_predict
from flask import render_template
import os

app = Flask(__name__)
app.config['UPLOAD_PATH'] = 'uploads'

@app.route('/')
def hello_world():
    return render_template('home.html')


@app.route('/predict', methods=['GET', 'POST'])
def predict():
    if request.method == 'POST':
        image_file = request.files['file']
        if image_file.filename != '':
            image_file.save(os.path.join(app.config['UPLOAD_PATH'], image_file.filename))
    
    img_path = app.config['UPLOAD_PATH'] + '/' + image_file.filename
    prediction_text = model_predict(img_path)
    return render_template('predict.html', prediction_text=prediction_text)
