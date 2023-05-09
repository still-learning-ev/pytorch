from flask import Flask, redirect, url_for, render_template, request
from PIL import Image
from predict import predict_image
app = Flask(__name__)

@app.route('/predict', methods=['GET', 'POST'])
def predict():
    if request.method == 'GET':
        return render_template('index.html')
    else:
        image_file = request.files['image']
        try:
            image = Image.open(image_file)
        except Exception as e:
            return render_template('error.html', error=e)
        
        predicted_label, predict_class = predict_image(image)
        return render_template('index.html',  label=predicted_label, predict_class=predict_class)

@app.route('/', methods=['GET'])
def root():
    return redirect(url_for('predict'))

if __name__=="__main__":
    app.run(debug=True)