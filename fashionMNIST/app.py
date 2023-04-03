from flask import Flask, render_template, request
from PIL import Image
from predict import predict_image
app = Flask(__name__)

@app.route('/', methods=['GET'])
def upload_page():
    return render_template('index.html')

@app.route('/', methods = ['POST'])
def predict():
    image_file = request.files['image']
    try:
        image = Image.open(image_file)
    except Exception as e:
        return render_template('error.html', error=e)
    
    predicted_label, predict_class = predict_image(image)
    
    return render_template('index.html', label=predicted_label, predict_class=predict_class)

if __name__=="__main__":
    app.run(port=5000, debug=True)