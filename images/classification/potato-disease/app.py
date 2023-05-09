from flask import Flask, redirect, url_for, render_template, request, Response
import matplotlib.pyplot as plt
from PIL import Image
import numpy as np
import cv2
from predict import predict_image
app = Flask(__name__)

@app.route('/predict', methods=['GET', 'POST'])
def predict():
    if request.method == 'GET':
        return render_template('index.html')
    else:
        image_file = request.files['image']
        try:
            image = Image.open(image_file).convert('RGB')
            # convert image to bgr method 1
            # r, g, b = image.split()
            # bgr_image = Image.merge('RGB', (b,g,r))

            # # method 2
            im_arr= cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)
            bgr_image = Image.fromarray(im_arr)
            image = np.array(bgr_image)
        except Exception as e:
            return render_template('error.html', error=e)
        
        conf, predicted_label, predict_class= predict_image(image)
        # return Response(img_bytes.getvalue(), mimetype='image/png')
        return render_template('index.html',  label=predicted_label, predict_class=predict_class)

@app.route('/', methods=['GET'])
def root():
    return redirect(url_for('predict'))

if __name__=="__main__":
    app.run(debug=True)