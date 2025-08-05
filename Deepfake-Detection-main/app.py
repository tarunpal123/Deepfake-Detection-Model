import os
import numpy as np
from PIL import Image
from flask import Flask, request, render_template
from werkzeug.utils import secure_filename
import tensorflow as tf
from keras.models import load_model

model = load_model("Model_version_3.0.h5")

app = Flask(__name__)

print('Model loaded. Check http://127.0.0.1:5000/')

ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg', 'gif'}

upload_folder = os.path.join(os.path.dirname(__file__), 'uploads')
if not os.path.exists(upload_folder):
    os.makedirs(upload_folder)

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

def get_className(classNo):
    if classNo == 0:
        return "Image Without Deepfake"
    elif classNo == 1:
        return "Image With Deepfake"

def getResult(img):
    try:
        image = Image.open(img)
    except Exception as e:
        print(f"Error opening image: {e}")
        return None

    image = image.resize((128, 128))
    image = np.array(image)

    if image.shape != (128, 128, 3):  # Ensure the image has the expected shape
        print(f"Unexpected image shape: {image.shape}. Expected (128, 128, 3)")
        return None

    image = image / 255.0
    input_img = np.expand_dims(image, axis=0)

    try:
        result = model.predict(input_img)
        result01 = np.argmax(result, axis=1)
        return result01
    except Exception as e:
        print(f"Error during prediction: {e}")
        return None

@app.route('/', methods=['GET'])
def index():
    return render_template('index.html')

@app.route('/predict', methods=['GET', 'POST'])
def upload():
    if request.method == 'POST':
        f = request.files['file']
        
        if f and allowed_file(f.filename):
            basepath = os.path.dirname(__file__)
            file_path = os.path.join(basepath, 'uploads', secure_filename(f.filename))
            f.save(file_path)

            value = getResult(file_path)
            if value is None:
                return "Error in image processing or prediction."

            result = get_className(value)
            return result
        else:
            return "Invalid file format. Only image files are allowed."

    return None

if __name__ == '__main__':
    app.run(debug=True)

