from tensorflow.keras.models import load_model
import numpy as np
from flask import Flask, request, jsonify, send_from_directory, json
from werkzeug.utils import secure_filename
from flask_cors import CORS, cross_origin
import os
from PIL import Image, ImageFilter
from io import BytesIO

os.environ['KMP_DUPLICATE_LIB_OK']='True'

class_names=['top','trouser','pullover','dress','coat','sandal','shirt','sneaker','bag','ankle boot']
ALLOWED_EXTENSIONS = set(['png'])
IMAGE_WIDTH = 28.0
IMAGE_HEIGHT = 28.0

# instantiate flask app:
app = Flask(__name__)
CORS(app)

def imageprepare(im):
    """
    This function reuturn the pixel values.
    The input is a png file Image object
    """

    width = float(im.size[0])
    height = float(im.size[1])
    new_image =Image.new('L', (int(IMAGE_WIDTH) ,int(IMAGE_HEIGHT)), (0))
    if(width>height):
        # Width is bigger, width becomes 28 pixels
        nheight = int(round((IMAGE_WIDTH /width *height), 0)) # resize height according to ratio
        if(nheight==0): # rare case but min is 1 pxel
            nheight =1
            # resize and sharpen
        img =im.resize((int(IMAGE_WIDTH), nheight), Image.ANTIALIAS).filter(ImageFilter.SHARPEN)
        wtop = int(round(((int(IMAGE_HEIGHT)-nheight ) /2), 0)) # calc vertical position
        new_image.paste(img, (0, wtop)) # paste resized image onto white canvas
    elif(height>width):
        # Hieght is bigger. Height becomes 28 pixels
        nwidth = int(round((IMAGE_WIDTH/height*width), 0))
        if(nwidth==0):
            nwidth =1
        img = im.resize((nwidth, int(IMAGE_WIDTH)), Image.ANTIALIAS).filter(ImageFilter.SHARPEN)
        wleft = int(round(((int(IMAGE_WIDTH)-nwidth ) /2), 0))
        new_image.paste(img, (wleft, 0))
    else:
        nheight = int(round((IMAGE_HEIGHT / width * height), 0))  # resize height according to ratio
        if (nheight == 0):  # rare case but min is 1 pxel
            nheight = 1
        # resize and sharpen
        img = im.resize((int(IMAGE_WIDTH), nheight), Image.ANTIALIAS).filter(ImageFilter.SHARPEN)
        new_image.paste(img, (0, 0))

    tv = list(new_image.getdata()) # get pixel values
    return tv


def allowed_file(filename):
    return '.' in filename and \
           filename.rsplit('.', 1)[1] in ALLOWED_EXTENSIONS


@app.route('/api/images', methods=['GET'])
def get_all_images():
    res = []
    for fileName in os.listdir('./images'):
        if fileName.endswith(('.png')):
            res.append({'fileName': fileName})
    body = {}
    body["items"] = res
    resp = app.response_class(
        response=json.dumps(body),
        status=200,
        mimetype='application/json')
    return resp


# @app.route('/api/imagesAttached/<path:filename>', methods=['GET'])
# def send_route_attach(filename):
#     return send_from_directory('./images', filename, as_attachment=True)


@app.route('/api/images/<path:filename>', methods=['GET'])
def send_route(filename):
    if not filename:
        return (jsonify({'error': 'Invalid request.'}), 400)
    filename = secure_filename(filename)
    if allowed_file(filename):
        resp = send_from_directory('./images', filename, mimetype='image/png', cache_timeout=30*60)
        return resp
    else:
        return (jsonify({'error': 'Invalid request.'}), 400)


@app.route('/api/recogImage/<path:filename>', methods=['GET'])
def recog_image(filename):
    if not filename:
        #print("/api/image POST -- ERROR: Invalid request.")
        # 400 - Bad request
        return (jsonify({'error': 'Invalid request.'}), 400)
    filename = secure_filename(filename)
    if allowed_file(filename):
        if (not os.path.exists("./images/" + filename)):
            # 400 - Bad request
            #print("/api/image POST -- ERROR: invalid image ;" + filename + ";")
            return (jsonify({'error': 'Invalid request.'}), 400)
        #print("/api/image POST -- file name is good...")
        try:
            img = Image.open("./images/"+filename)
            x = imageprepare(img)  # x is 1d list 784

            # model is trained with 28x28=784 pixels, values (0.0 - 1.0)
            # convert 1d list x[] of 784 pixels with values (1 - 255) to
            # mnistInputArr 3d numpy array 1x28x28 with values (0.0 - 1.0)
            mnist_input_arr = np.zeros((1, int(IMAGE_WIDTH), int(IMAGE_HEIGHT)))
            k = 0
            for i in range(int(IMAGE_WIDTH)):
                for j in range(int(IMAGE_HEIGHT)):
                    mnist_input_arr[0][i][j] = x[k] / 255.0
                    k = k + 1

            pred = modelFromFile.predict(mnist_input_arr)
            #print("/api/image POST -- after the prediction: ", pred)
            items = []
            items.append({'item': class_names[np.argmax(pred[0])], 'probability': float(pred[0][np.argmax(pred[0])])})
            response = {'predictions': items}
            #print("/api/image POST -- end: ", items)
            return (jsonify(response), 200)
        except Exception:
            # 400 - Bad request
            #print("/api/image POST -- ERROR: invalid request ;" + filename + ";")
            return (jsonify({'error': 'Invalid request'}), 400)
    else:
        #print("/api/image POST -- file has invalid extension")
        # 422- Unprocessable Entity
        return (jsonify({'error': 'File has invalid extension'}), 422)


@app.route('/api/image', methods=['POST'])
@cross_origin(supports_credentials=True)
def recognize_image():
    """Accepts arbitrary PNG image file to recognize.
       Not used in current implementation of the client
    """
    #print("/api/image POST -- start --")
    # check if the post request has the file part
    if 'image' not in request.files:
        #print("/api/image POST -- ERROR: No posted image. Should be attribute named image.")
        # 400 - Bad request
        return (jsonify({'ERROR': 'No posted image. Should be attribute named image.'}), 400)
    file = request.files['image']

    # user did not select file
    if file and file.filename == '':
        #print("/api/image POST -- ERROR: Empty filename submitted.")
        # 400 - Bad request
        return (jsonify({'error': 'Empty filename submitted.'}), 400)
    if file and allowed_file(file.filename):
        #print("/api/image POST -- file is good...")
        filename = secure_filename(file.filename)
        #print("processing input file:" + filename)
        img = Image.open(BytesIO(file.read()))
        img.load()
        x = imageprepare(img) # x is 1d list 784

        # model is trained with 28x28=784 pixels, values (0.0 - 1.0)
        # convert 1d list x[] of 784 pixels with values (1 - 255) to
        # mnistInputArr 3d numpy array 1x28x28 with values (0.0 - 1.0)
        mnist_input_arr = np.zeros((1, int(IMAGE_WIDTH), int(IMAGE_HEIGHT)))
        k = 0
        for i in range(int(IMAGE_WIDTH)):
            for j in range(int(IMAGE_HEIGHT)):
                mnist_input_arr[0][i][j] = x[k] / 255.0
                k = k + 1

        pred = modelFromFile.predict(mnist_input_arr)
        #print("/api/image POST -- after the prediction: ", pred)
        items = []
        items.append( {'item': class_names[np.argmax(pred[0])], 'probability': float(pred[0][np.argmax(pred[0])])} )
        response = {'predictions': items}
        #print("/api/image POST -- end: ", items)
        return(jsonify(response), 200)
    else:
        #print("/api/image POST -- file has invalid extension")
        # 422- Unprocessable Entity
        return (jsonify({'error': 'File has invalid extension'}), 422)


# define routes using route decorator:
@app.route('/')
@app.route('/home')
def home():
    return f'<h3>Home page!</h3>'


# Load trained model. It is trained using a school example of cloth items images:
# - 60,000 training images
# - 10,000 test images
# See https://github.com/zalandoresearch/fashion-mnist
modelFromFile = load_model(os.path.join("./trainResults", "clothesModel01.h5"))


# NOTE: keras library is not loading if the Flask is run in DEBUG mode
# See: https://github.com/tensorflow/tensorflow/issues/34607
if __name__ == '__main__':
    app.run()