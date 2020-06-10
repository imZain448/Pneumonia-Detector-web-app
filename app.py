import flask
import os
from werkzeug.utils import secure_filename
from werkzeug.middleware.shared_data import SharedDataMiddleware
import numpy as np
import base64
import cv2
from PIL import Image
import io
import tensorflow as tf
from tensorflow.keras.models import load_model

app  = flask.Flask(__name__)

images_path = "C:/projects/DEPLOYED/Pnuemonia_Detector_web_app/static/uploaded_images"
app.config['UPLOAD_FOLDER'] = images_path

global graph , model
model = load_model("C:\projects\DEPLOYED\Pnuemonia_Detector_web_app\model_1.h5")
graph = tf.compat.v1.get_default_graph()

@app.route('/' , methods = ['POST' , 'GET'])
def upload_image():
    if flask.request.method == 'POST':
        if flask.request.files:
            image = flask.request.files['image']
            print(image)
            f = image.read()
            npimg = np.fromstring(f , np.uint8)
            img = cv2.imdecode(npimg ,cv2.IMREAD_COLOR)
            img_d = cv2.resize(img , dsize = (256 , 256))
            img = Image.fromarray(img_d.astype('uint8'))
            rawBytes = io.BytesIO()
            img.save(rawBytes , "JPEG")
            rawBytes.seek(0)
            img_base64 = base64.b64encode(rawBytes.getvalue()).decode("ascii")
            mime = "image/jpeg"
            uri = "data:%s;base64,%s"%(mime, img_base64)
            # preparing the image to load into the model
            #print(img_d)
            img_d = img_d/255
            img_d = img_d.reshape(1,256,256,3)
            # loading the model
            out = model.predict(img_d)
            y = np.round(out).astype('int')
            #print(y[0,0])
            if y[0,0] == 0:
                result = "NEGATIVE"
            else:
                result = "POSITIVE"
            return flask.render_template("index.html", image= uri , result= result)
    return flask.render_template("index.html")

# @app.route('/uploaded_images/<filename>')
# def uploaded_file(filename):
#     return flask.send_from_directory(app.config['UPLOAD_FOLDER'] , filename)

app.add_url_rule('/uploaded_image/<filename>' , 'uploaded_file' , build_only = True)
app.wsgi_app = SharedDataMiddleware(app.wsgi_app , {
    '/uploaded_images': app.config['UPLOAD_FOLDER']
})

if (__name__== "__main__"):
    app.run(debug=True , port=5000)