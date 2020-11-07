#ShengpingJiang- Face recognition model as a flask application

import pickle
import os
import numpy as np
from flask import Flask, flash, request, redirect, url_for, send_from_directory, render_template
from werkzeug.utils import secure_filename
from PIL import Image
import face_recognition as frg

#model = None
app = Flask(__name__)


def load_model():
    global model
    # model variable refers to the global variable
    with open('face_model_file_frg', 'rb') as f:
        model = pickle.load(f)


@app.route('/')
def home_endpoint():
    return 'Hello World!'


@app.route('/predict', methods=['GET','POST'])
def get_prediction():
    dist_threshold = 0.4
    name=''
    # Works only for a single sample
    if request.method == 'POST':
        data = request.get_json()  # Get data posted as a json
        #data[0] means 1st {} in the JSON data [{..},{..}]. data[0]['encoding'] means 
        #the value of key 'encoding' in data[0]
        #print(type(data[0]['encoding']))
        #print(data[0]['encoding'])
        #The value of the key 'encoding' is a string '[-0.17077433  0.086519...]'
        str1 = data[0]['encoding']
        # str1[1:-1] from '[-0.17077433  0.086519...]' to '-0.17077433  0.086519...'. Remove brackets
        # np.fromstring changes a string '-0.17077433  0.086519...' to a numpy array 
        # [-0.17077433  0.086519...]
        encoding = np.fromstring(str1[1:-1], dtype=float, sep=' ')
        #print("ecoding type:", type(encoding))
        #print(encoding)
        
        # reshape(1,-1) change [-0.17077433  0.086519...] to [[-0.17077433 0.086519 0.04608656...]]
        xt = encoding.reshape(1,-1)
        #print('xt:', xt)
        closest_distance = model.kneighbors(xt, n_neighbors=1, return_distance=True)
        #print("closest_distance[0][0][0]:",closest_distance[0][0][0])
        if closest_distance[0][0][0] <= dist_threshold :
	# model.predict(xt) returns a string list ['name']
	# model.predict(xt)[0] returns 'name'
            name = model.predict(xt)[0]
            print('name:', name)
        else:
            name = "Unknown"
    elif request.method == 'GET':
        print("Shengping")
        
    return name

ALLOWED_EXTENSIONS = ['jpg', 'jpeg', 'gif']
UPLOAD_FOLDER = './uploads'
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

def allowed_file(filename):
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

@app.route('/uploads', methods=['GET', 'POST'])
def upload_file():
    name = ""
    if request.method == 'POST':
        # check if the post request has the file part
        if 'file' not in request.files:
            flash('No file part')
            return redirect(request.url)
        file = request.files['file']
        # if user does not select file, browser also
        # submit an empty part without filename
        if file.filename == '':
            flash('No selected file')
            return redirect(request.url)
        if file and allowed_file(file.filename):
            filename = secure_filename(file.filename)
	#Uncomment below two lines will save uploaded file in './uploads'
            #fpath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            #file.save(fpath)
	#file.stream is a file-like object. And load_image_file() needs filename 
	# or file-like object
            image = frg.load_image_file(file.stream, mode='RGB')
            print("type of image1:", type(image))
            name = predict_file(image)
            return render_template('prediction.html', value=name)

    elif request.method == 'GET':
        print("Shengping")

    return '''
    <!doctype html>
    <title>Upload new File</title>
    <h1>Upload new File</h1>
    <form method=post enctype=multipart/form-data>
      <input type=file name=file>
      <input type=submit value=Upload>
    </form>
    '''
@app.route('/uploads/<filename>')
def uploaded_file(filename):
    return send_from_directory(app.config['UPLOAD_FOLDER'], filename)

def predict_file(image):
    dist_threshold = 0.4
    print("type of image2:", type(image))
    name=''
    # face_location: (top, right, bottom, left) 
    f_location = frg.face_locations(image, model='cnn')
    if len(f_location) != 1:
      return 'Incorrect face image!'
    print("type of f_location:", type(f_location))
    print("f_location:", f_location)
    encoding = frg.face_encodings(image, known_face_locations=f_location)
    if len(encoding) == 0:
      return 'No face encording'
    else:
      encoding = encoding[0]
    print("encoding type:", type(encoding))
    print(encoding)
    # reshape(1,-1) change [-0.17077433  0.086519...] to [[-0.17077433 0.086519 0.04608656...]]
    xt = encoding.reshape(1,-1)
    #print('xt:', xt)
    closest_distance = model.kneighbors(xt, n_neighbors=1, return_distance=True)
    #print("closest_distance[0][0][0]:",closest_distance[0][0][0])
    if closest_distance[0][0][0] <= dist_threshold :
    # model.predict(xt) returns a string list ['name']
    # model.predict(xt)[0] returns 'name'
      name = model.predict(xt)[0]
      print('name:', name)
    else:
      name = "Unknown"
    return name


if __name__ == '__main__':
    load_model()  # load model at the beginning once only
    app.run(host='0.0.0.0', port=3000)

