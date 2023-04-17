import os
#from app import app
#import urllib.request
from flask import Flask,flash,url_for,redirect, request, jsonify, render_template
from werkzeug.utils import secure_filename
from util import config
from sklearn.metrics.pairwise import cosine_similarity
#from sklearn.metrics.pairwise import euclidean_distances
#from sklearn.metrics import mean_squared_error
import numpy as np
import cv2
import pickle as pk
import features as fp
import hgfeatures as hgf
#from imutils import paths
from scipy import spatial


UPLOAD_FOLDER = 'static/upload/'
photos= 'static/photo/'
ALLOWED_EXTENSIONS = set(['jpeg','jpg'])
global sketchname

app = Flask(__name__)
app.secret_key = "secret key"
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
#app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024
app.config['PHOTOS'] = photos

e = open('hgft.pickle','rb')
hg= pk.load(e)
f = open('vgg16.pickle','rb')
vg= pk.load(f)


def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS


@app.route('/')
def home():
    return render_template('index.html')

@app.route('/upload',methods=['POST','GET'])
def upload():
    if request.method == 'POST':
        file = request.files['file']
        if file and allowed_file(file.filename):
            filename = secure_filename(file.filename)
            file.save(os.path.join(app.config['UPLOAD_FOLDER'], filename))
            pic1 =os.path.join(app.config['UPLOAD_FOLDER'], filename)
            config.SKETCH_IMAGE=pic1
            #print(config.SKETCH_IMAGE)
            return render_template('index.html', filename=pic1)
        return

@app.route('/display/<filename>')
def display_image(filename):
    return render_template('index.html', filename=filename)


def load_sketch():
    print("[INFO] loading sketch image ...")
    
    image = cv2.imread(config.SKETCH_IMAGE)
    image =cv2.medianBlur(image,3)
    image = cv2.resize(image, (224, 224))
    image = np.expand_dims(image, axis=0)
    #sketch_id = config.SKETCH_IMAGE.split(os.path.sep)[-1].split(".")[0]
    # get features of sketch image
    pred = fp.model.predict(image, batch_size=1)
    sketch_features = pred.reshape((pred.shape[0], -1))
    return sketch_features

    



@app.route('/imagedesc',methods=['POST','GET'])
def imagedesc():
    image = cv2.imread(config.SKETCH_IMAGE)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    image =cv2.medianBlur(image,3)
    image=cv2.resize(image,(64*2 ,128*2))
    sk1= hgf.hogcal(image)
    sgl= hgf.glcmcal(image)
    #svg= vgg16f(b)
    f3 = np.concatenate((sk1,sgl))
    ids = list()
    similarity_scores = list()
    dict1 = {}
    # iterate through dataset and get similarity score
    for (id, feature) in hg.items():
            score = 1-spatial.distance.cosine(f3,feature)
            ids.append(id)
            similarity_scores.append(score)
            dict1[id]= score
    ids = np.array(ids)
    similarity_scores = np.array(similarity_scores)
    top3=[]
    cscore=[]
    for i in range(config.MAX_MATCHES):
        index = np.argmax(similarity_scores)
        #print("Match {}: {} with Similarity score: {:.4f}%".format(i+1, 
           # ids[index], similarity_scores[index]*100))
        top3.append(ids[index])
        cscore.append(similarity_scores[index]*100)
        ids = np.delete(ids, [index])
        similarity_scores = np.delete(similarity_scores, [index])
    file1=os.path.join(app.config['PHOTOS'],top3[0]+".jpg")
    file2=os.path.join(app.config['PHOTOS'],top3[1]+".jpg")
    file3=os.path.join(app.config['PHOTOS'],top3[2]+".jpg")
    sketch_id = config.SKETCH_IMAGE.split(os.path.sep)[-1].split(".")[0]
    filename=os.path.join(sketch_id+".jpg")
    return render_template('index.html', file1=file1,file2=file2,file3=file3,filename=filename,
                           score1="{:.2f} %".format(cscore[0]),score2="{:.2f} %".format(cscore[1]),score3="{:.2f} %".format(cscore[2]))
            
    '''
    dict2={k: v for k, v in sorted(dict1.items(),reverse=True, key=lambda item: item[1])}
    imgs=list(dict2.keys())
    score1=float( dict2[imgs[0]])*100
    score2= float(dict2[imgs[1]])*100
    score3= float(dict2[imgs[2]])*100
    file1=os.path.join(app.config['PHOTOS'],imgs[0]+".jpg")
    #cv2.imwrite()
    #file1.save(os.path.join(app.config['UPLOAD_FOLDER'], imgs[0]+'.jpg'))
    file2=os.path.join(app.config['PHOTOS'],imgs[1]+".jpg")
    file3=os.path.join(app.config['PHOTOS'],imgs[2]+".jpg")
    sketch_id = config.SKETCH_IMAGE.split(os.path.sep)[-1].split(".")[0]
    filename=os.path.join(sketch_id+".jpg")
    #file2=imgs[1]
    #file3=imgs[2]
    return render_template('index.html', file1=file1,file2=file2,file3=file3,
                           filename=filename,score1="{:.4f}".format(score1),score2="{:.4f}".format(score2),score3="{:.4f}".format(score3))'''

@app.route('/VGG16',methods=['POST','GET'])
def vgg():
    # initialize lists to store id and similarity score
    ids = list()
    similarity_scores = list()
    sketch_features = load_sketch()
# iterate through dataset and get similarity scores
    for (id, feature) in vg.items():
            score = cosine_similarity(sketch_features, feature)[0][0]
            ids.append(id)
            similarity_scores.append(score)
    ids = np.array(ids)
    similarity_scores = np.array(similarity_scores)
    top3=[]
    cscore=[]
    for i in range(config.MAX_MATCHES):
        index = np.argmax(similarity_scores)
        #print("Match {}: {} with Similarity score: {:.4f}%".format(i+1, 
           # ids[index], similarity_scores[index]*100))
        top3.append(ids[index])
        cscore.append(similarity_scores[index]*100)
        ids = np.delete(ids, [index])
        similarity_scores = np.delete(similarity_scores, [index])
    file1=os.path.join(app.config['PHOTOS'],top3[0]+".jpg")
    file2=os.path.join(app.config['PHOTOS'],top3[1]+".jpg")
    file3=os.path.join(app.config['PHOTOS'],top3[2]+".jpg")
    sketch_id = config.SKETCH_IMAGE.split(os.path.sep)[-1].split(".")[0]
    filename=os.path.join(sketch_id+".jpg")
    return render_template('index.html', file1=file1,file2=file2,file3=file3,filename=filename,
                           score1="{:.2f} %".format(cscore[0]),score2="{:.2f} %".format(cscore[1]),score3="{:.2f} %".format(cscore[2]))
'''
@app.route('/MSE',methods=['GET','POST'])
def mse():
    ids = list()
    similarity_scores = list()
    dict = {}
    sketch_features = load_sketch()
    # iterate through dataset and get similarity scores
    for (id, feature) in fp.photo_features.items():
            score = mean_squared_error(sketch_features, feature)
            ids.append(id)
            similarity_scores.append(score)
            dict[id]= score
    dict2={k: v for k, v in sorted(dict.items(), key=lambda item: item[1])}
    imgs=list(dict2.keys())
    score1=float( dict2[imgs[0]])/100
    score2= float(dict2[imgs[1]])/100
    score3= float(dict2[imgs[2]])/100
    file1=os.path.join(app.config['PHOTOS'],imgs[0]+".jpg")
    file2=os.path.join(app.config['PHOTOS'],imgs[1]+".jpg")
    file3=os.path.join(app.config['PHOTOS'],imgs[2]+".jpg")
    sketch_id = config.SKETCH_IMAGE.split(os.path.sep)[-1].split(".")[0]
    filename=os.path.join(sketch_id+".jpg")
    
    return render_template('index.html', file1=file1,file2=file2,file3=file3,
     filename=filename,score1="{:.4f}".format(score1),score2="{:.4f}".format(score2),score3="{:.4f}".format(score3))
'''
if __name__ == "__main__":
    app.run(host="localhost",port=5000,debug=True)

