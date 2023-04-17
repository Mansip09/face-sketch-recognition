# -*- coding: utf-8 -*-
"""
Created on Fri May  7 12:55:33 2021

@author: MANSI
"""


import cv2
import os
from util import config
from imutils import paths
from skimage.feature import hog
from skimage.feature import greycomatrix,greycoprops
import numpy as np
import  pickle as pk

def glcmcal(image):
    #glcm features
    res1=[]
    res2=[]
    res3=[]
    eye=image[100:140,35:95]
    a=greycomatrix(eye,[1],[0])
    #res1.append(greycoprops(a,'contrast')[0][0])
    res1.append(greycoprops(a,'correlation')[0][0])
    res1.append(greycoprops(a,'energy')[0][0])
    res1.append(greycoprops(a,'homogeneity')[0][0])
    
    nose = image[140:170,52:75]
    a=greycomatrix(nose,[1],[0])
    #res2.append(greycoprops(a,'contrast')[0][0])
    res2.append(greycoprops(a,'correlation')[0][0])
    res2.append(greycoprops(a,'energy')[0][0])
    res2.append(greycoprops(a,'homogeneity')[0][0])
    
    mouth = image[170:200,0:128]
    a=greycomatrix(mouth,[1],[0])
    #res3.append(greycoprops(a,'contrast')[0][0])
    res3.append(greycoprops(a,'correlation')[0][0])
    res3.append(greycoprops(a,'energy')[0][0])
    res3.append(greycoprops(a,'homogeneity')[0][0])
    
    return np.array(res1+res2+res3)

def hogcal(image):
    fd, hog_image = hog(image, orientations=9, pixels_per_cell=(8, 8),
                cells_per_block=(2, 2), visualize=True)
    return fd


if __name__=="__main__":
    #print("__name__")
    print("[INFO] loading images ...")
    imagePaths = list(paths.list_images(config.PHOTOS_PATH))
    print(len(imagePaths))
    # init dictionary to capture features
    hogglcm_features = dict()
    
    for (i, imagePath) in enumerate(imagePaths):
        
        # load image and preprocess it
        image = cv2.imread(imagePath)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        image =cv2.medianBlur(image,3)
        image=cv2.resize(image,(64*2 ,128*2))
        # get id
        id = imagePath.split(os.path.sep)[-1].split(".")[0]
    
        # get hog features
        fd=hogcal(image)
        #glcm features
        gl=glcmcal(image)
        #vg= vgg16f(imagePath)
        ft = np.concatenate((fd,gl))
        # store features in features dictionary
        hogglcm_features[id] = ft
    with open("hgft.pickle",'wb') as fh:
        pk.dump(hogglcm_features,fh)
    #e = open('hgft.pickle','rb')
    #d= pk.load(e)