# ignore tf warnings
import logging, os
logging.disable(logging.WARNING)
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"

# import
import tensorflow
import numpy as np
import cv2
import pickle as pk
from tensorflow.keras.applications import VGG16
from imutils import paths
from util import config


# load model
print("[INFO] loading network ...")
model = VGG16(weights="imagenet", include_top=False)

if __name__ == "__main__":
    # load images
    print("[INFO] loading images ...")
    imagePaths = list(paths.list_images(config.PHOTOS_PATH))
    print(len(imagePaths))
    # init dictionary to capture features
    photo_features = dict()
    
    for (i, imagePath) in enumerate(imagePaths):
        # load image and preprocess it
        image = cv2.imread(imagePath, 0)
        image = cv2.resize(image, (224, 224))
        image = np.expand_dims(image, axis=2)
        image = tensorflow.image.grayscale_to_rgb(tensorflow.convert_to_tensor(image))
        image = np.expand_dims(image, axis=0)
    
        # get id
        id = imagePath.split(os.path.sep)[-1].split(".")[0]
    
        # get features
        pred = model.predict(image, batch_size=1)
        features = pred.reshape((pred.shape[0], -1))
    
        # store features in features dictionary
        photo_features[id] = features
        #print(photo_features)
    with open("vgg16.pickle",'wb') as fh:
        pk.dump(photo_features,fh)