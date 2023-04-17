# import 
import os

# path to original dataset
DIRECTORY = "CUHK"
PHOTOS_PATH = os.path.sep.join([DIRECTORY, "photo"])
SKETCH_PATH = os.path.sep.join([DIRECTORY, "sketch"])
#path =  os.path.sep.join([DIRECTORY, "photo\"])
# path to sketch
SKETCH_IMAGE = os.path.sep.join([SKETCH_PATH, "sketch1.jpg"])
P_IMAGE = os.path.sep.join([DIRECTORY,"sketch2.jpg"])
# maximum images to match
MAX_MATCHES = 3