from os import listdir
from PIL import Image
import os.path
import numpy as np

height = 256   
width = 256

file = "file.exe"

def file2images(filename):
    file = filename
    images = []

    for f in listdir(file):
        with open(os.path.join(file, f), 'rb') as img_set:
            img_arr = img_set.read(height*width)
            while img_arr:
                if len(img_arr) == height*width and img_arr not in images:
                    images.append(img_arr)
                img_arr = img_set.read(height*width)


    return images

images = []

count = 0

for img in images:
    png = Image.fromarray(np.reshape(list(img), (height, width)).astype("float32"), mode='L')
    png.save('image_l%d.png'%count)
    count += 1