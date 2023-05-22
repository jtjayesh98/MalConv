import keras
from keras.datasets import mnist
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten
from keras.layers import Conv2D, MaxPooling2D
from filetoimage import file2images, height, width
import numpy as np


benignfileloc = "benign"
maliciousfileloc = "malicious"
benign_images = file2images(benignfileloc)
malicious_images = file2images(maliciousfileloc)

benign_img_list = np.zeros(shape = (len(benign_images), height, width, 1), dtype = np.uint8)
malicious_img_list = np.zeros(shape = (len(malicious_images), height, width, 1), dtype = np.uint8)
for j in range(len(benign_images)):
    benign_img_list[j,:,:,0] = np.reshape(list(benign_images[j]), (height,width))
for j in range(len(malicious_img_list)):
    malicious_img_list[j,:,:,0] = np.reshape(list(malicious_images[j]), (height, width))


benign_img_list = benign_img_list.astype('float32')
benign_img_list /= 255
benign_img_list = np.array(benign_img_list)
malicious_img_list = malicious_img_list.astype("float32")
malicious_img_list /= 255
malicious_img_list = np.array(malicious_img_list)


model = Sequential()

model.add(Conv2D(12, (25, 25), padding='same',input_shape=benign_img_list.shape[1:], activation = 'relu'))
model.add(Conv2D(12, (25, 25), activation = 'relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Conv2D(12, (13, 13), padding='same', activation = 'relu'))
model.add(Conv2D(12, (13, 13), activation = 'relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Flatten())
model.add(Dense(1024, activation = 'relu'))
model.add(Dense(1, activation = 'sigmoid'))

model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['binary_accuracy'])

model.summary()


#Change the parameters to whatever suits you
batch_size = 512 
epochs = 100
labels = [0 for _ in benign_img_list] + [1 for _ in malicious_img_list]
labels = np.array(labels)
model.fit(benign_img_list+malicious_img_list, labels, batch_size = batch_size, epochs = epochs, validation_split = 0.25, shuffle = True)