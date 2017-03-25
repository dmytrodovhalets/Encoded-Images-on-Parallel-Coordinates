import csv
import math
from sklearn.datasets import fetch_mldata
import numpy as np
from PIL import Image
from keras.layers import Input, Dense
from keras.models import Model

"""Get Data
"""
custom_data_home = '/home/dovhaletsd/mnist'
mnist = fetch_mldata('MNIST original', data_home=custom_data_home)
x_all, y_all = mnist.data, mnist.target


"""Save image from array
"""
def save_image(image_array, name):
	image = np.asarray(image_array)
	size = math.sqrt(len(image_array))
	image = np.reshape(image, (size, size))
	image = Image.fromarray(image)
	image.save('mnist_example_'+str(name)+'.png')


"""Crop the image to remove padding
"""
def crop_images(image_array):
	image = np.asarray(image_array)
	size = math.sqrt(len(image_array))
        image = np.reshape(image, (size, size))
	image = Image.fromarray(image)
	origional_w = size
	origional_h = size
	percent_to_keep = 0.80
        w = int(origional_w * percent_to_keep)
        h = int(origional_h * percent_to_keep)
        difference_w = (origional_w - w)/2
        difference_h = (origional_h - h)/2
        x = difference_w
        y = difference_h
        box = [x, y, x+w, y+h]
        cropped_img = image.crop(box)
	cropped_img = np.asarray(cropped_img)
	return np.ravel(cropped_img)


"""Test to get images
img_index = 0
image = x_all[img_index]
image = crop_images(image)
print(len(image))
save_image(image, str(img_index)+'_cropped')
"""

x_all_cropped = []
for i in xrange(len(x_all)):
	image = x_all[i]
	image = crop_images(image)
	x_all_cropped.append(image)

x_all_cropped = np.asarray(x_all_cropped)
x_train = x_all_cropped[:60000]
x_test = x_all_cropped[60000:]

x_train = x_train.astype('float32')/255.0 #normolizing between 0 and 1
x_test = x_test.astype('float32')/255.0
x_train = x_train.reshape((len(x_train), np.prod(x_train.shape[1:])))
x_test = x_test.reshape((len(x_test), np.prod(x_test.shape[1:])))

"""Autoencoder using Neural Network
"""
#encoding_dimension = 34 # 7%(484)
encoding_dimension = 24 # 5%(484)
image_size = len(x_train[0])

input_img = Input(shape=(image_size,)) #484 origional
encoded = Dense(encoding_dimension, activation='relu')(input_img) #34
decoded = Dense(image_size, activation='sigmoid')(encoded) #484 reconstruction
autoencoder = Model(input=input_img, output=decoded)


encoder = Model(input=input_img, output=encoded) #484 -> 34
encoded_input = Input(shape=(encoding_dimension,)) #34
decoder_layer = autoencoder.layers[-1]
decoder = Model(input=encoded_input, output=decoder_layer(encoded_input)) #34 -> 484

#training the autoencoder
autoencoder.compile(optimizer='adadelta', loss='binary_crossentropy')
autoencoder.fit(x_train, x_train, nb_epoch=50, batch_size=300, shuffle=True,  verbose=1)

#compiling the  encoder and decoder
encoder.compile(optimizer='adadelta', loss='binary_crossentropy')
decoder.compile(optimizer='adadelta', loss='binary_crossentropy')
encoded_images = encoder.predict(x_test)

print("Length of encoded images: ", len(encoded_images[0]))
decoded_images = decoder.predict(encoded_images)
print("Length of decoded images: ", len(decoded_images[0]))


"""Saving encoded representation into a CSV file with labels
"""
encoded_representation = []
for i in xrange(len(encoded_images)):
	one_representation = []
	one_representation.append(y_all[(60000+i)])
	for x in xrange(len(encoded_images[0])):
		one_representation.append(encoded_images[i][x])
	encoded_representation.append(one_representation)

with open("mnist_encoded_test_24.csv", "w") as output:
	writer = csv.writer(output, lineterminator='\n')
	for y in encoded_representation:
		writer.writerow(y)
output.close()


#one_image = decoded_images[8299]
#one_image = one_image*255
#one_image = one_image.astype('uint8')
#save_image(x_all[65299], "origional_image_5299")
#save_image(one_image, "decoded_50epochs_8299_24")

print("Done")
