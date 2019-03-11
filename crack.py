"""
UTCTF 2019
FaceSafe
Can you get the secret? http://facesafe.xyz
"""
import keras
import numpy as np

from keras.models import load_model
from keras import backend as K
from PIL import Image

# The class we want to obtain (I have just counted from 0 to 9). No magic here!
TARGET = 4

# The strength of the perturbation
# ! VERY IMPORTANT: it is an integer beause we want to produce images in RGB, not greyscale!
eps = 1

# Produce encoding of the output for class 4 (canonical base, e_4)
target = np.zeros(10)
target[TARGET] = 1

#l Lad the model using Keras, standard way for saving netwrok's weights is HDF5 format.
model = load_model('model.model')

# Produce np array from image, using PIL (one of the thousand ways for loading an image)
img = Image.open('img2.png')
img.load()
data = np.asarray(img, dtype="int32")

print(np.argmax(model.predict(np.array([data]))[0]),' should be 4 BUT NOT NOW!')

# We need te function that incapsulate the gradient of the loss wrt the input.
# ! MOST IMPORTANT: the loss function is the main actor here. It defines what we want to search.
# In this case, we want the distance between the prediciton and the target label 4.
# Hence, we produce the loss written there.
session = K.get_session()
d_model_d_x = K.gradients( keras.losses.mean_squared_error(target, model.output), model.input)

x0 = data
conf = model.predict(np.array([x0]))[0]

# The attack may last forever? 
# YES! But I tried with a black image and it converges. 
# You should put here a fixed number of iterations...
while np.argmax(conf) != TARGET:

	# Thank you Keras + Tensorflow! 
	# That [0][0] is just ugly, but it is needed to obtain the value as an array.
	eval_grad = session.run(d_model_d_x, feed_dict={model.input:np.array([x0])} )[0][0]

	# Compute the perturbation! 
	# This is the Fast Sign Gradient Method attack.
	fsgm = eps * np.sign(eval_grad)

	# The gradient always points to maximum ascent direction, but we need to minimize.
	# Hence, we swap the sign of the gradient.
	x0 = x0 - fsgm

	# Here we need to bound the editing. No negative values in images! 
	# So we clip all the negative values to 0.
	# We also clip all values above 255.
	x0[x0 < 0] = 0
	x0[x0 > 255] = 255
	conf = model.predict(np.array([x0]))[0]
	print("Confdence of target class {}: {:.3f}%\nPredicted class: {}\nConfidence of predicted class: {:.3f}%\n----".format(TARGET, conf[TARGET]*100, np.argmax(conf), conf[np.argmax(conf)]*100))

# If we obtained the evasion, we just save the new image
i = Image.fromarray(x0.astype('uint8'), 'RGB')
i.save('adv.png')
