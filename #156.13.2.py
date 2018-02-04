#156.13.2.py
from keras.applications.vgg16 import VGG16
from keras.preprocessing import image
from keras.applications.vgg16 import preprocess_input
from keras.layers import Input, Flatten, Dense
from keras.models import Model
import numpy as np


img_path = '/Users/oba2311/Desktop/Minerva/Junior/CS156/week 5/300_men'

#Get back the convolutional part of a VGG network trained on ImageNet
model_vgg16_conv = VGG16(weights='imagenet', include_top=False)
model_vgg16_conv.summary()

#Create your own input format (here 3x200x300, that is the image I'm using here)
input = Input(shape=(3,200,300),name = 'image_input')

#Use the generated model 
output_vgg16_conv = model_vgg16_conv(input)

#Add the fully-connected layers 
x = Flatten(name='flatten')(output_vgg16_conv)
x = Dense(4096, activation='relu', name='fc1')(x)
x = Dense(4096, activation='relu', name='fc2')(x)
x = Dense(8, activation='softmax', name='predictions')(x)

#Create your own model 
my_model = Model(input=input, output=x)

#In the summary, weights and layers from VGG part will be hidden, but they will be fit during the training
my_model.summary()


#Then training with your data ! 

"""
The technique that we will explore here is known as "transfer learning", or "feature extraction". This assumes that the VGG16 network has already found a general and robust representation of most images, and we can reuse this representation, rather than relearning from scratch. To do this we will freeze all the lower level layers and replace the final layer with something suited to our task.
For this step, you have two options:
Add a new network layer, and train this final layer to predict your desired targets (eg. 0 for Men's clothing and 1 for Women's clothing).
Use the deep neural network as a nonlinear transformation which creates a rich representation on which to build a conventional classifier.
What cost function did you choose, and why? What performance do you achieve on your test set and how does this compare to the performance you were originally able to achieve with the linear methods?
(Optional) If you want, you can also perform a 'fine-tuning' step. In this step we unfreeze the weights and then perform a few more iterations of gradient descent. This fine tuning, can help the network specialize its performance in the particular task that it is needed for. Now measure, the new performance on you test set and compare it to the performance from the previous step.
Be sure to bring to class, all your code, as well as results in a format suitable for pasting in a poll, or Google document.
"""

#Ray's Code:
