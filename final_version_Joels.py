from PIL import Image
import PIL.ImageOps

from collections import defaultdict
from glob import glob
from random import shuffle, seed
import numpy as np
import pylab as pl
import pandas as pd
import re
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split


STANDARD_SIZE = (138,138)
HALF_SIZE = (STANDARD_SIZE[0]/2,STANDARD_SIZE[1]/2)

    
def img_to_array(filename):
    """
    takes a filename and turns it into a numpy array of RGB pixels
    """
    img = Image.open(filename)
    img = img.resize(STANDARD_SIZE)
    img = list(img.getdata())
    img = map(list, img)
    img = np.array(img)
    s = img.shape[0] * img.shape[1]
    img_wide = img.reshape(1, s)
    return img_wide[0]

male_files = glob('/images/boys/*')
female_files = glob('/images/girls/*')

process_file = img_to_array

raw_data = []
for filename in male_files:
    print (filename)
    raw_data.append((process_file(filename),'male',filename))
for filename in female_files:
    print (filename)
    raw_data.append((process_file(filename),'female',filename))
    
# randomly order the data
seed(0)
shuffle(raw_data)

# pull out the features and the labels
data = np.array([cd for (cd,_y,f) in raw_data])
labels = np.array([_y for (cd,_y,f) in raw_data])
    
X_train, X_test, y_train, y_test = train_test_split(data, labels, test_size=0.2)

print(X_train)

# find the principal components
N_COMPONENTS = 10
pca = PCA(n_components=N_COMPONENTS, random_state=0)
X = pca.fit_transform(X_train)
y = [1 if label == 'boy' else 0 for label in labels]

