import os
from PIL import Image
import numpy as np
from scipy.misc import imread
from tqdm import tqdm


def main():
    men = "/Users/oba2311/Desktop/Minerva/Junior/CS156/week 5/Man's Clothing"
    women = "/Users/oba2311/Desktop/Minerva/Junior/CS156/week 5/Woman's Clothing"

    size_100 = (100, 100)
    size_300 = (300, 300)
    imlist = np.array([])
    for f in os.listdir(men):
        if f.endswith(".JPEG"):
            i = Image.open(r"/Users/oba2311/Desktop/Minerva/Junior/CS156/week 5/Man's Clothing{}".format("/" + f))
            fn, fext = os.path.splitext(f)
            i.thumbnail(size_300)
            i.save("/Users/oba2311/Desktop/Minerva/Junior/CS156/week 5/300_men/{}_300{}".format(fn, fext))
            i.thumbnail(size_100)
            i.save("/Users/oba2311/Desktop/Minerva/Junior/CS156/week 5/100_men/{}_100{}".format(fn, fext))
            i = imread(r"/Users/oba2311/Desktop/Minerva/Junior/CS156/week 5/Man's Clothing{}".format("/" + f))
            imlist = np.append([imlist], [i])
    imlist = np.array(imlist).reshape(-1, 1)
    "women block"
    list2 = np.array([])
    for f in os.listdir(women):
        if f.endswith(".JPEG"):
            i = Image.open(r"/Users/oba2311/Desktop/Minerva/Junior/CS156/week 5/Woman's Clothing{}".format("/" + f))
            fn, fext = os.path.splitext(f)
            i.thumbnail(size_300)
            i.save("/Users/oba2311/Desktop/Minerva/Junior/CS156/week 5/300_women/{}_300{}".format(fn, fext))
            i.thumbnail(size_100)
            i.save("/Users/oba2311/Desktop/Minerva/Junior/CS156/week 5/100_women/{}_100{}".format(fn, fext))
            i = imread(r"/Users/oba2311/Desktop/Minerva/Junior/CS156/week 5/Woman's Clothing{}".format("/" + f))
            list2 = np.append([list2], [i])
    list2 = np.array(list2).reshape(1, -1)

    data, labels = imlist + list2, [np.ones(len(imlist))] + [np.zeros(len(list2))]
    return data, labels


def mini_main():
    """
    This is the same func, for minimized testable workflow, with 10X10 images and n=20, for men only.
    """
    minimen = "/Users/oba2311/Desktop/Minerva/Junior/CS156/week 5/minimen"
    size_10 = (10, 10)
    for i in tqdm(range(10)):
        imlist = np.array([])
        for f in os.listdir(minimen):
            i += 1
            if f.endswith(".JPEG"):
                i = Image.open(
                    r"/Users/oba2311/Desktop/Minerva/Junior/CS156/week 5/Man's Clothing{}".format("/" + f))
                fn, fext = os.path.splitext(f)
                i.thumbnail(size_10)
                i.save("/Users/oba2311/Desktop/Minerva/Junior/CS156/week 5/100_men/{}_100{}".format(fn, fext))
                i = imread(r"/Users/oba2311/Desktop/Minerva/Junior/CS156/week 5/Man's Clothing{}".format("/" + f))
                imlist = np.append([imlist], [i])
    imlist = np.array(imlist).reshape(-1, 1)
    data, labels = imlist, np.ones(len(imlist))

    # This is the same func, for women.
    miniwomen = "/Users/oba2311/Desktop/Minerva/Junior/CS156/week 5/miniwomen"
    size_10 = (10, 10)
    for i in tqdm(range(10)):
        imlist = np.array([])
        for f in os.listdir(miniwomen):
            i += 1
            if f.endswith(".JPEG"):
                i = Image.open(
                    r"/Users/oba2311/Desktop/Minerva/Junior/CS156/week 5/Woman's Clothing{}".format("/" + f))
                fn, fext = os.path.splitext(f)
                i.thumbnail(size_10)
                i.save("/Users/oba2311/Desktop/Minerva/Junior/CS156/week 5/100_women/{}_100{}".format(fn, fext))
                i = imread(r"/Users/oba2311/Desktop/Minerva/Junior/CS156/week 5/Woman's Clothing{}".format("/" + f))
                imlist = np.append([imlist], [i])
    imlist = np.array(imlist).reshape(-1, 1)
    data_i, labels_i = imlist, np.zeros(len(imlist))
    integ_data = np.concatenate((data, data_i))
    integ_labels = np.concatenate((labels, labels_i))
    return integ_data, integ_labels


mini_main()
