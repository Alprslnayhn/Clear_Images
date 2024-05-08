"""
Script for pre-processing the data by
resizing, median filtering the images.
And finally training the Neural Network model
for the task of classifying blur and clear images.

"""
# Loading required Libraries
from __future__ import print_function
from config import *
from utils import (h, sigmoid, validate, resize,
                    model_score, path_validation)
import numpy as np
import matplotlib.pyplot as plt
import os
from sklearn.model_selection import train_test_split
import joblib
import imageio.v2 as imageio 
import scipy.ndimage as nd
import argparse

def data_preprocess(GOOD_IMG_PATH, BAD_IMG_PATH, radius=3):
    """
    Extracts the images from the given paths
    then pre-process them by applying
    median filter to filter out the 
    noise present in the images and 
    finally concatenate the good and bad 
    preprocessed images to 
    one input images

    @ Parameters:
    -------------
    GOOD_IMG_PATH: str
        Path of the folder containing
        good images
    BAD_IMG_PATH: str
        Path of the folder containing
        bad images
    radius: int
        Radius of the median filter 
        applied to the image

    @ Returns:
    ----------
    combined_img: np.array
        filtered and pre-processed combined
        images arrays of both good and clear 
        iamges
    labels: np.array
        labels containing 1, if images is good
        and 0, if image is bad

    """
    print('Pre-Processing the Data...\n')
    # Reading the Good Images 
    good_img = []
    for filename in os.listdir(GOOD_IMG_PATH):
        good_img.append(imageio.imread(os.path.join(GOOD_IMG_PATH, filename)))
    good_img = np.asarray(good_img)

    # Reading the Bad Images 
    bad_img = []
    for filename in os.listdir(BAD_IMG_PATH):
        bad_img.append(imageio.imread(os.path.join(BAD_IMG_PATH, filename)))
    bad_img = np.asarray(bad_img)

    # Concatenate the array of Good & Bad images
    combined_img = np.concatenate((good_img, bad_img))  
    labels = np.concatenate((np.ones(good_img.shape[0]), 
                            np.zeros(bad_img.shape[0])))
 
    # Filtering the combined images to Reduce the Noise present
    combined_img = nd.median_filter(combined_img, size=radius)

    return combined_img, labels


def save_data(train_images, train_labels,
            test_images, test_labels):
    """
    Checking the existence of path
    if not exists then creates one
    and save the train & test data
    """

    if path_validation(TRAIN_DATA_PATH):
        print('Train Data Path Success...')
    if path_validation(TRAIN_LABEL_PATH):
        print('Train Label Path Success...')
    if path_validation(TEST_DATA_PATH):
        print('Test Data Path Success...')
    if path_validation(TEST_LABEL_PATH):
        print('Test Label Path Success...')

    print('Saving the splitting results...\n')
    np.save(TRAIN_DATA_PATH, train_images)
    np.save(TRAIN_LABEL_PATH, train_labels)
    np.save(TEST_DATA_PATH, test_images)
    np.save(TEST_LABEL_PATH, test_labels)


def main():
    """
    Pre-process the data with filtering, resizing 
    and trained the Neural Networks with 
    resulting pre-processed data using backpropagation

    """

    # Construct the argument parse and parse the arguments
    ap = argparse.ArgumentParser()
    ap.add_argument("-path1", "--good_path", required=True, 
                        help="path to good images directory")
    ap.add_argument("-path2", "--bad_path", required=True, 
                        help="path to bad images directory")
    args = vars(ap.parse_args())

    # Taking Absolute Path needed for reading images
    GOOD_IMG_PATH = os.path.abspath(args["good_path"]) + str('/')
    BAD_IMG_PATH = os.path.abspath(args["bad_path"]) + str('/')

    # Path Validation
    if not path_validation(GOOD_IMG_PATH, read_access=True):
        exit(0)
    if not path_validation(BAD_IMG_PATH, read_access=True):
        exit(0)

    # Model Path Validation
    if path_validation(MODEL_PATH):
        print('Model Path Success...\n')

    # Getting the Same Result in Shuffle in each Run.
    np.random.seed(SEED)

    # Convert the Good & Bad Images to Cumulative numpy array 
    imgs, labels = data_preprocess(GOOD_IMG_PATH, BAD_IMG_PATH, radius=RADIUS)
    			
    # Resizing the feature space for easier handling
    imgs = resize(imgs, width=WIDTH, height=HEIGHT)

    # Splitting the Data for Training and Testing Purpose
    print('Splitting of Data...\n')
    train_images, test_images, train_labels, test_labels = train_test_split(imgs, labels, 
                                                        test_size=SPLIT_RATIO, random_state=SEED) 

    # Saving the splitted data to disk
    save_data(train_images, train_labels, test_images, test_labels)

    # No of unique classes in data
    nclass = np.unique(labels).shape[0]
    			
    # Adding Bias in Train/Test Images
    train_images = np.insert(train_images, 0, 1, axis=1) 
    test_images = np.insert(test_images, 0, 1, axis=1)

    # Initializing the Model
    theta = NN_Model([train_images.shape[1], NEURONS_SIZE, nclass])

    print("BackPROP...\n")
    params = back_propagate(theta['Theta1'], theta['Theta2'], train_images, train_labels,
                            nclass, alpha=ALPHA, lambdaa=LAMBDA, max_iter=MAX_ITER, act=ACT, 
                            batch_size=BATCH_SIZE, logging=LOGGING_STEPS)

    # Accuracy Score on Train set
    accuracy = model_score(params, train_images, train_labels, act=ACT) 
    print('Accuracy on Train Data:', accuracy)

    # Accuracy Score on test set
    accuracy = model_score(params, test_images, test_labels, act=ACT) 
    print('Accuracy on Test Data:', accuracy)

    # Storing the Results
    print('Saving Results...\n')
    joblib.dump(params, MODEL_PATH)

    # Plotting the Curve
    show_plot(params['Loss'], PLOT_PATH)


if __name__ == "__main__":
    main()

