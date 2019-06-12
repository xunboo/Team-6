# import the necessary packages
import os
import random
import sys
import pdb

import cv2 as cv
import keras.backend as K
import numpy as np
import sklearn.neighbors as nn

from config import img_rows, img_cols
from config import nb_neighbors, T, epsilon
from model import build_model

from PIL import Image

'''
takes PIL image
returns PIL image
'''
def colorize_image(image):
    bgr = np.array(image)
    gray = np.squeeze(np.rint(bgr.dot(np.asarray([[0.2989], [0.5870], [0.1140]]))))
    img_rows, img_cols = gray.shape[:2]

    # need to make multiple of 4
    img_rows = img_rows - (img_rows % 4)
    img_cols = img_cols - (img_cols % 4)

    # load pretrained weights and build model
    model_weights_path = 'models/model.06-2.5489.hdf5'
    model = build_model(img_rows, img_cols)
    model.load_weights(model_weights_path)

    # load other pretrained LAB data
    h, w = img_rows // 4, img_cols // 4
    q_ab = np.load("data/pts_in_hull.npy")
    nb_q = q_ab.shape[0]
    nn_finder = nn.NearestNeighbors(n_neighbors=nb_neighbors, algorithm='ball_tree').fit(q_ab)

    # resize images
    bgr = cv.resize(bgr, (img_cols, img_rows), cv.INTER_CUBIC)
    gray = cv.resize(gray, (img_cols, img_rows), cv.INTER_CUBIC)
    lab = cv.cvtColor(bgr, cv.COLOR_BGR2LAB)
    L = lab[:, :, 0]
    a = lab[:, :, 1]
    b = lab[:, :, 2]
    # Load the array of quantized ab value
    # Fit a NN to q_ab
    x_test = np.empty((1, img_rows, img_cols, 1), dtype=np.float32)
    x_test[0, :, :, 0] = gray / 255.

    # L: 0 <=L<= 255, a: 42 <=a<= 226, b: 20 <=b<= 223.
    X_colorized = model.predict(x_test)
    h = X_colorized.shape[1]
    w = X_colorized.shape[2]
    X_colorized = X_colorized.reshape((-1, nb_q))
    #h,w = X_colorized.shape

    # Reweight probas
    X_colorized = np.exp(np.log(X_colorized + epsilon) / T)
    X_colorized = X_colorized / np.sum(X_colorized, 1)[:, np.newaxis]

    # Reweighted
    q_a = q_ab[:, 0].reshape((1, 313))
    q_b = q_ab[:, 1].reshape((1, 313))

    # create output image
    X_a = np.sum(X_colorized * q_a, 1).reshape((h, w))
    X_b = np.sum(X_colorized * q_b, 1).reshape((h, w))
    X_a = cv.resize(X_a, (img_cols, img_rows), cv.INTER_CUBIC)
    X_b = cv.resize(X_b, (img_cols, img_rows), cv.INTER_CUBIC)
    X_a = X_a + 128
    X_b = X_b + 128
    out_lab = np.zeros((img_rows, img_cols, 3), dtype=np.int32)
    out_lab[:, :, 0] = lab[:, :, 0]
    out_lab[:, :, 1] = X_a
    out_lab[:, :, 2] = X_b
    out_L = out_lab[:, :, 0]
    out_a = out_lab[:, :, 1]
    out_b = out_lab[:, :, 2]
    out_lab = out_lab.astype(np.uint8)
    out_bgr = cv.cvtColor(out_lab, cv.COLOR_LAB2RGB)
    out_bgr = out_bgr.astype(np.uint8)
    #cv.imwrite('out1.jpeg', out_bgr)
    K.clear_session()
    #return Image.fromarray(out_lab, 'LAB')
    return Image.fromarray(out_bgr, 'RGB')

if __name__ == '__main__':
    # Read in and parse args, making sure they make sense
    if len(sys.argv) < 2:
        print('Usage: python colorize.py [filename]')
        sys.exit()
    filename = sys.argv[1]
    if not os.path.isfile(filename):
        print('Invalid filename:', filename)
        sys.exit()

    image = Image.open(filename)
    colorized_image = colorize_image(image)
    colorized_image.save("out.png", "PNG")
