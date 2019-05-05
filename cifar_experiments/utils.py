import numpy as np
from numpy import load
from PIL import Image
import pdb

def convert_bw(arr):
    grey_arr = np.dot(arr, [0.299, 0.587, 0.114]).astype(np.uint8)
    return grey_arr

def save_arr(arr, filename, mode='RGB'):
    arr_int = arr.astype(np.uint8)
    image = Image.fromarray(arr_int, mode)
    image.save(filename)

def load_cifar():
    data = load('cifar-10.npz')
    train_data = data['train_data']
    return train_data

if __name__ == '__main__':
    data = load('cifar-10.npz')
    train_data = data['train_data']

    for i in range(10):
        save_arr(train_data[i], 'images/color' + str(i) + '.png')
        save_arr(convert_bw(train_data[i]), 'images/grey' + str(i) + '.png', 'L')
