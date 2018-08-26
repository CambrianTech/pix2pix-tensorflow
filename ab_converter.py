from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf
import numpy as np
import argparse
import os
import json
import glob
import random
import collections
import math
import time
from PIL import Image

parser = argparse.ArgumentParser()
parser.add_argument("--a_input_dir", required=True, help="Source Input, image A, usually rgb camera data")
parser.add_argument("--b_input_dir", required=True, help="Target Input, image B, usually labels")
parser.add_argument("--output_dir", required=True, help="where to put output files")
parser.add_argument("--output_filetype", default="png", choices=["png", "jpeg"])
a = parser.parse_args()

def is_valid_image(path):
    try:
        Image.open(path)
    except IOError:
        return False
    return True

def get_image_paths(path):
    file_names=[]
    for file in os.listdir(path):
        file_path = os.path.join(path, file)
        if is_valid_image(file_path):
            file_names.append(file_path)

    file_names.sort()

    return file_names

def main():

    a_names=get_image_paths(a.a_input_dir)
    b_names=get_image_paths(a.b_input_dir)

    num_a = len(a_names)
    num_b = len(b_names)

    if (num_a != num_b):
        print("A and B directories must contain the same number of images", num_a, num_b)
        return
    
    if not os.path.exists(a.output_dir):
        os.makedirs(a.output_dir)

    for i in range(0, num_a):
        a_name = a_names[i]
        b_name = b_names[i]
        print (a_name, b_name)

main()
