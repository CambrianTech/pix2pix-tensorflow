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
from scipy import misc

parser = argparse.ArgumentParser()
parser.add_argument("--a_input_dir", required=True, help="Source Input, image A, usually rgb camera data")
parser.add_argument("--b_input_dir", required=True, help="Target Input, image B, usually labels")
parser.add_argument("--output_dir", required=True, help="where to put output files")
#parser.add_argument("--output_filetype", default="png", choices=["png", "jpeg"])
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

        a_image = misc.imread(a_name)
        b_image = misc.imread(b_name)

        ha,wa = a_image.shape[:2]
        hb,wb = b_image.shape[:2]

        if (ha != hb or wa != wb):
            print("A and B images must match but do not for ", a_name, b_name)
            return

        total_width = 2 * wa
        combined_img = np.zeros(shape=(ha, total_width, 3))

        combined_img[:ha,:wa]=a_image
        combined_img[:ha,wa:total_width]=b_image

        combined_img_name = os.path.basename(a_name)
        combined_img_path = os.path.join(a.output_dir, combined_img_name)
        
        misc.imsave(combined_img_path, combined_img)

        print (a_name, b_name)

main()
