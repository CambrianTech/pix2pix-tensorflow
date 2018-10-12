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
import fnmatch
import cv2
import re
import utils
import ast
import image_processing

parser = argparse.ArgumentParser()

# required together:
parser.add_argument("--a_input_dir", required=False, help="Source Input, image A, usually rgb camera data")
parser.add_argument("--b_input_dir", required=False, help="Target Input, image B, usually labels")

# required together:
parser.add_argument("--input_dir", required=False, help="Combined Source and Target Input Path")
parser.add_argument("--a_match_exp", required=False, help="Source Input expression to match files")
parser.add_argument("--b_match_exp", required=False, help="Source Input expression to match files")

parser.add_argument("--margin", type=str, required=False, default="0,0,0,0", help="Crop margin as: top, right, bottom, left")

parser.add_argument("--filter_categories", required=False, help="Path to file with valid categories")
parser.add_argument("--replace_colors", required=False, help="Path to file with GT color replacements. See replace-colors.txt")

parser.add_argument("--filter", required=True, help="Image filter to apply to two images")
parser.add_argument("--opacity", type=float, default=1.0, required=False, help="Opacity of filter")

# Place to output A/B images
parser.add_argument("--output_dir", required=True, help="where to put output files")

a = parser.parse_args()

# python ab_combiner.py --output_dir=bullshit --a_input_dir ade-output --b_input_dir normals-output --filter=difference --opacity=0.6

def combine(a_names, b_names):
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

        image_a = cv2.cvtColor(misc.imread(a_name), cv2.COLOR_RGB2RGBA).astype(float)

        image_b = cv2.cvtColor(misc.imread(b_name), cv2.COLOR_RGB2RGBA).astype(float)

        processed = eval("image_processing." + a.filter + "(image_a, image_b, a.opacity)")

        output_filename = a.output_dir + "/" + os.path.basename(a_name)
        print(output_filename)
        misc.imsave(output_filename, processed) 

def main():
    a_dir = {
        "input_dir": a.input_dir,
        "a_input_dir": a.a_input_dir,
        "b_input_dir": a.b_input_dir,
        "a_match_exp": a.a_match_exp,
        "b_match_exp": a.b_match_exp,
    }
    a_names, b_names = utils.getABImagePaths(a_dir, require_rgb=False)

    combine(a_names, b_names)  
    

main()
