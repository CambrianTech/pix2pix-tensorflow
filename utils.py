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

def is_valid_image(path):
    try:
        Image.open(path)
    except IOError:
        return False
    return True

def get_image_paths(path, expression=None, filtered_dirs=None):
    file_names=[]

    #print("Checking for images at ", path, expression)

    valid_image_dir = True

    if not filtered_dirs is None and not os.path.basename(path) in filtered_dirs:
        valid_image_dir = False

    for file in os.listdir(path):
        file_path = os.path.join(path, file)
        if os.path.isdir(file_path):
            file_names.extend(get_image_paths(file_path, expression, filtered_dirs))
        elif valid_image_dir and is_valid_image(file_path) and (expression is None or fnmatch.fnmatch(file, expression)):
            file_names.append(file_path)

    file_names.sort()

    return file_names

def getCombinedImage(a_path, b_path):

    a_image = misc.imread(a_path)
    if (len(a_image.shape)<3):
        a_image = cv2.cvtColor(a_image, cv2.COLOR_GRAY2RGB)

    b_image = misc.imread(b_path)
    if (len(b_image.shape)<3):
        b_image = cv2.cvtColor(b_image, cv2.COLOR_GRAY2RGB)

    ha,wa = a_image.shape[:2]
    hb,wb = b_image.shape[:2]

    if (ha != hb or wa != wb):
        print("A and B images must match but do not for ", a_path, b_path)
        return None

    total_width = 2 * wa
    combined_img = np.zeros(shape=(ha, total_width, 3))

    combined_img[:ha,:wa]=a_image
    combined_img[:ha,wa:total_width]=b_image

    return combined_img

def hasParams(args, params):
    for param in params:
        paramValue = eval("args." + param)
        if paramValue is None:
            print("Error: argument --%s is required" % param)
            return False
    return True

def getABImagePaths(args):
    filtered_dirs = None
    if not args.filter_categories is None:

        if not os.path.isfile(args.filter_categories): 
            print("Error: filter_categories file %s does not exist" % args.filter_categories)
            return [], []

        filtered_dirs = []

        with open(args.filter_categories) as f:
            content = f.readlines()
            content = [x.strip() for x in content] 

        #/b/banquet_hall 38
        for line in content:
            category_search = re.search('/[a-z]/(\\w+)', line, re.IGNORECASE)
            if category_search:
                category = category_search.group(1)
                filtered_dirs.append(category)

    if not args.a_input_dir is None:
        if not hasParams(args, ["a_input_dir", "b_input_dir"]):
            return [], []

        if not os.path.isdir(args.a_input_dir): 
            print("Error: a_input_dir %s does not exist" % args.a_input_dir)
            return [], []

        if not os.path.isdir(args.b_input_dir): 
            print("Error: b_input_dir %s does not exist" % args.b_input_dir)
            return [], []

        if not filtered_dirs is None:
            filtered_dirs.append(os.path.basename(args.a_input_dir))
            filtered_dirs.append(os.path.basename(args.b_input_dir))

        a_names=get_image_paths(args.a_input_dir, args.a_match_exp, filtered_dirs=filtered_dirs)
        b_names=get_image_paths(args.b_input_dir, args.b_match_exp, filtered_dirs=filtered_dirs)
    else:

        if not hasParams(args, ["a_match_exp", "b_match_exp"]):
            return [], []

        if not os.path.isdir(args.input_dir): 
            print("Error: input_dir %s does not exist" % args.input_dir)
            return [], []

        if not filtered_dirs is None:
            filtered_dirs.append(os.path.basename(args.input_dir))

        a_names=get_image_paths(args.input_dir, args.a_match_exp, filtered_dirs=filtered_dirs)
        b_names=get_image_paths(args.input_dir, args.b_match_exp, filtered_dirs=filtered_dirs)

    return a_names, b_names

