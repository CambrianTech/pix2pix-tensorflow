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
import imghdr

def str2bool(v):
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')

def getSize(filename):
    if os.path.isfile(filename): 
        st = os.stat(filename)
        return st.st_size
    else:
        return -1

def is_valid_image(path, require_rgb=True):
    try:
        file_extension = os.path.splitext(path)[1].lower()
        return file_extension == ".jpeg" or file_extension == ".jpg" or file_extension == ".png"
        # im=Image.open(path)
        # im.verify()
        # return not require_rgb or im.mode == "RGB"
    except IOError:
        print("IOError with image " + path)
        return False
    return False

def get_image_paths(path, expression=None, filtered_dirs=None, require_rgb=True):
    file_names=[]

    # print("Checking for images at ", path, expression)

    valid_image_dir = True

    if not filtered_dirs is None and not os.path.basename(path) in filtered_dirs:
        valid_image_dir = False

    paths = os.listdir(path)

    # print("Got %d candidates" % len(paths))

    for file in paths:
        file_path = os.path.join(path, file)
        if os.path.isdir(file_path):
            file_names.extend(get_image_paths(file_path, expression, filtered_dirs, require_rgb))
        elif valid_image_dir and is_valid_image(file_path, require_rgb) and (expression is None or fnmatch.fnmatch(file, expression)):
            file_names.append(file_path)

    file_names.sort()

    return file_names

def getRGBImage(img):
    if (len(img.shape)<3):
        img = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)
    elif img.shape[2] == 4:
        img = cv2.cvtColor(img, cv2.COLOR_RGBA2RGB)
    return img

def getCombinedImage(a_path, b_path, a_margin=(0,0,0,0), b_margin=(0,0,0,0), a_function=None, b_function=None):

    a_image = misc.imread(a_path)
    a_image = getRGBImage(a_image)

    b_image = misc.imread(b_path)
    b_image = getRGBImage(b_image)

    ha,wa = a_image.shape[:2]
    #crop
    xa0 = a_margin[3]
    ya0 = a_margin[0]
    wa = wa - a_margin[1] - a_margin[3]
    ha = ha - a_margin[0] - a_margin[2]

    hb,wb = b_image.shape[:2]
    #crop
    xb0 = b_margin[3]
    yb0 = b_margin[0]
    wb = wb - b_margin[1] - b_margin[3]
    hb = hb - b_margin[0] - b_margin[2]

    if (ha != hb or wa != wb):
        print("A and B images must match but do not for ", a_path, b_path)
        return None

    # image[y0:y0+height , x0:x0+width, :]
    a_image = a_image[ya0:ya0+ha , xa0:xa0+wa, :]
    b_image = b_image[yb0:yb0+hb , xb0:xb0+wb, :]    

    if not a_function is None:
        a_image = a_function(a_image)

    if not b_function is None:
        b_image = b_function(b_image)

    if a_image is None or b_image is None:
        return None

    total_width = 2 * wa
    combined_img = np.zeros(shape=(ha, total_width, 3))

    combined_img[:ha,:wa]=a_image
    combined_img[:ha,wa:total_width]=b_image

    return combined_img

def hasParams(args, params):
    for param in params:
        paramValue = eval('args["' + param + '"]')
        if paramValue is None:
            print("Error: argument --%s is required" % param)
            return False
    return True

def getFilteredDirs(args):
    
    filtered_dirs = None

    if not args["filter_categories"] is None:
        filtered_dirs = []
        if not os.path.isfile(args["filter_categories"]): 
                print("Error: filter_categories file %s does not exist" % args["filter_categories"])
                return [], []

        with open(args["filter_categories"]) as f:
            content = f.readlines()
            content = [x.strip() for x in content] 

        #/b/banquet_hall 38
        for line in content:
            category_search = re.search('/[a-z]/(\\w+)', line, re.IGNORECASE)
            if category_search:
                category = category_search.group(1)
                filtered_dirs.append(category)

    return filtered_dirs

def getABImagePaths(args, require_rgb=True):
    filtered_dirs = getFilteredDirs(args)

    if not args["a_input_dir"] is None:
        if not hasParams(args, ["a_input_dir", "b_input_dir"]):
            return [], []

        if not os.path.isdir(args["a_input_dir"]): 
            print("Error: a_input_dir %s does not exist" % args["a_input_dir"])
            return [], []

        if not os.path.isdir(args["b_input_dir"]): 
            print("Error: b_input_dir %s does not exist" % args["b_input_dir"])
            return [], []

        if not filtered_dirs is None:
            filtered_dirs.append(os.path.basename(args["a_input_dir"]))
            filtered_dirs.append(os.path.basename(args["b_input_dir"]))

        a_names=get_image_paths(args["a_input_dir"], args["a_match_exp"], filtered_dirs=filtered_dirs, require_rgb=require_rgb)
        b_names=get_image_paths(args["b_input_dir"], args["b_match_exp"], filtered_dirs=filtered_dirs, require_rgb=require_rgb)
    else:

        if not hasParams(args, ["a_match_exp", "b_match_exp"]):
            return [], []

        if not os.path.isdir(args["input_dir"]): 
            print("Error: input_dir %s does not exist" % args["input_dir"])
            return [], []

        if not filtered_dirs is None:
            filtered_dirs.append(os.path.basename(args["input_dir"]))

        a_names=get_image_paths(args["input_dir"], args["a_match_exp"], filtered_dirs=filtered_dirs, require_rgb=require_rgb)
        b_names=get_image_paths(args["input_dir"], args["b_match_exp"], filtered_dirs=filtered_dirs, require_rgb=require_rgb)

    return a_names, b_names

