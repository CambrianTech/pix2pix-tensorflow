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

#example usage:
# CamVid dataset:
# export datasets=../../datasets; \
# python ab_converter.py \
# --a_input_dir $datasets/CamVid/train \
# --b_input_dir $datasets/CamVid/train_labels \
# --output_dir $datasets/CamVidAB/train

# ADE20K dataset (indoor only):
# export datasets=../../datasets; \
# python ab_converter.py \
# --input_dir $datasets/ADE20K_2016_07_26/images/training \
# --filter_categories $datasets/ADE20K_2016_07_26/indoor-categories.txt \
# --a_match_exp "ADE_*.jpg" \
# --b_match_exp "ADE_*_seg.png" \
# --output_dir $datasets/ADE20K_indoor_AB/train

# python pix2pix.py --mode train --output_dir ade20k_train --max_epochs 2000 --input_dir $datasets/ADE20KAB/train --which_direction AtoB

# python pix2pix.py --mode train --output_dir ade20k_train --max_epochs 2000 --input_dir ADE20KAB/train --which_direction AtoB

parser = argparse.ArgumentParser()

# required together:
parser.add_argument("--a_input_dir", required=False, help="Source Input, image A, usually rgb camera data")
parser.add_argument("--b_input_dir", required=False, help="Target Input, image B, usually labels")

# required together:
parser.add_argument("--input_dir", required=False, help="Combined Source and Target Input Path")
parser.add_argument("--a_match_exp", required=False, help="Source Input expression to match files")
parser.add_argument("--b_match_exp", required=False, help="Source Input expression to match files")

parser.add_argument("--filter_categories", required=False, help="Path to file with valid categories")

# Place to output A/B images
parser.add_argument("--output_dir", required=True, help="where to put output files")

a = parser.parse_args()

def is_valid_image(path):
    try:
        Image.open(path)
    except IOError:
        return False
    return True

def get_image_paths(path, expression=None, filtered_dirs=None):
    file_names=[]

    #print("Checking %s for images" % path)
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

def processFiles(a_names, b_names):
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
	    if (len(a_image.shape)<3):
	    	a_image = cv2.cvtColor(a_image, cv2.COLOR_GRAY2RGB)

	    b_image = misc.imread(b_name)
	    if (len(b_image.shape)<3):
	    	b_image = cv2.cvtColor(b_image, cv2.COLOR_GRAY2RGB)

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

	    print ("%s + %s = %s" % (a_name, b_name, combined_img_path))

def hasParams(params):
	for param in params:
		paramValue = eval("a." + param)
		if paramValue is None:
			print("Error: argument --b_input_dir is required")
			return False
	return True

def main():

	filtered_dirs = None
	if not a.filter_categories is None:

		if not os.path.isfile(a.filter_categories): 
			print("Error: filter_categories file %s does not exist" % a.filter_categories)
			return

		filtered_dirs = []

		with open(a.filter_categories) as f:
			content = f.readlines()
			content = [x.strip() for x in content] 

		#/b/banquet_hall 38
		for line in content:
			category_search = re.search('/[a-z]/(\\w+)', line, re.IGNORECASE)
			if category_search:
				category = category_search.group(1)
				filtered_dirs.append(category)

	if not a.input_dir is None:
		if not hasParams(["a_match_exp", "b_match_exp"]):
			return

		if not os.path.isdir(a.input_dir): 
			print("Error: input_dir %s does not exist" % a.input_dir)
			return

		if not filtered_dirs is None:
			filtered_dirs.append(os.path.basename(a.input_dir))

		a_names=get_image_paths(a.input_dir, a.a_match_exp, filtered_dirs=filtered_dirs)
		b_names=get_image_paths(a.input_dir, a.b_match_exp, filtered_dirs=filtered_dirs)
	else:
		if not hasParams(["a_input_dir", "b_input_dir"]):
			return

		if not os.path.isdir(a.a_input_dir): 
			print("Error: a_input_dir %s does not exist" % a.a_input_dir)
			return

		if not os.path.isdir(a.b_input_dir): 
			print("Error: b_input_dir %s does not exist" % a.b_input_dir)
			return

		if not filtered_dirs is None:
			filtered_dirs.append(os.path.basename(a.a_input_dir))
			filtered_dirs.append(os.path.basename(a.b_input_dir))

		a_names=get_image_paths(a.a_input_dir, filtered_dirs=filtered_dirs)
		b_names=get_image_paths(a.b_input_dir, filtered_dirs=filtered_dirs)

	processFiles(a_names, b_names)	

    

main()
