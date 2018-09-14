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

#example usage:
# CamVid dataset:
# export datasets=~/Development/datasets; \
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

# python ab_converter.py \
# --a_input_dir $datasets/adenormals \
# --b_input_dir $datasets/ADE20K_indoor_256 \
# --a_match_exp "ADE_*.png" \
# --b_match_exp "ADE_*.png" \
# --output_dir $datasets/adenormals_indoor_AB/train

# My notes:

## Train:
# python pix2pix.py --mode train --output_dir ade20k_train --max_epochs 2000 --input_dir ADE20KAB/train --which_direction AtoB --lr=0.0001 --batch_size=10
# Download a lot of images:

# Downloading lots of images:
# googleimagesdownload --keywords "nose before and after" --size medium --limit 20000 --chromedriver="/usr/local/bin/chromedriver"

# Deploy:
# python pix2pix.py --mode=deploy --output_dir=$CB/CBAssets/nnets --checkpoint=$checkpoints/ade20k_train --deploy_name=ade20k.pb
# python pix2pix.py --mode=deploy --output_dir=$CB/CBAssets/nnets --checkpoint=$checkpoints/normals_train_fast --deploy_name=normals.pb

# Test:
# python pix2pix.py --mode=export datasets=~/Development/datasets; python pix2pix.py --mode=test --checkpoint=$checkpoints/normals_train_web --input_dir=$datasets/test_rooms --output_dir=$datasets/test_results 

# export datasets=/datasets; \
# python pix2pix.py --mode train \
# --output_dir normals_train \
# --max_epochs 2000 \
# --a_input_dir $datasets/mlt_v2 \
# --a_match_exp '*.png' \
# --b_input_dir $datasets/normals_v2 \
# --b_match_exp '*_norm_camera.png' \
# --which_direction AtoB --no_flip \
# --ngf=128 --ndf=128

# export datasets=/datasets; \
# python pix2pix.py --mode train \
# --output_dir normals_train \
# --max_epochs 2000 \
# --a_input_dir $datasets/mlt_v2 \
# --a_match_exp '*.png' \
# --b_input_dir $datasets/normals_v2 \
# --b_match_exp '*_norm_camera.png' \
# --which_direction AtoB \
# --lr=0.0001 --batch_size=10

# export datasets=/datasets; \
# python pix2pix.py --mode train \
# --output_dir nosejobs_train \
# --max_epochs 2000 \
# --input_dir $datasets/nosejobs \
# --which_direction AtoB 

# export datasets=/datasets; \
# python ab_converter.py \
# --input_dir $datasets/nyu_surface_normals \
# --a_match_exp "*_color.png" \
# --b_match_exp "*_norm_camera.png" \
# --output_dir $datasets/nyu_surface_normals_AB \
# --margin=45,40,10,40

#   download lots of images:
# googleimagesdownload room images

# python ab_converter.py \
# --input_dir $datasets/ADE20K_2016_07_26/images/training \
# --filter_categories $datasets/ADE20K_2016_07_26/indoor-categories.txt \
# --a_match_exp "ADE_*.jpg" --b_match_exp "ADE_*_seg.png" \
# --output_dir $datasets/ADE20K_simplified_AB/train \
# --replace_colors $datasets/ADE20K_2016_07_26/replace-colors.txt 

# python ab_converter.py \
# --input_dir $datasets/ADE20K_2016_07_26/images/training \
# --filter_categories $datasets/ADE20K_2016_07_26/indoor-categories.txt \
# --a_match_exp "ADE_*01638.jpg" --b_match_exp "ADE_*01638_seg.png" \
# --output_dir $datasets/ADE20K_simplified_AB/train \
# --replace_colors $datasets/ADE20K_2016_07_26/replace-colors.txt 

# python ab_converter.py \
# --a_input_dir $datasets/adenormals \
# --b_input_dir mloutput \
# --a_match_exp "ADE_*.png" --b_match_exp "ADE_*.png" \
# --output_dir $datasets/adenormals_simplified_AB/train \
# --replace_colors $datasets/ADE20K_2016_07_26/replace-colors.txt 

# python pix2pix.py --mode train --output_dir normals512 \
# --a_input_dir ../datasets/mlt_v2 --a_match_exp '*.png' \
# --b_input_dir ../datasets/normal_v2 --b_match_exp '*_norm_camera.png' \
# --which_direction AtoB --no_flip --ndf 128 --ngf 128 --crop_size 512 --max_epochs 2000

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

parser.add_argument("--image_filter", required=False, help="Image filter to apply to two images")

# Place to output A/B images
parser.add_argument("--output_dir", required=True, help="where to put output files")

a = parser.parse_args()

matches = []
replacements = []

def replaceColors(im):

    h,w = im.shape[:2]
    
    # print(im[16,100])

    red, green, blue = im[:,:,0], im[:,:,1], im[:,:,2]

    default = None
    total_mask = np.zeros([h,w],dtype=np.uint8)

    num_elements = 0
    lastZeroCount = 0
    for i in range(0, len(matches)):
        if matches[i] == "*":
            default = replacements[i]
        else:
            for j in range(0, len(matches[i])):
                color_to_replace = matches[i][j]
                mask = (red == color_to_replace[0]) & (green == color_to_replace[1])
                im[:,:,:3][mask] = replacements[i] #codes for below
                total_mask[mask] = 255
                nzCount = cv2.countNonZero(total_mask)
                if nzCount > lastZeroCount:
                    num_elements = num_elements + 1
                lastZeroCount = nzCount
    
    if num_elements < 3:
        return None

    if not default is None:
        im[total_mask != 255] = default

    return im

def getColor(input): 
    if not input.startswith('['):
        input = '[' + input

    if not input.endswith(']'):
        input = input + ']'

    return ast.literal_eval(input)

def processFiles(a_names, b_names):
    num_a = len(a_names)
    num_b = len(b_names)

    if (num_a != num_b):
        print("A and B directories must contain the same number of images", num_a, num_b)
        return

    if not os.path.exists(a.output_dir):
        os.makedirs(a.output_dir)

    margins = a.margin.split(",")

    a_margin=(int(margins[0]),int(margins[1]),int(margins[2]),int(margins[3]))
    b_margin=a_margin

    a_function = None
    b_function = None

    if not a.replace_colors is None:
        if not os.path.isfile(a.replace_colors): 
            print("Error: replace_colors file %s does not exist" % a.replace_colors)
            return
        b_function = replaceColors

        with open(a.replace_colors) as f:
            content = f.readlines()
            content = [x.strip() for x in content] 

        #/b/banquet_hall 38

        for line in content:
            line = re.sub(r'\s+', '', line) # Remove spaces
            data_search = re.search('(.+):(.+)//', line, re.IGNORECASE)
            if data_search:
                if data_search.group(1).startswith('*'):
                    to_replace = data_search.group(1)
                else:
                    to_replace = data_search.group(1).split('],[')
                    to_replace = [getColor(x) for x in to_replace]
                matches.append(to_replace)
                replace_with = data_search.group(2)
                replacements.append(ast.literal_eval(replace_with.strip()))


    for i in range(0, num_a):
        a_name = a_names[i]
        b_name = b_names[i]

        combined_img = utils.getCombinedImage(a_name, b_name, a_margin=a_margin, b_margin=b_margin, b_function=b_function)

        if not combined_img is None:
            combined_img_name = os.path.basename(a_name)
            combined_img_path = os.path.join(a.output_dir, combined_img_name)

            misc.imsave(combined_img_path, combined_img)

            print ("%s + %s = %s" % (a_name, b_name, combined_img_path))        

def main():

    a_names, b_names = utils.getABImagePaths(a, require_rgb=False)

    processFiles(a_names, b_names)  
    

main()
