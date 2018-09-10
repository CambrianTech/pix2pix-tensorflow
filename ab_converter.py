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

# My notes:

## Train:
# python pix2pix.py --mode train --output_dir ade20k_train --max_epochs 2000 --input_dir ADE20KAB/train --which_direction AtoB --lr=0.0001 --batch_size=10
# Download a lot of images:
# googleimagesdownload --keywords "nose before and after" --size medium --limit 20000 --chromedriver="/usr/local/bin/chromedriver"

# Deploy::
# python pix2pix.py --mode=deploy --output_dir=$CB/CBAssets/nnets --checkpoint=../../checkpoints/ade20k_train --deploy_name=ade20k.pb
# python pix2pix.py --mode=deploy --output_dir=$CB/CBAssets/nnets --checkpoint=../../checkpoints/normals_train_fast --deploy_name=normals.pb

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

# Place to output A/B images
parser.add_argument("--output_dir", required=True, help="where to put output files")

a = parser.parse_args()

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

    for i in range(0, num_a):
        a_name = a_names[i]
        b_name = b_names[i]

        combined_img = utils.getCombinedImage(a_name, b_name, a_margin, b_margin)

        if not combined_img is None:
            combined_img_name = os.path.basename(a_name)
            combined_img_path = os.path.join(a.output_dir, combined_img_name)

            misc.imsave(combined_img_path, combined_img)

            print ("%s + %s = %s" % (a_name, b_name, combined_img_path))        

def main():

    a_names, b_names = utils.getABImagePaths(a)

    processFiles(a_names, b_names)  
    

main()
