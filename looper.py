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
import io
from scipy import misc
import cv2
import pix2pix_model

import utils

from tensorflow.python.framework import graph_util,dtypes
from tensorflow.python.tools import optimize_for_inference_lib, selective_registration_header_lib

# python looper.py --checkpoint /Volumes/YUGE/checkpoints/normals_fake_web_60k \
# --input_dir /Volumes/YUGE/datasets/ADE20K_2016_07_26/images/training --input_match_exp *.jpg

parser = argparse.ArgumentParser()

parser.add_argument("--input_dir", required=False, default="uploads", help="Combined Source and Target Input Path")
parser.add_argument("--input_match_exp", required=False, help="Input Match Expression")
parser.add_argument("--filter_categories", required=False, help="Path to file with valid categories")

parser.add_argument("--output_dir", default="mloutput", required=False, help="where to put output files")
parser.add_argument("--seed", type=int)
parser.add_argument("--checkpoint", default=None, help="directory with checkpoint to resume training from or use for testing")
parser.add_argument("--max_steps", type=int, help="number of training steps (0 to disable)")
parser.add_argument("--max_epochs", type=int, help="number of training epochs")
parser.add_argument("--summary_freq", type=int, default=100, help="update summaries every summary_freq steps")
parser.add_argument("--progress_freq", type=int, default=50, help="display progress every progress_freq steps")
parser.add_argument("--trace_freq", type=int, default=0, help="trace execution every trace_freq steps")
parser.add_argument("--display_freq", type=int, default=0, help="write current training images every display_freq steps")
parser.add_argument("--save_freq", type=int, default=5000, help="save model every save_freq steps, 0 to disable")

parser.add_argument("--separable_conv", action="store_true", help="use separable convolutions in the generator")
parser.add_argument("--aspect_ratio", type=float, default=1.0, help="aspect ratio of output images (width/height)")
parser.add_argument("--lab_colorization", action="store_true", help="split input image into brightness (A) and color (B)")
parser.add_argument("--batch_size", type=int, default=1, help="number of images in batch")
parser.add_argument("--which_direction", type=str, default="AtoB", choices=["AtoB", "BtoA"])
parser.add_argument("--ngf", type=int, default=64, help="number of generator filters in first conv layer")
parser.add_argument("--ndf", type=int, default=64, help="number of discriminator filters in first conv layer")
parser.add_argument("--crop_size", type=int, default=256, help="crop images")
parser.add_argument("--flip", dest="flip", action="store_true", help="flip images horizontally")
parser.add_argument("--no_flip", dest="flip", action="store_false", help="don't flip images horizontally")
parser.set_defaults(flip=True)
parser.add_argument("--lr", type=float, default=0.0002, help="initial learning rate for adam")
parser.add_argument("--beta1", type=float, default=0.5, help="momentum term of adam")
parser.add_argument("--l1_weight", type=float, default=100.0, help="weight on L1 term for generator gradient")
parser.add_argument("--gan_weight", type=float, default=1.0, help="weight on GAN term for generator gradient")
parser.add_argument("--delete_src", type=bool, default=False, help="delete source images")

# export options
parser.add_argument("--output_filetype", default="png", choices=["png", "jpeg"])
a = parser.parse_args()

def main():

    print("Image flipping is turned", ('ON' if a.flip else 'OFF'))

    if a.seed is None:
        a.seed = random.randint(0, 2**31 - 1)

    tf.set_random_seed(a.seed)
    np.random.seed(a.seed)
    random.seed(a.seed)

    if not os.path.exists(a.output_dir):
        os.makedirs(a.output_dir)

    if a.checkpoint is None:
        raise Exception("checkpoint required")

    # load some options from the checkpoint
    options = {"which_direction", "ngf", "ndf", "lab_colorization"}
    with open(os.path.join(a.checkpoint, "options.json")) as f:
        for key, val in json.loads(f.read()).items():
            if key in options:
                print("loaded", key, "=", val)
                setattr(a, key, val)
    # disable these features in test mode
    a.flip = False

    for k, v in a._get_kwargs():
        print(k, "=", v)
    
    input_image = tf.placeholder(dtype=tf.float32, shape=[a.crop_size, a.crop_size, 3], name='input')
    batch_input = tf.expand_dims(input_image, axis=0)

    with tf.variable_scope("generator"):
        batch_output = pix2pix_model.deprocess(pix2pix_model.create_generator(a, pix2pix_model.preprocess(batch_input), 3))

    output_image = tf.image.convert_image_dtype(batch_output, dtype=tf.uint8, saturate=True)[0]
    if a.output_filetype == "png":
        output_data = tf.image.encode_png(output_image)
    elif a.output_filetype == "jpeg":
        output_data = tf.image.encode_jpeg(output_image, quality=80)
    else:
        raise Exception("invalid filetype")

    init_op = tf.global_variables_initializer()
    restore_saver = tf.train.Saver()

    image_shape = (a.crop_size, a.crop_size, 3)

    with tf.Session() as sess:
        sess.run(init_op)
        print("Loading model from checkpoint")
        checkpoint = tf.train.latest_checkpoint(a.checkpoint)
        restore_saver.restore(sess, checkpoint)
        print("Loaded model, waiting on images at %s ..." % a.input_dir)

        while True:

            filtered_dirs = utils.getFilteredDirs(a)
            paths = utils.get_image_paths(a.input_dir, a.input_match_exp, require_rgb=False, filtered_dirs=filtered_dirs)

            num_images = len(paths)
            if num_images:
                print("Processing %d images" % num_images)
                for i in range(num_images):
                    path = paths[i]
                    filename = os.path.splitext(os.path.basename(path))[0]

                    percent_complete = float(100 * i) / float(num_images)
                    print("(%.2f%%) Processing image %s" % (percent_complete, filename))

                    test = misc.imread(path, mode='RGB')
                    test = misc.imresize(test, image_shape)

                    test = test.astype('float32') / 255.0

                    results = sess.run(output_data, feed_dict={input_image:test})
                    
                    output_name = filename + "." + a.output_filetype

                    with open("mloutput/" + output_name, 'w') as fd:
                        fd.write(results)

                    if a.delete_src:
                        os.remove(path)
                print("Waiting on images...")
            else:
                print("none found, waiting")

            if not a.delete_src:
                print("DONE")
                return

            time.sleep(0.25)
        


main()
