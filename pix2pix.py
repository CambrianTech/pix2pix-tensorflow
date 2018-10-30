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

import shutil
import utils
import pix2pix_model

from tensorflow.python.framework import graph_util,dtypes
from tensorflow.python.tools import optimize_for_inference_lib, selective_registration_header_lib

from sacred import Experiment
from sacred.stflow import LogFileWriter

ex = Experiment("pix2pix")

#Examples:

# Training:
#train with composite a/b images:
# python pix2pix.py with "args = {'input_dir': '/Volumes/YUGE/datasets/shadows_ab'}"

#train with two directories:
# python pix2pix.py with "args = {'a_input_dir': '/Volumes/YUGE/datasets/unreal_rugs_binary/train', 'b_input_dir': '/Volumes/YUGE/datasets/unreal_rugs_binary/train_labels'}"

#train with two directories using four channels:
# python pix2pix.py with "args = {'channels': 4, 'a_input_dir': '/Users/joelteply/Desktop/unreal_rugs_binary/train', 'b_input_dir': '/Users/joelteply/Desktop/unreal_rugs_binary/train_labels'}"

# Exporting for feed forward
# python pix2pix.py with "args = {'mode': 'export', 'checkpoint':'/Users/joelteply/Desktop/normals_pix2pix'}"

@ex.config
def pix2pix_config():
    args = {
        "a_input_dir": None,
        "b_input_dir": None,
        "input_dir": None,
        "input_match_exp": None,
        "a_match_exp": None,
        "b_match_exp": None,
        "filter_categories": None,
        "mode": "train",
        "channels": 3,
        "output_dir": "output",
        "deploy_name": "model.pb",
        "checkpoint": None,
        "transform_ops": "strip_unused_nodes,add_default_attributes, \
                            remove_nodes(op=Identity,op=CheckNumerics,op=HashTable,op=HashTableV2,op=MutableHashTable,op=MutableHashTableV2,op=MutableDenseHashTable,op=MutableDenseHashTableV2,op=MutableHashTableOfTensors,op=MutableHashTableOfTensorsV2,op=LookupTableImport,op=LookupTableImportV2,op=LookupTableExport,op=LookupTableExportV2,op=LookupTableSize,op=LookupTableSizeV2,op=LookupTableFind,op=LookupTableFindV2,op=InitializeTableFromTextFile,op=InitializeTableFromTextFileV2),\
                            fold_constants(ignore_errors=true),fold_batch_norms,fold_old_batch_norms,quantize_weights,sort_by_execution_order",
        "max_steps": None,
        "max_epochs": 3000,
        "summary_freq": 100,
        "progress_freq": 50,
        "trace_freq": 0,
        "display_freq": 0,
        "save_freq": 5000,
        "separable_conv": False,
        "aspect_ratio": 1.0,
        "lab_colorization": False,
        "batch_size": 1,
        "which_direction": "AtoB",
        "ngf": 64,
        "ndf": 64,
        "crop_size": 256,
        "scale_size": 0,
        "flip": True,
        "lr": 0.0002,
        "beta1": 0.5,
        "l1_weight": 100.0,
        "gan_weight": 1.0,
        "output_filetype": "png",
        "no_disc_bn": False,
        "no_gen_bn": False,
        "layer_norm": False,
        "lr_g": 0.0002,
        "lr_d": 0.0002,
        "beta1": 0.5,
        "beta2": 0.999,
        "l1_weight": 100.0,
        "gan_weight": 1.0,
        "gan_loss": "gan",
        "gp_weight": 0,
        "init_stddev": 0.02,
    }

EPS = 1e-12

Examples = collections.namedtuple("Examples", "paths, inputs, targets, count, steps_per_epoch")

def check_image(image, channels=3):
    assertion = tf.assert_equal(tf.shape(image)[-1], channels, message=("image must have %d color channels" % channels))
    with tf.control_dependencies([assertion]):
        image = tf.identity(image)

    # make the last dimension 3 so that you can unstack the colors
    shape = list(image.get_shape())
    shape[-1] = channels
    image.set_shape(shape)
    return image

# based on https://github.com/torch/image/blob/9f65c30167b2048ecbe8b7befdc6b2d6d12baee9/generic/image.c
def rgb_to_lab(srgb):
    with tf.name_scope("rgb_to_lab"):
        srgb = check_image(srgb)
        srgb_pixels = tf.reshape(srgb, [-1, 3])

        with tf.name_scope("srgb_to_xyz"):
            linear_mask = tf.cast(srgb_pixels <= 0.04045, dtype=tf.float32)
            exponential_mask = tf.cast(srgb_pixels > 0.04045, dtype=tf.float32)
            rgb_pixels = (srgb_pixels / 12.92 * linear_mask) + (((srgb_pixels + 0.055) / 1.055) ** 2.4) * exponential_mask
            rgb_to_xyz = tf.constant([
                #    X        Y          Z
                [0.412453, 0.212671, 0.019334], # R
                [0.357580, 0.715160, 0.119193], # G
                [0.180423, 0.072169, 0.950227], # B
            ])
            xyz_pixels = tf.matmul(rgb_pixels, rgb_to_xyz)

        # https://en.wikipedia.org/wiki/Lab_color_space#CIELAB-CIEXYZ_conversions
        with tf.name_scope("xyz_to_cielab"):
            # convert to fx = f(X/Xn), fy = f(Y/Yn), fz = f(Z/Zn)

            # normalize for D65 white point
            xyz_normalized_pixels = tf.multiply(xyz_pixels, [1/0.950456, 1.0, 1/1.088754])

            epsilon = 6/29
            linear_mask = tf.cast(xyz_normalized_pixels <= (epsilon**3), dtype=tf.float32)
            exponential_mask = tf.cast(xyz_normalized_pixels > (epsilon**3), dtype=tf.float32)
            fxfyfz_pixels = (xyz_normalized_pixels / (3 * epsilon**2) + 4/29) * linear_mask + (xyz_normalized_pixels ** (1/3)) * exponential_mask

            # convert to lab
            fxfyfz_to_lab = tf.constant([
                #  l       a       b
                [  0.0,  500.0,    0.0], # fx
                [116.0, -500.0,  200.0], # fy
                [  0.0,    0.0, -200.0], # fz
            ])
            lab_pixels = tf.matmul(fxfyfz_pixels, fxfyfz_to_lab) + tf.constant([-16.0, 0.0, 0.0])

        return tf.reshape(lab_pixels, tf.shape(srgb))


def lab_to_rgb(lab):
    with tf.name_scope("lab_to_rgb"):
        lab = check_image(lab)
        lab_pixels = tf.reshape(lab, [-1, 3])

        # https://en.wikipedia.org/wiki/Lab_color_space#CIELAB-CIEXYZ_conversions
        with tf.name_scope("cielab_to_xyz"):
            # convert to fxfyfz
            lab_to_fxfyfz = tf.constant([
                #   fx      fy        fz
                [1/116.0, 1/116.0,  1/116.0], # l
                [1/500.0,     0.0,      0.0], # a
                [    0.0,     0.0, -1/200.0], # b
            ])
            fxfyfz_pixels = tf.matmul(lab_pixels + tf.constant([16.0, 0.0, 0.0]), lab_to_fxfyfz)

            # convert to xyz
            epsilon = 6/29
            linear_mask = tf.cast(fxfyfz_pixels <= epsilon, dtype=tf.float32)
            exponential_mask = tf.cast(fxfyfz_pixels > epsilon, dtype=tf.float32)
            xyz_pixels = (3 * epsilon**2 * (fxfyfz_pixels - 4/29)) * linear_mask + (fxfyfz_pixels ** 3) * exponential_mask

            # denormalize for D65 white point
            xyz_pixels = tf.multiply(xyz_pixels, [0.950456, 1.0, 1.088754])

        with tf.name_scope("xyz_to_srgb"):
            xyz_to_rgb = tf.constant([
                #     r           g          b
                [ 3.2404542, -0.9692660,  0.0556434], # x
                [-1.5371385,  1.8760108, -0.2040259], # y
                [-0.4985314,  0.0415560,  1.0572252], # z
            ])
            rgb_pixels = tf.matmul(xyz_pixels, xyz_to_rgb)
            # avoid a slightly negative number messing up the conversion
            rgb_pixels = tf.clip_by_value(rgb_pixels, 0.0, 1.0)
            linear_mask = tf.cast(rgb_pixels <= 0.0031308, dtype=tf.float32)
            exponential_mask = tf.cast(rgb_pixels > 0.0031308, dtype=tf.float32)
            srgb_pixels = (rgb_pixels * 12.92 * linear_mask) + ((rgb_pixels ** (1/2.4) * 1.055) - 0.055) * exponential_mask

        return tf.reshape(srgb_pixels, tf.shape(lab))

@ex.capture
def load_examples(args):
    a_names = []
    b_names = []
    combined_names = []
    num_images = 0

    if not args["a_input_dir"] is None or not args["a_match_exp"] is None:
        a_input_dirs = args["a_input_dir"].split(",")
        b_input_dirs = args["b_input_dir"].split(",")
        if args["a_match_exp"] is not None:
            a_match_exps = args["a_match_exp"].split(",")
        else:
            a_match_exps = None

        if args["b_match_exp"] is not None:
            b_match_exps = args["b_match_exp"].split(",")
        else:
            b_match_exps = None

        assert a_match_exps is None or len(a_input_dirs) == len(a_match_exps)
        assert b_match_exps is None or len(b_input_dirs) == len(b_match_exps)

        if a_match_exps:
            a_names = [utils.get_image_paths(input_dir, match_exp) for input_dir, match_exp in zip(a_input_dirs, a_match_exps)]
        else:
            a_names = [utils.get_image_paths(input_dir, None) for input_dir in a_input_dirs]

        if b_match_exps:
            b_names = [utils.get_image_paths(input_dir, match_exp) for input_dir, match_exp in zip(b_input_dirs, b_match_exps)]
        else:
            b_names = [utils.get_image_paths(input_dir, None) for input_dir in b_input_dirs]

        if any([len(a_names[0]) != names for names in a_names]):
            raise Exception("Image count for a_input_dirs not equal")

        if any([len(b_names[0]) != names for names in b_names]):
            raise Exception("Image count for b_input_dirs not equal")

        if len(a_names[0]) != len(b_names[0]):
            raise Exception("len a_input_dirs not equal to len b_input_dirs")

        if not a_names is None:
            num_images = len(a_names[0])
    elif not args["input_dir"] is None:
        combined_names = utils.get_image_paths(args["input_dir"], args["input_match_exp"])
        if not combined_names is None:
            num_images = len(combined_names)
    else:
        raise Exception("input_dir or a_input_dir/b_input_dir required")

    if num_images == 0:
        raise Exception("No images found at input path")

    if len(a_names) > 0:
        filename, file_extension = os.path.splitext(a_names[0][0])
    else:
        filename, file_extension = os.path.splitext(combined_names[0])

    if file_extension == ".png":
        decode = tf.image.decode_png
    elif file_extension == ".jpg" or file_extension == ".jpeg":
        decode = tf.image.decode_jpeg
    else:
        raise Exception("input_dir contains no image files")
    
    def get_name(path):
        name, _ = os.path.splitext(os.path.basename(path))
        return name

    with tf.name_scope("load_images"):
        if len(combined_names) > 0:
            path_queue = tf.train.string_input_producer(combined_names, shuffle=args["mode"] == "train")
            reader = tf.WholeFileReader()
            paths, contents = reader.read(path_queue)
            raw_input = decode(contents, channels=args["channels"])
            raw_input = tf.image.convert_image_dtype(raw_input, dtype=tf.float32)

            assertion = tf.assert_equal(tf.shape(raw_input)[2], args["channels"], message=("image does not have %d channels" % args["channels"]))
            with tf.control_dependencies([assertion]):
                raw_input = tf.identity(raw_input)

            raw_input.set_shape([None, None, args["channels"]])

            # break apart image pair and move to range [-1, 1]
            width = tf.shape(raw_input)[1] # [height, width, channels]
            a_images = pix2pix_model.preprocess(raw_input[:,:width//2,:])
            b_images = pix2pix_model.preprocess(raw_input[:,width//2:,:])

# https://github.com/affinelayer/pix2pix-tensorflow/issues/49
# I think you will want to comment out / skip the colour space code 
# as that will not work with more than three channels.
# I am not sure whether you meant that just conceptually. 
# But just to clarify, inputs need to be a TensorFlow tensor. 
# If input1 and input2 are both TensorFlow tensors then 
# you will want to use tf.concat to combine them together.

        else:
            path_queues = tf.train.slice_input_producer(a_names + b_names, shuffle=args["mode"] == "train")

            images = []

            for path_queue in path_queues:
                path = tf.decode_raw(path_queue, tf.uint8)
                contents = tf.read_file(path_queue)
                raw_input = decode(contents, channels=args["channels"])
                raw_input = tf.image.convert_image_dtype(raw_input, dtype=tf.float32)

                images.append(pix2pix_model.preprocess(raw_input))

            a_paths_count = len(a_names)
            b_paths_count = len(b_names)

            assert len(images) == a_paths_count + b_paths_count

            a_images = images[:a_paths_count]
            b_images = images[a_paths_count:]

            # Stack along channel dimension
            a_images = tf.stack(a_images, axis=-1)
            b_images = tf.stack(b_images, axis=-1)

    if args["which_direction"] == "AtoB":
        inputs, targets = [a_images, b_images]
    elif args["which_direction"] == "BtoA":
        inputs, targets = [b_images, a_images]
    else:
        raise Exception("invalid direction")

    # synchronize seed for image operations so that we do the same operations to both
    # input and output images
    seed = random.randint(0, 2**31 - 1)
    def transform(image):
        r = image
        if args["flip"]:
            r = tf.image.random_flip_left_right(r, seed=seed)

        # area produces a nice downscaling, but does nearest neighbor for upscaling
        # assume we're going to be doing downscaling here
        r = tf.image.resize_images(r, [args["scale_size"], args["scale_size"]], method=tf.image.ResizeMethod.AREA)

        offset = tf.cast(tf.floor(tf.random_uniform([2], 0, args["scale_size"] - args["crop_size"] + 1, seed=seed)), dtype=tf.int32)
        if args["scale_size"] > args["crop_size"]:
            r = tf.image.crop_to_bounding_box(r, offset[0], offset[1], args["crop_size"], args["crop_size"])
        elif args["scale_size"] < args["crop_size"]:
            raise Exception("scale size cannot be less than crop size")
        return r

    with tf.name_scope("input_images"):
        input_images = transform(inputs)

    with tf.name_scope("target_images"):
        target_images = transform(targets)

    paths_batch, inputs_batch, targets_batch = tf.train.batch([paths, input_images, target_images], batch_size=args["batch_size"])
    steps_per_epoch = int(math.ceil(num_images / args["batch_size"]))

    return Examples(
        paths=paths_batch,
        inputs=inputs_batch,
        targets=targets_batch,
        count=num_images,
        steps_per_epoch=steps_per_epoch,
    )

@ex.capture
def save_images(fetches, output_dir, step=None):
    image_dir = os.path.join(output_dir, "images")
    if not os.path.exists(image_dir):
        os.makedirs(image_dir)

    filesets = []
    for i, in_path in enumerate(fetches["paths"]):
        name, _ = os.path.splitext(os.path.basename(in_path.decode("utf8")))
        fileset = {"name": name, "step": step}
        for kind in ["inputs", "outputs", "targets"]:
            filename = name + "-" + kind + ".png"
            if step is not None:
                filename = "%08d-%s" % (step, filename)
            fileset[kind] = filename
            out_path = os.path.join(image_dir, filename)
            contents = fetches[kind][i]
            with open(out_path, "wb") as f:
                f.write(contents)
        filesets.append(fileset)
    return filesets


def append_index(filesets, output_dir, step=False):
    index_path = os.path.join(output_dir, "index.html")
    if os.path.exists(index_path):
        index = open(index_path, "a")
    else:
        index = open(index_path, "w")
        index.write("<html><body><table><tr>")
        if step:
            index.write("<th>step</th>")
        index.write("<th>name</th><th>input</th><th>output</th><th>target</th></tr>")

    for fileset in filesets:
        index.write("<tr>")

        if step:
            index.write("<td>%d</td>" % fileset["step"])
        index.write("<td>%s</td>" % fileset["name"])

        for kind in ["inputs", "outputs", "targets"]:
            index.write("<td><img src='images/%s'></td>" % fileset[kind])

        index.write("</tr>")
    return index_path

@ex.automain
@LogFileWriter(ex)
def main(args, _seed):
    print("\n\npython pix2pix.py with \"args =", args, "\"\n")

    if args["scale_size"] == 0:
        args["scale_size"] = args["crop_size"]

    print("Image flipping is turned", ('ON' if args["flip"] else 'OFF'))

    tf.set_random_seed(_seed)
    np.random.seed(_seed)
    random.seed(_seed)

    if not os.path.exists(args["output_dir"]):
        os.makedirs(args["output_dir"])

    if args["mode"] == "test" or args["mode"] == "export" or args["mode"] == "deploy":
        if args["checkpoint"] is None:
            raise Exception("checkpoint required for mode: " + args["mode"])

        # load some options from the checkpoint
        options = {"which_direction", "ngf", "ndf", "lab_colorization", "crop_size", "channels"}
        with open(os.path.join(args["checkpoint"], "options.json")) as f:
            for key, val in json.loads(f.read()).items():
                if key in options:
                    print("loaded", key, "=", val)
                    args[key] = val
        # disable these features in test mode
        args["scale_size"] = args["crop_size"]
        args["flip"] = False

    if args["mode"] != "deploy":
        with open(os.path.join(args["output_dir"], "options.json"), "w") as f:
            f.write(json.dumps(args, sort_keys=True, indent=4))

    if args["mode"] == "export":
        # export the generator to a meta graph that can be imported later for standalone generation
        shape = [args["crop_size"], args["crop_size"], args["channels"]]
        input_image = tf.placeholder(dtype=tf.float32, shape=shape, name='input')
        batch_input = tf.expand_dims(input_image, axis=0)

        with tf.variable_scope("generator"):
            batch_output = pix2pix_model.deprocess(pix2pix_model.create_generator(args, pix2pix_model.preprocess(batch_input), args["channels"]))

        # output_image = tf.image.convert_image_dtype(batch_output, dtype=tf.uint8)[0]
        # output_image = tf.identity(output_image, name='output')
        # output_image = tf.identity(batch_output[0], name='output')

        input_name = batch_input.name.split(':')[0]
        output_name = batch_output.name.split(':')[0]

         #[print(n.name) for n in tf.get_default_graph().as_graph_def().node]
        print("##############################################################\n")
        print("Input Name:", input_name)
        print("Output Name:", output_name)
        print("##############################################################\n")

        init_op = tf.global_variables_initializer()
        restore_saver = tf.train.Saver()
        export_saver = tf.train.Saver()

        with tf.Session() as sess:
            sess.run(init_op)

            print("##############################################################\n")
            print("\nLoading model from checkpoint %s" % args["checkpoint"])
            checkpoint = tf.train.latest_checkpoint(args["checkpoint"])
            restore_saver.restore(sess, checkpoint)
            
            # Save the model
            inputs = {'input': batch_input}
            outputs = {'output': batch_output}

            shutil.rmtree(args["output_dir"])
            tf.saved_model.simple_save(sess, args["output_dir"], inputs, outputs)

            print("Finished: %d ops in the final graph." % len(tf.get_default_graph().as_graph_def().node))
            print("##############################################################\n") 

        return

        return

    if args["mode"] == "deploy":
        # export the generator to a meta graph that can be imported later for standalone generation
        shape = [args["crop_size"], args["crop_size"], args["channels"]]
        input_image = tf.placeholder(dtype=tf.float32, shape=shape, name='input')
        batch_input = tf.expand_dims(input_image, axis=0)

        with tf.variable_scope("generator"):
            batch_output = pix2pix_model.deprocess(pix2pix_model.create_generator(args, pix2pix_model.preprocess(batch_input), args["channels"]))

        # output_image = tf.image.convert_image_dtype(batch_output, dtype=tf.uint8)[0]
        # output_image = tf.identity(output_image, name='output')
        output_image = tf.identity(batch_output[0], name='output')

        input_name = input_image.name.split(':')[0]
        output_name = output_image.name.split(':')[0]

         #[print(n.name) for n in tf.get_default_graph().as_graph_def().node]
        print("##############################################################\n")
        print("Input Name:", input_name)
        print("Output Name:", output_name)
        print("##############################################################\n")

        init_op = tf.global_variables_initializer()
        restore_saver = tf.train.Saver()
        export_saver = tf.train.Saver()

        with tf.Session() as sess:
            from tensorflow.tools.graph_transforms import TransformGraph

            sess.run(init_op)

            print("##############################################################\n")
            print("\nLoading model from checkpoint")
            checkpoint = tf.train.latest_checkpoint(args["checkpoint"])
            restore_saver.restore(sess, checkpoint)
            
            print("\nDeploying model, has %d ops" % len(tf.get_default_graph().as_graph_def().node))
            output_graph_def = graph_util.convert_variables_to_constants(sess, tf.get_default_graph().as_graph_def(), [output_name])

            if not args["transform_ops"] is None:
                print("\nStripping model and quantizing, has %d ops" % len(output_graph_def.node))
                transforms = args["transform_ops"].split(',')
                output_graph_def = TransformGraph(output_graph_def, [input_name], [output_name], transforms)

            #print("\n##### Optimizing model:") #issue: Didn't find expected Conv2D input to 'generator/encoder_2/batch_normalization/FusedBatchNorm'
            #output_graph_def = optimize_for_inference_lib.optimize_for_inference(output_graph_def, [input_name], [output_name], dtypes.float32.as_datatype_enum)

            print("\nOutputting model in binary format, %d ops" % len(output_graph_def.node))

            path = os.path.join(args["output_dir"], args["deploy_name"])
            with tf.gfile.GFile(path, "wb") as f:
                f.write(output_graph_def.SerializeToString())

            print("\nCreating selective registration header", path)
            
            with open(os.path.join(args["output_dir"], args["deploy_name"] + ".h"), "w") as f:
                header_str = selective_registration_header_lib.get_header([path], 'rawproto', 'NoOp:NoOp,_Recv:RecvOp,_Send:SendOp')
                f.write(header_str)

            [print(n.name) for n in output_graph_def.node]

            if os.environ.get('TF_ROOT') is not None:
                mm_path = os.path.join(os.environ.get('TF_ROOT'),"bazel-bin/tensorflow/contrib/util/convert_graphdef_memmapped_format")
                command = "{0} --in_graph='{1}' --out_graph='{1}'".format(mm_path, path)
                print("\nMemmapping binary: ", command)
                os.system(command)
            else:
                print(os.environ)
                print("##############################################################\n") 
                print("IMPORTANT: Be sure to run: bazel-bin/tensorflow/contrib/util/convert_graphdef_memmapped_format")
            

            print("Finished: %d ops in the final graph." % len(output_graph_def.node))
            print("##############################################################\n") 

        return

    examples = load_examples()
    print("examples count = %d" % examples.count)

    # inputs and targets are [batch_size, height, width, channels]
    model = pix2pix_model.create_model(args, examples.inputs, examples.targets, EPS)

    # undo colorization splitting on images that we use for display/output
    inputs = pix2pix_model.deprocess(examples.inputs)
    targets = pix2pix_model.deprocess(examples.targets)
    outputs = pix2pix_model.deprocess(model.outputs)

    def convert(image):
        if args["aspect_ratio"] != 1.0:
            # upscale to correct aspect ratio
            size = [args["crop_size"], int(round(args["crop_size"] * args["aspect_ratio"]))]
            image = tf.image.resize_images(image, size=size, method=tf.image.ResizeMethod.BICUBIC)

        return tf.image.convert_image_dtype(image, dtype=tf.uint8, saturate=True)

    # reverse any processing on images so they can be written to disk or displayed to user
    with tf.name_scope("convert_inputs"):
        converted_inputs = convert(inputs)

    with tf.name_scope("convert_targets"):
        converted_targets = convert(targets)

    with tf.name_scope("convert_outputs"):
        converted_outputs = convert(outputs)

    with tf.name_scope("encode_images"):
        display_fetches = {
            "paths": examples.paths,
            "inputs": tf.map_fn(tf.image.encode_png, converted_inputs, dtype=tf.string, name="input_pngs"),
            "targets": tf.map_fn(tf.image.encode_png, converted_targets, dtype=tf.string, name="target_pngs"),
            "outputs": tf.map_fn(tf.image.encode_png, converted_outputs, dtype=tf.string, name="output_pngs"),
        }

    # summaries
    with tf.name_scope("inputs_summary"):
        tf.summary.image("inputs", converted_inputs)

    with tf.name_scope("targets_summary"):
        tf.summary.image("targets", converted_targets)

    with tf.name_scope("outputs_summary"):
        tf.summary.image("outputs", converted_outputs)

    with tf.name_scope("predict_real_summary"):
        tf.summary.image("predict_real", tf.image.convert_image_dtype(model.predict_real, dtype=tf.uint8))

    with tf.name_scope("predict_fake_summary"):
        tf.summary.image("predict_fake", tf.image.convert_image_dtype(model.predict_fake, dtype=tf.uint8))

    tf.summary.scalar("discriminator_loss", model.discrim_loss)
    tf.summary.scalar("generator_loss_GAN", model.gen_loss_GAN)
    tf.summary.scalar("generator_loss_L1", model.gen_loss_L1)

    if args["gan_loss"] == "wgan":
        tf.summary.scalar("wgan_d_plus_g", model.discrim_loss + model.gen_loss_GAN)

    if model.gradient_penalty is not None:
        tf.summary.scalar("gradient_penalty", model.gradient_penalty)

    for var in tf.trainable_variables():
        tf.summary.histogram(var.op.name + "/values", var)

    for grad, var in model.discrim_grads_and_vars + model.gen_grads_and_vars:
        tf.summary.histogram(var.op.name + "/gradients", grad)

    with tf.name_scope("parameter_count"):
        parameter_count = tf.reduce_sum([tf.reduce_prod(tf.shape(v)) for v in tf.trainable_variables()])

    saver = tf.train.Saver(max_to_keep=1)

    logdir = args["output_dir"] if (args["trace_freq"] > 0 or args["summary_freq"] > 0) else None
    sv = tf.train.Supervisor(logdir=logdir, save_summaries_secs=0, saver=None)
    with sv.managed_session() as sess:
        print("parameter_count =", sess.run(parameter_count))

        if args["checkpoint"] is not None:
            print("loading model from checkpoint")
            checkpoint = tf.train.latest_checkpoint(args["checkpoint"])
            saver.restore(sess, checkpoint)

        max_steps = 2**32
        if args["max_epochs"] is not None:
            max_steps = examples.steps_per_epoch * args["max_epochs"]
        if args["max_steps"] is not None:
            max_steps = args["max_steps"]

        if args["mode"] == "test":
            # testing
            # at most, process the test data once
            start = time.time()
            max_steps = min(examples.steps_per_epoch, max_steps)
            for step in range(max_steps):
                results = sess.run(display_fetches)
                filesets = save_images(results, args["output_dir"])
                for i, f in enumerate(filesets):
                    print("evaluated image", f["name"])
                index_path = append_index(filesets, args["output_dir"])
            print("wrote index at", index_path)
            print("rate", (time.time() - start) / max_steps)
        else:
            # training
            start = time.time()

            for step in range(max_steps):
                def should(freq):
                    return freq > 0 and ((step + 1) % freq == 0 or step == max_steps - 1)

                options = None
                run_metadata = None
                if should(args["trace_freq"]):
                    options = tf.RunOptions(trace_level=tf.RunOptions.FULL_TRACE)
                    run_metadata = tf.RunMetadata()

                fetches = {
                    "train": model.train,
                    "global_step": sv.global_step,
                }

                if should(args["progress_freq"]):
                    fetches["discrim_loss"] = model.discrim_loss
                    fetches["gen_loss_GAN"] = model.gen_loss_GAN
                    fetches["gen_loss_L1"] = model.gen_loss_L1

                if should(args["summary_freq"]):
                    fetches["summary"] = sv.summary_op

                if should(args["display_freq"]):
                    fetches["display"] = display_fetches

                results = sess.run(fetches, options=options, run_metadata=run_metadata)

                if should(args["summary_freq"]):
                    print("recording summary")
                    sv.summary_writer.add_summary(results["summary"], results["global_step"])

                if should(args["display_freq"]):
                    print("saving display images")
                    filesets = save_images(results["display"], args["output_dir"], step=results["global_step"])
                    append_index(filesets, args["output_dir"], step=True)

                if should(args["trace_freq"]):
                    print("recording trace")
                    sv.summary_writer.add_run_metadata(run_metadata, "step_%d" % results["global_step"])

                if should(args["progress_freq"]):
                    # global_step will have the correct step count if we resume from a checkpoint
                    train_epoch = math.ceil(results["global_step"] / examples.steps_per_epoch)
                    train_step = (results["global_step"] - 1) % examples.steps_per_epoch + 1
                    rate = (step + 1) * args["batch_size"] / (time.time() - start)
                    remaining = (max_steps - step) * args["batch_size"] / rate
                    print("progress  epoch %d  step %d  image/sec %0.1f  remaining %dm" % (train_epoch, train_step, rate, remaining / 60))
                    print("discrim_loss", results["discrim_loss"])
                    print("gen_loss_GAN", results["gen_loss_GAN"])
                    print("gen_loss_L1", results["gen_loss_L1"])

                if should(args["save_freq"]):
                    print("saving model")
                    saver.save(sess, os.path.join(args["output_dir"], "model"), global_step=sv.global_step)

                if sv.should_stop():
                    break
