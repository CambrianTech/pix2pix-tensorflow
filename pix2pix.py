from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf
import numpy as np
import os
import json
import random
import collections
import math
import cv2

import shutil
import cambrian
from cambrian import utils
from pix2pix_model import Pix2PixModel

from tensorflow.python.framework import graph_util, dtypes
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
        "a_input": [],
        "b_input": [],
        "a_channels": [3],
        "b_channels": [3],

        "mode": "train",
        "model_dir": "models",
        "export_dir": "export",

        "epochs": 3000,
        
        "batch_size": 1,
        "ngf": 64,
        "ndf": 64,
        "init_stddev": 0.02,

        "crop_size": 256,
        "scale_size": 0,

        "flip": True,
        
        "lr_g": 0.0002,
        "lr_d": 0.0002,
        "beta1": 0.5,
        "beta2": 0.999,

        "gan_loss": "gan",
        "gp_weight": 0,
        "l1_weight": 100.0,
        "gan_weight": 1.0,
        
        "num_gpus": 1,

        "separable_conv": False,
        "no_disc_bn": False,
        "no_gen_bn": False,
        "layer_norm": False,
        "angle_output": False,
    }

def get_serving_input_receiver_fn(a_specs):
	def serving_input_receiver_fn():
		inputs = {
            cambrian.nn.get_input_name(i): tf.placeholder(spec.dtype, (None, spec.crop_size, spec.crop_size, spec.channels))
            for i, spec in enumerate(a_specs)
        }
		return tf.estimator.export.ServingInputReceiver(inputs, inputs)
	return serving_input_receiver_fn

def get_specs_from_args(args):
    # If we only have single elements for inputs or channels
    # make a list out of them
    def ensure_list(x):
        if not isinstance(x, list) and not isinstance(x, tuple):
            return [x]
        return x
    a_input, b_input = ensure_list(args["a_input"]), ensure_list(args["b_input"])
    a_channels, b_channels = ensure_list(args["a_channels"]), ensure_list(args["b_channels"])

    num_a, num_b = len(a_input), len(b_input)

    # If only one channel was passed use that channel for all inputs
    if len(a_channels) == 1:
        a_channels = a_channels * num_a

    if len(b_channels) == 1:
        b_channels = b_channels * num_b

    # Other args
    scale_size = args["scale_size"]
    crop_size = args["crop_size"]

    if scale_size <= 0:
        scale_size = crop_size

    # Combine a and b so we can process them together
    # and split them after
    matches = a_input + b_input
    channels = a_channels + b_channels

    specs = [cambrian.nn.IOSpecification(m, c, scale_size, crop_size)
                for m, c in zip(matches, channels)]

    a_specs = specs[:num_a]
    b_specs = specs[num_a:]

    return a_specs, b_specs

@ex.automain
@LogFileWriter(ex)
def main(args, _seed):
    print("python pix2pix.py with \"args =", args, "\"")

    tf.set_random_seed(_seed)
    np.random.seed(_seed)
    random.seed(_seed)

    distribute_strategy = cambrian.nn.get_distribution_strategy(args["num_gpus"])

    session_config = tf.ConfigProto()
    session_config.gpu_options.allow_growth = True

    run_config = tf.estimator.RunConfig(
		model_dir=args["model_dir"],
		train_distribute=distribute_strategy,
		eval_distribute=distribute_strategy,
        session_config=session_config,
	)

    a_specs, b_specs = get_specs_from_args(args)
    args["a_specs"], args["b_specs"] = a_specs, b_specs

    print("A specs:", a_specs)
    print("B specs:", b_specs)

    model = Pix2PixModel(args)
    model_fn = cambrian.nn.get_model_fn_ab(model, a_specs, b_specs)

    estimator = tf.estimator.Estimator(model_fn=model_fn, config=run_config, params=args)

    print("Start", args["mode"])

    if args["mode"] == "train":
        train_input_fn_args = cambrian.nn.InputFnArgs.train(epochs=args["epochs"], batch_size=args["batch_size"])
        train_input_fn = cambrian.nn.get_input_fn_ab(a_specs, b_specs, train_input_fn_args)
        train_spec = tf.estimator.TrainSpec(train_input_fn)

        eval_input_fn_args = cambrian.nn.InputFnArgs.eval(epochs=args["epochs"], batch_size=args["batch_size"])
        eval_input_fn = cambrian.nn.get_input_fn_ab(a_specs, b_specs, eval_input_fn_args)
        eval_spec = tf.estimator.EvalSpec(eval_input_fn)

        tf.estimator.train_and_evaluate(estimator, train_spec, eval_spec)
    elif args["mode"] == "test":
        eval_input_fn_args = cambrian.nn.InputFnArgs.eval(epochs=args["epochs"], batch_size=args["batch_size"])
        eval_input_fn = cambrian.nn.get_input_fn_ab(a_specs, b_specs, eval_input_fn_args)
        estimator.evaluate(eval_input_fn)
    elif args["mode"] == "export":
        estimator.export_saved_model(args["export_dir"], get_serving_input_receiver_fn(a_specs))
    else:
        print("Unknown mode", args.mode)
