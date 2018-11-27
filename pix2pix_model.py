import tensorflow as tf
import cambrian

EPS = 1e-12

class Pix2PixModel(cambrian.nn.ModelBase):
    def __init__(self, args):
        super().__init__()
        self.args = args
        self.is_general_setup = False
        self.is_train_setup = False

    def _setup_general(self):
        if self.is_general_setup:
           raise Exception("General was already set up.")

        with tf.variable_scope("generator"):
            out_channels = sum([spec.channels for spec in self.args["b_specs"]])
            self._outputs = create_generator(self.args, self.inputs, out_channels)

    def _setup_train(self):
        if self.is_train_setup:
            raise Exception("Train was already set up.")

        real_inputs = [self.targets]
        fake_inputs = [self.outputs]
        img_inputs = [self.inputs]

        # Collect the losses for different scales and average them later
        multiscale_gradient_penalties = [] if self.args["gp_weight"] and self.args["gp_weight"] > 0 else None
        multiscale_gen_losses_gan = []
        multiscale_discrim_losses_gan = []

        for _ in range(self.args["num_downsampled_discs"]):
            size = real_inputs[-1].shape[1:3]
            assert size == fake_inputs[-1].shape[1:3]
            assert size == img_inputs[-1].shape[1:3]
            real_inputs.append(tf.image.resize_images(real_inputs[-1], (size[0] // 2, size[1] // 2), tf.image.ResizeMethod.BILINEAR))
            fake_inputs.append(tf.image.resize_images(fake_inputs[-1], (size[0] // 2, size[1] // 2), tf.image.ResizeMethod.BILINEAR))
            img_inputs.append(tf.image.resize_images(img_inputs[-1], (size[0] // 2, size[1] // 2), tf.image.ResizeMethod.BILINEAR))

        for scale_index, (inputs, outputs, targets) in enumerate(reversed(list(zip(img_inputs, fake_inputs, real_inputs)))):
            # create two copies of discriminator, one for real pairs and one for fake pairs
            # they share the same underlying variables
            with tf.name_scope("real_discriminator_%d" % scale_index):
                with tf.variable_scope("discriminator_%d" % scale_index):
                    # 2x [batch, height, width, channels] => [batch, 30, 30, 1]
                    predict_real = create_discriminator(self.args, inputs, targets)

            with tf.name_scope("fake_discriminator_%d" % scale_index):
                with tf.variable_scope("discriminator_%d" % scale_index, reuse=True):
                    # 2x [batch, height, width, channels] => [batch, 30, 30, 1]
                    predict_fake = create_discriminator(self.args, inputs, outputs)

            with tf.name_scope("discriminator_loss_%d" % scale_index):
                # minimizing -tf.log will try to get inputs to 1
                # predict_real => 1
                # predict_fake => 0
                if self.args["gan_loss"] == "gan":
                    discrim_loss = tf.reduce_mean(-(tf.log(tf.sigmoid(predict_real) + EPS) + tf.log(1 - tf.sigmoid(predict_fake) + EPS)))
                elif self.args["gan_loss"] == "wgan":
                    discrim_loss = tf.reduce_mean(predict_fake - predict_real)
                elif self.args["gan_loss"] == "ganqp":
                    diff = outputs - targets
                    # They chose a 10x multiplier for the norm in the paper, but seems like another hyperparameter.
                    # Also choosing L1 norm instead of L2 works but L2 gave them slightly better results.
                    norm_axes = list(range(1, len(diff.shape))) # All axes except first (batch)
                    norm = 10 * tf.sqrt(tf.reduce_mean(tf.square(diff), axis=norm_axes, keepdims=True))
                    #norm = 10 * tf.reduce_mean(tf.abs(diff), axis=norm_axes, keepdims=True)
                    disc_diff = predict_fake - predict_real
                    discrim_loss = tf.reduce_mean(disc_diff + 0.5 * tf.square(disc_diff) / norm)
                else:
                    raise Exception("Unknown gan_loss:", self.args["gan_loss"])
                multiscale_discrim_losses_gan.append(discrim_loss)

            with tf.name_scope("generator_loss_%d" % scale_index):
                # predict_fake => 1
                # abs(self.targets - self.outputs) => 0
                if self.args["gan_loss"] == "gan":
                    gen_loss_GAN = tf.reduce_mean(-tf.log(tf.sigmoid(predict_fake) + EPS))
                elif self.args["gan_loss"] == "wgan" or self.args["gan_loss"] == "ganqp":
                    gen_loss_GAN = -tf.reduce_mean(predict_fake)
                else:
                    raise Exception("Unknown gan_loss:", self.args["gan_loss"])
                multiscale_gen_losses_gan.append(gen_loss_GAN)

            # Gradient penalty
            if multiscale_gradient_penalties is not None:
                with tf.name_scope("gradient_penalty_%d" % scale_index):
                    rand_interp = random_interpolate(targets, outputs)
                    with tf.variable_scope("discriminator_%d" % scale_index, reuse=True):
                        predict_rand_interp = create_discriminator(self.args, inputs, rand_interp)
                    gradient_penalty = self.args["gp_weight"] * get_gradient_penalty(self.args, predict_rand_interp, rand_interp)
                    multiscale_gradient_penalties.append(gradient_penalty)

        # Calculate actual losses averaging over multiscale
        gen_loss_GAN = tf.reduce_mean(multiscale_gen_losses_gan)
        discrim_loss = tf.reduce_mean(multiscale_discrim_losses_gan)
        discrim_total_loss = discrim_loss
        gen_loss_L1 = tf.reduce_mean(tf.abs(self.targets - self.outputs))
        gen_loss = gen_loss_GAN * self.args["gan_weight"] + gen_loss_L1 * self.args["l1_weight"]
        gradient_penalty = None if multiscale_gradient_penalties is None else tf.reduce_mean(multiscale_gradient_penalties) * self.args["gp_weight"]
        if gradient_penalty is not None:
            discrim_total_loss += gradient_penalty

        with tf.name_scope("discriminator_train"):
            discrim_tvars = [var for var in tf.trainable_variables() if var.name.startswith("discriminator")]
            discrim_optim = tf.train.AdamOptimizer(self.args["lr_d"], self.args["beta1"], self.args["beta2"])
            discrim_grads_and_vars = discrim_optim.compute_gradients(discrim_total_loss, var_list=discrim_tvars)
            discrim_train = discrim_optim.apply_gradients(discrim_grads_and_vars)

        with tf.name_scope("generator_train"):
            with tf.control_dependencies([discrim_train]):
                gen_tvars = [var for var in tf.trainable_variables() if var.name.startswith("generator")]
                gen_optim = tf.train.AdamOptimizer(self.args["lr_g"], self.args["beta1"], self.args["beta2"])
                gen_grads_and_vars = gen_optim.compute_gradients(gen_loss, var_list=gen_tvars)
                gen_train = gen_optim.apply_gradients(gen_grads_and_vars, global_step=tf.train.get_global_step())

        # "gen_train" also includes discrim_train through control dependencies
        self._train_op = gen_train
        self._loss = gen_loss_L1
        
        # Metrics
        self.metrics["gen_l1"] = gen_loss_L1
        self.metrics["gen_gan"] = gen_loss_GAN
        self.metrics["disc_gan"] = discrim_loss
        self.metrics["gen_total"] = gen_loss
        self.metrics["disc_total"] = discrim_total_loss
        if gradient_penalty is not None:
            self.metrics["disc_gp"] = gradient_penalty
            for scale_index, gp in enumerate(multiscale_gradient_penalties):
                self.metrics["disc_gp_%d" % scale_index] = gp
        for scale_index, (d_loss, g_loss) in enumerate(zip(multiscale_discrim_losses_gan, multiscale_gen_losses_gan)):
            self.metrics["disc_loss_%d" % scale_index] = d_loss
            self.metrics["gen_loss_%d" % scale_index] = g_loss

        # Summaries
        summaries = []

        for spec in self.args["a_specs"]:
            with tf.name_scope("inputs_summary"):
                summaries.append(tf.summary.image("inputs_%d" % spec.index, tf.image.convert_image_dtype(self.inputs[:, :, :, spec.start_channel:spec.start_channel+spec.channels], dtype=tf.uint8)))
        
        for spec in self.args["b_specs"]:
            with tf.name_scope("targets_summary"):
                summaries.append(tf.summary.image("targets_%d" % spec.index, tf.image.convert_image_dtype(self.targets[:, :, :, spec.start_channel:spec.start_channel+spec.channels], dtype=tf.uint8)))
            with tf.name_scope("outputs_summary"):
                summaries.append(tf.summary.image("outputs_%d" % spec.index, tf.image.convert_image_dtype(self.outputs[:, :, :, spec.start_channel:spec.start_channel+spec.channels], dtype=tf.uint8)))

        with tf.name_scope("predict_real_summary"):
            summaries.append(tf.summary.image("predict_real", tf.image.convert_image_dtype(predict_real, dtype=tf.uint8)))

        with tf.name_scope("predict_fake_summary"):
            summaries.append(tf.summary.image("predict_fake", tf.image.convert_image_dtype(predict_fake, dtype=tf.uint8)))

        with tf.name_scope("scalar_summaries"):
            summaries.append(tf.summary.scalar("discriminator_loss", discrim_loss))
            summaries.append(tf.summary.scalar("generator_loss_GAN", gen_loss_GAN))

            summaries.append(tf.summary.scalar("generator_loss_L1", gen_loss_L1))

            if self.args["gan_loss"] == "wgan" or self.args["gan_loss"] == "ganqp":
                summaries.append(tf.summary.scalar("wgan_d_minus_g", discrim_loss - gen_loss_GAN))

            if gradient_penalty is not None:
                summaries.append(tf.summary.scalar("gradient_penalty", gradient_penalty))

        if len(multiscale_discrim_losses_gan) > 1:
            with tf.name_scope("multiscale_scalar_summaries"):
                for scale_index, (d_loss, g_loss) in enumerate(zip(multiscale_discrim_losses_gan, multiscale_gen_losses_gan)):
                    summaries.append(tf.summary.scalar("discriminator_loss_downsampled_%d" % scale_index, d_loss))
                    summaries.append(tf.summary.scalar("generator_loss_downsampled_%d" % scale_index, g_loss))
                if multiscale_gradient_penalties is not None:
                    for scale_index, gp in enumerate(multiscale_gradient_penalties):
                        summaries.append(tf.summary.scalar("gradient_penalty_downsampled_%d" % scale_index, gp))

        for var in tf.trainable_variables():
            summaries.append(tf.summary.histogram(var.op.name + "/values", var))

        for grad, var in discrim_grads_and_vars + gen_grads_and_vars:
            summaries.append(tf.summary.histogram(var.op.name + "/gradients", grad))

        self._summary_op = tf.summary.merge(summaries)

    def set_inputs(self, inputs):
        super().set_inputs(inputs)
        self._setup_general()

    def set_targets(self, targets):
        super().set_targets(targets)
        self._setup_train()

def random_interpolate(a, b):
    shape = tf.concat((tf.shape(a)[0:1], tf.tile([1], [a.shape.ndims - 1])), axis=0)
    alpha = tf.random_uniform(shape=shape, minval=0., maxval=1.)
    inter = a + alpha * (b - a)
    inter.set_shape(a.get_shape().as_list())
    return inter

def get_gradient_penalty(args, pred, x):
    gradients = tf.gradients(pred, x)[0]
    slopes = tf.sqrt(tf.reduce_sum(tf.square(gradients), reduction_indices=[-1]))
    gp = tf.reduce_mean((slopes - 1.)**2)
    return gp

def discrim_conv(batch_input, out_channels, stride, init_stddev):
    padded_input = tf.pad(batch_input, [[0, 0], [1, 1], [1, 1], [0, 0]], mode="CONSTANT")
    return tf.layers.conv2d(padded_input, out_channels, kernel_size=4, strides=(stride, stride), padding="valid", kernel_initializer=tf.random_normal_initializer(0, init_stddev))

def gen_conv(batch_input, out_channels, init_stddev, separable_conv):
    # [batch, in_height, in_width, in_channels] => [batch, out_height, out_width, out_channels]
    initializer = tf.random_normal_initializer(0, init_stddev)
    if separable_conv:
        return tf.layers.separable_conv2d(batch_input, out_channels, kernel_size=4, strides=(2, 2), padding="same", depthwise_initializer=initializer, pointwise_initializer=initializer)
    else:
        return tf.layers.conv2d(batch_input, out_channels, kernel_size=4, strides=(2, 2), padding="same", kernel_initializer=initializer)

def gen_deconv(batch_input, out_channels, init_stddev, separable_conv):
    # [batch, in_height, in_width, in_channels] => [batch, out_height, out_width, out_channels]
    initializer = tf.random_normal_initializer(0, init_stddev)
    if separable_conv:
        _b, h, w, _c = batch_input.shape
        resized_input = tf.image.resize_images(batch_input, [h * 2, w * 2], method=tf.image.ResizeMethod.NEAREST_NEIGHBOR)
        return tf.layers.separable_conv2d(resized_input, out_channels, kernel_size=4, strides=(1, 1), padding="same", depthwise_initializer=initializer, pointwise_initializer=initializer)
    else:
        return tf.layers.conv2d_transpose(batch_input, out_channels, kernel_size=4, strides=(2, 2), padding="same", kernel_initializer=initializer)

def batchnorm(inputs, init_stddev):
    return tf.layers.batch_normalization(inputs, axis=3, epsilon=1e-5, momentum=0.1, training=True, gamma_initializer=tf.random_normal_initializer(1.0, init_stddev))

def layernorm(inputs):
    return tf.contrib.layers.layer_norm(inputs)

def create_generator(args, generator_inputs, generator_outputs_channels):
    layers = []

    # encoder_1: [batch, 256, 256, in_channels] => [batch, 128, 128, ngf]
    with tf.variable_scope("encoder_1"):
        output = gen_conv(generator_inputs, args["ngf"], args["init_stddev"], args["separable_conv"])
        layers.append(output)

    layer_specs = [
        args["ngf"] * 2, # encoder_2: [batch, 128, 128, ngf] => [batch, 64, 64, ngf * 2]
        args["ngf"] * 4, # encoder_3: [batch, 64, 64, ngf * 2] => [batch, 32, 32, ngf * 4]
        args["ngf"] * 8, # encoder_4: [batch, 32, 32, ngf * 4] => [batch, 16, 16, ngf * 8]
        args["ngf"] * 8, # encoder_5: [batch, 16, 16, ngf * 8] => [batch, 8, 8, ngf * 8]
        args["ngf"] * 8, # encoder_6: [batch, 8, 8, ngf * 8] => [batch, 4, 4, ngf * 8]
        args["ngf"] * 8, # encoder_7: [batch, 4, 4, ngf * 8] => [batch, 2, 2, ngf * 8]
        args["ngf"] * 8, # encoder_8: [batch, 2, 2, ngf * 8] => [batch, 1, 1, ngf * 8]
    ]

    for out_channels in layer_specs:
        with tf.variable_scope("encoder_%d" % (len(layers) + 1)):
            rectified = tf.nn.leaky_relu(layers[-1], 0.2)
            # [batch, in_height, in_width, in_channels] => [batch, in_height/2, in_width/2, out_channels]
            convolved = gen_conv(rectified, out_channels, args["init_stddev"], args["separable_conv"])
            if not args["no_gen_bn"]:
                convolved = layernorm(convolved) if args["layer_norm"] else batchnorm(convolved, args["init_stddev"])
            layers.append(convolved)

    layer_specs = [
        (args["ngf"] * 8, 0.5),   # decoder_8: [batch, 1, 1, ngf * 8] => [batch, 2, 2, ngf * 8 * 2]
        (args["ngf"] * 8, 0.5),   # decoder_7: [batch, 2, 2, ngf * 8 * 2] => [batch, 4, 4, ngf * 8 * 2]
        (args["ngf"] * 8, 0.5),   # decoder_6: [batch, 4, 4, ngf * 8 * 2] => [batch, 8, 8, ngf * 8 * 2]
        (args["ngf"] * 8, 0.0),   # decoder_5: [batch, 8, 8, ngf * 8 * 2] => [batch, 16, 16, ngf * 8 * 2]
        (args["ngf"] * 4, 0.0),   # decoder_4: [batch, 16, 16, ngf * 8 * 2] => [batch, 32, 32, ngf * 4 * 2]
        (args["ngf"] * 2, 0.0),   # decoder_3: [batch, 32, 32, ngf * 4 * 2] => [batch, 64, 64, ngf * 2 * 2]
        (args["ngf"], 0.0),       # decoder_2: [batch, 64, 64, ngf * 2 * 2] => [batch, 128, 128, ngf * 2]
    ]

    num_encoder_layers = len(layers)
    for decoder_layer, (out_channels, dropout) in enumerate(layer_specs):
        skip_layer = num_encoder_layers - decoder_layer - 1
        with tf.variable_scope("decoder_%d" % (skip_layer + 1)):
            if decoder_layer == 0:
                # first decoder layer doesn't have skip connections
                # since it is directly connected to the skip_layer
                input = layers[-1]
            else:
                input = tf.concat([layers[-1], layers[skip_layer]], axis=3)

            rectified = tf.nn.relu(input)
            # [batch, in_height, in_width, in_channels] => [batch, in_height*2, in_width*2, out_channels]
            output = gen_deconv(rectified, out_channels, args["init_stddev"], args["separable_conv"])
            if not args["no_gen_bn"]:
                output = layernorm(output) if args["layer_norm"] else batchnorm(output, args["init_stddev"])

            if dropout > 0.0:
                output = tf.nn.dropout(output, keep_prob=1 - dropout)

            layers.append(output)

    # decoder_1: [batch, 128, 128, ngf * 2] => [batch, 256, 256, generator_outputs_channels]
    with tf.variable_scope("decoder_1"):
        input = tf.concat([layers[-1], layers[0]], axis=3)
        rectified = tf.nn.relu(input)

        # Make sure angle output is list of correct length
        # TODO: Probably want this on the output spec (maybe add a dict to it or new subclass)
        is_angle_output = args["angle_output"]
        if not isinstance(is_angle_output, list) and not isinstance(is_angle_output, tuple):
            is_angle_output = [is_angle_output]
        if len(is_angle_output) != len(args["b_specs"]):
            is_angle_output *= len(args["b_specs"])

        outputs = []
        for is_angle, output_spec in zip(is_angle_output, args["b_specs"]):
            if is_angle:
                assert output_spec.channels == 3

                # Produce 3D unit vector from 2 angles
                angles = gen_deconv(rectified, 2, args["init_stddev"], args["separable_conv"])
                angle_x = angles[:, :, :, 0:1]
                angle_y = angles[:, :, :, 1:2]

                sin_x, cos_x = tf.sin(angle_x), tf.cos(angle_x)
                sin_y, cos_y = tf.sin(angle_y), tf.cos(angle_y)
                output_x = sin_x * cos_y
                output_y = cos_x * cos_y
                output_z = sin_y

                output = tf.concat((output_x, output_y, output_z), axis=-1, )

                # [-1, 1] -> [0, 1]
                output = tf.div(output + 1., 2., name="output_%d" % output_spec.index)
            else:
                output = gen_deconv(rectified, output_spec.channels, args["init_stddev"], args["separable_conv"])
                output = tf.sigmoid(output, name="output_%d" % output_spec.index)
            outputs.append(output)

        # Combine all outputs along channels
        print(outputs)
        output = tf.concat(outputs, axis=-1, name="output")

        layers.append(output)

    return layers[-1]

def create_discriminator(args, discrim_inputs, discrim_targets):
    n_layers = 3
    layers = []

    # 2x [batch, height, width, in_channels] => [batch, height, width, in_channels * 2]
    input = tf.concat([discrim_inputs, discrim_targets], axis=3)

    # layer_1: [batch, 256, 256, in_channels * 2] => [batch, 128, 128, ndf]
    with tf.variable_scope("layer_1"):
        convolved = discrim_conv(input, args["ndf"], 2, args["init_stddev"])
        rectified = tf.nn.leaky_relu(convolved, 0.2)
        layers.append(rectified)

    # layer_2: [batch, 128, 128, ndf] => [batch, 64, 64, ndf * 2]
    # layer_3: [batch, 64, 64, ndf * 2] => [batch, 32, 32, ndf * 4]
    # layer_4: [batch, 32, 32, ndf * 4] => [batch, 31, 31, ndf * 8]
    for i in range(n_layers):
        with tf.variable_scope("layer_%d" % (len(layers) + 1)):
            out_channels = args["ndf"] * min(2**(i+1), 8)
            stride = 1 if i == n_layers - 1 else 2  # last layer here has stride 1
            convolved = discrim_conv(layers[-1], out_channels, stride, args["init_stddev"])
            if not args["no_disc_bn"]:
                convolved = layernorm(convolved) if args["layer_norm"] else batchnorm(convolved, args["init_stddev"])
            rectified = tf.nn.leaky_relu(convolved, 0.2)
            layers.append(rectified)

    # layer_5: [batch, 31, 31, ndf * 8] => [batch, 30, 30, 1]
    with tf.variable_scope("layer_%d" % (len(layers) + 1)):
        output = discrim_conv(rectified, 1, 1, args["init_stddev"])
        layers.append(output)

    return layers[-1]