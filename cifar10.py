# Copyright 2019, The TensorFlow Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Training a CNN on MNIST with Keras and the DP SGD optimizer."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from absl import app
from absl import flags
from absl import logging

import numpy as np
import tensorflow as tf
import keras_applications

from privacy.analysis.rdp_accountant import compute_rdp
from privacy.analysis.rdp_accountant import get_privacy_spent
from dp_optimizer import DPGradientDescentGaussianOptimizer


flags.DEFINE_boolean('dpsgd', True, 'If True, train with DP-SGD. If False, train with vanilla SGD.')
flags.DEFINE_float('learning_rate', 1e-3, 'Learning rate for training')
flags.DEFINE_float('noise_multiplier', 1.1, 'Ratio of the standard deviation to the clipping norm')
flags.DEFINE_float('l2_norm_clip', 1.0, 'Clipping norm')
flags.DEFINE_integer('batch_size', 16, 'Batch size')
flags.DEFINE_integer('epochs', 200, 'Number of epochs')
flags.DEFINE_integer('microbatches', 16, 'Number of microbatches (must evenly divide batch_size)')
flags.DEFINE_float('delta', 1e-5, 'Delta') # Delta is set to 1e-5 because MNIST has 50000 training points.

FLAGS = flags.FLAGS


def compute_epsilon(steps):
  """Computes epsilon value for given hyperparameters."""
  if FLAGS.noise_multiplier == 0.0:
    return float('inf')
  orders = [1 + x / 10. for x in range(1, 100)] + list(range(12, 64))
  sampling_probability = FLAGS.batch_size / 60000
  rdp = compute_rdp(q=sampling_probability, noise_multiplier=FLAGS.noise_multiplier, steps=steps, orders=orders)
  return get_privacy_spent(orders, rdp, target_delta=FLAGS.delta)[0]


class EpsilonPrintingCallback(tf.keras.callbacks.Callback):
  """Callback for Keras model to evaluate epsilon after every epoch."""
  def __init__(self):
    self.eps_history = []

  def on_epoch_end(self, epoch, logs=None):
    if FLAGS.dpsgd:
      eps = compute_epsilon((epoch + 1) * (60000 // FLAGS.batch_size))
      self.eps_history.append(eps)
      print(', eps = {}'.format(eps))


def load_mnist():
  """Loads MNIST and preprocesses to combine training and validation data."""
  train, test = tf.keras.datasets.cifar10.load_data()
  train_data, train_labels = train
  test_data, test_labels = test

  train_data = np.array(train_data, dtype=np.float32) / 255
  test_data = np.array(test_data, dtype=np.float32) / 255

  train_data = train_data.reshape(train_data.shape[0], 32, 32, 3)
  test_data = test_data.reshape(test_data.shape[0], 32, 32, 3)

  train_labels = np.array(train_labels, dtype=np.int32)
  test_labels = np.array(test_labels, dtype=np.int32)

  train_labels = tf.keras.utils.to_categorical(train_labels, num_classes=10)
  test_labels = tf.keras.utils.to_categorical(test_labels, num_classes=10)

  return train_data, train_labels, test_data, test_labels


def resnet_block(x, filters, kernel_size=3, stride=1,
           conv_shortcut=False, name=None):
    """A residual block.
    # Arguments
        x: input tensor.
        filters: integer, filters of the bottleneck layer.
        kernel_size: default 3, kernel size of the bottleneck layer.
        stride: default 1, stride of the first layer.
        conv_shortcut: default False, use convolution shortcut if True,
            otherwise identity shortcut.
        name: string, block label.
    # Returns
        Output tensor for the residual block.
    """
    bn_axis = 3 if tf.keras.backend.image_data_format() == 'channels_last' else 1

    preact = tf.keras.layers.BatchNormalization(axis=bn_axis, epsilon=1.001e-5,
                                       name=name + '_preact_bn')(x)
    preact = tf.keras.layers.Activation('relu', name=name + '_preact_relu')(preact)

    if conv_shortcut is True:
        shortcut = tf.keras.layers.Conv2D(4 * filters, 1, strides=stride,
                                 name=name + '_0_conv')(preact)
    else:
        shortcut = tf.keras.layers.MaxPooling2D(1, strides=stride)(x) if stride > 1 else x

    x = tf.keras.layers.Conv2D(filters, 1, strides=1, use_bias=False,
                      name=name + '_1_conv')(preact)
    x = tf.keras.layers.BatchNormalization(axis=bn_axis, epsilon=1.001e-5,
                                  name=name + '_1_bn')(x)
    x = tf.keras.layers.Activation('relu', name=name + '_1_relu')(x)

    x = tf.keras.layers.ZeroPadding2D(padding=((1, 1), (1, 1)), name=name + '_2_pad')(x)
    x = tf.keras.layers.Conv2D(filters, kernel_size, strides=stride,
                      use_bias=False, name=name + '_2_conv')(x)
    x = tf.keras.layers.BatchNormalization(axis=bn_axis, epsilon=1.001e-5,
                                  name=name + '_2_bn')(x)
    x = tf.keras.layers.Activation('relu', name=name + '_2_relu')(x)

    x = tf.keras.layers.Conv2D(4 * filters, 1, name=name + '_3_conv')(x)
    x = tf.keras.layers.Add(name=name + '_out')([shortcut, x])
    return x


def resnet_stack(x, filters, blocks, stride1=2, name=None):
    """A set of stacked residual blocks.
    # Arguments
        x: input tensor.
        filters: integer, filters of the bottleneck layer in a block.
        blocks: integer, blocks in the stacked blocks.
        stride1: default 2, stride of the first layer in the first block.
        name: string, stack label.
    # Returns
        Output tensor for the stacked blocks.
    """
    x = resnet_block(x, filters, conv_shortcut=True, name=name + '_block1')
    for i in range(2, blocks):
        x = resnet_block(x, filters, name=name + '_block' + str(i))
    x = resnet_block(x, filters, stride=stride1, name=name + '_block' + str(blocks))
    return x


def ResNet(stack_fn, default_size=32, model_name='resnet', classes=10, **kwargs):
    """Instantiates the ResNet architecture.
    # Arguments
        stack_fn: a function that returns output tensor for the
            stacked residual blocks.
        default_size: default size of image.
        model_name: string, model name.
        classes: number of classes to classify images into.
    # Returns
        A Keras model instance.
    """
    # Determine proper input shape
    input_shape = keras_applications.imagenet_utils._obtain_input_shape(None,
                                      default_size=default_size,
                                      min_size=32,
                                      data_format=tf.keras.backend.image_data_format(),
                                      require_flatten=True,
                                      weights=None)

    input_tensor = tf.keras.layers.Input(shape=input_shape)

    bn_axis = 3 if tf.keras.backend.image_data_format() == 'channels_last' else 1

    x = tf.keras.layers.ZeroPadding2D(padding=((3, 3), (3, 3)), name='conv1_pad')(input_tensor)
    x = tf.keras.layers.Conv2D(64, 7, strides=2, use_bias=False, name='conv1_conv')(x)

    x = tf.keras.layers.ZeroPadding2D(padding=((1, 1), (1, 1)), name='pool1_pad')(x)
    x = tf.keras.layers.MaxPooling2D(3, strides=2, name='pool1_pool')(x)

    x = stack_fn(x)
    x = tf.keras.layers.BatchNormalization(axis=bn_axis, epsilon=1.001e-5,
                                  name='post_bn')(x)
    x = tf.keras.layers.Activation('relu', name='post_relu')(x)

    x = tf.keras.layers.GlobalAveragePooling2D(name='avg_pool')(x)
    x = tf.keras.layers.Dense(classes, activation='softmax', name='probs')(x)

    # Create model.
    model = tf.keras.models.Model(input_tensor, x, name=model_name)
    return model


def ResNet18V2(default_size=16, classes=10, **kwargs):
    def stack_fn(x):
        x = resnet_stack(x, 16, 2, name='conv2')
        x = resnet_stack(x, 32, 2, name='conv3')
        x = resnet_stack(x, 64, 2, name='conv4')
        x = resnet_stack(x, 128, 2, stride1=1, name='conv5')
        return x
    return ResNet(stack_fn, default_size, 'resnet18v2', classes, **kwargs)


def main(unused_argv):
  logging.set_verbosity(logging.INFO)
  if FLAGS.dpsgd and FLAGS.batch_size % FLAGS.microbatches != 0:
    raise ValueError('Number of microbatches should divide evenly batch_size')

  # Load training and test data.
  train_data, train_labels, test_data, test_labels = load_mnist()

  # Define a sequential Keras model
  model = ResNet18V2(default_size=32)
  print(model.summary())

  if FLAGS.dpsgd:
    optimizer = DPGradientDescentGaussianOptimizer(
        l2_norm_clip=FLAGS.l2_norm_clip,
        noise_multiplier=FLAGS.noise_multiplier,
        num_microbatches=FLAGS.microbatches,
        learning_rate=FLAGS.learning_rate)
    # Compute vector of per-example loss rather than its mean over a minibatch.
    loss = tf.keras.losses.CategoricalCrossentropy(from_logits=True, reduction=tf.compat.v1.losses.Reduction.NONE)
  else:
    optimizer = tf.compat.v1.train.GradientDescentOptimizer(learning_rate=FLAGS.learning_rate)
    loss = tf.keras.losses.CategoricalCrossentropy(from_logits=True)

  # Compile model with Keras
  model.compile(optimizer=optimizer, loss=loss, metrics=['accuracy'])

  # Train model with Keras
  eps_callback = EpsilonPrintingCallback()
  fit_history = model.fit(train_data, train_labels, epochs=FLAGS.epochs, validation_data=(test_data, test_labels), batch_size=FLAGS.batch_size, callbacks=[eps_callback])
  eps_history = eps_callback.eps_history
  val_acc_history = fit_history.history['val_accuracy']
  with open('cifar_{}_delta_{}_lr_{}.txt'.format('dpsgd' if FLAGS.dpsgd else 'sgd', FLAGS.delta, FLAGS.learning_rate), 'w') as f:
    f.write('eps: {}\n'.format(eps_history))
    f.write('validation acc: {}\n'.format(val_acc_history))

if __name__ == '__main__':
  app.run(main)