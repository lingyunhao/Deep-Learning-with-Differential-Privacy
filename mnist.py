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

from privacy.analysis.rdp_accountant import compute_rdp
from privacy.analysis.rdp_accountant import get_privacy_spent
from dp_optimizer import DPGradientDescentGaussianOptimizer

GradientDescentOptimizer = tf.compat.v1.train.GradientDescentOptimizer

flags.DEFINE_boolean('dpsgd', True, 'If True, train with DP-SGD. If False, train with vanilla SGD.')
flags.DEFINE_float('learning_rate', 1e-3, 'Learning rate for training')
flags.DEFINE_float('noise_multiplier', 1.1, 'Ratio of the standard deviation to the clipping norm')
flags.DEFINE_float('l2_norm_clip', 1.0, 'Clipping norm')
flags.DEFINE_integer('batch_size', 250, 'Batch size')
flags.DEFINE_integer('epochs', 400, 'Number of epochs')
flags.DEFINE_integer('microbatches', 250, 'Number of microbatches (must evenly divide batch_size)')
flags.DEFINE_string('model_dir', None, 'Model directory')

FLAGS = flags.FLAGS
delta = 1e-2  # Delta is set to 1e-5 because MNIST has 60000 training points.


def compute_epsilon(steps):
  """Computes epsilon value for given hyperparameters."""
  if FLAGS.noise_multiplier == 0.0:
    return float('inf')
  orders = [1 + x / 10. for x in range(1, 100)] + list(range(12, 64))
  sampling_probability = FLAGS.batch_size / 60000
  rdp = compute_rdp(q=sampling_probability, noise_multiplier=FLAGS.noise_multiplier, steps=steps, orders=orders)
  return get_privacy_spent(orders, rdp, target_delta=delta)[0]


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
  train, test = tf.keras.datasets.mnist.load_data()
  train_data, train_labels = train
  test_data, test_labels = test

  train_data = np.array(train_data, dtype=np.float32) / 255
  test_data = np.array(test_data, dtype=np.float32) / 255

  train_data = train_data.reshape(train_data.shape[0], 28, 28, 1)
  test_data = test_data.reshape(test_data.shape[0], 28, 28, 1)

  train_labels = np.array(train_labels, dtype=np.int32)
  test_labels = np.array(test_labels, dtype=np.int32)

  train_labels = tf.keras.utils.to_categorical(train_labels, num_classes=10)
  test_labels = tf.keras.utils.to_categorical(test_labels, num_classes=10)

  return train_data, train_labels, test_data, test_labels


def main(unused_argv):
  logging.set_verbosity(logging.INFO)
  if FLAGS.dpsgd and FLAGS.batch_size % FLAGS.microbatches != 0:
    raise ValueError('Number of microbatches should divide evenly batch_size')

  # Load training and test data.
  train_data, train_labels, test_data, test_labels = load_mnist()

  # Define a sequential Keras model
  model = tf.keras.Sequential([
      tf.keras.layers.Conv2D(16, 8, strides=2, padding='same', activation='relu', input_shape=(28, 28, 1)),
      tf.keras.layers.MaxPool2D(2, 1),
      tf.keras.layers.Conv2D(32, 4, strides=2,  padding='valid', activation='relu'),
      tf.keras.layers.MaxPool2D(2, 1),
      tf.keras.layers.Flatten(),
      tf.keras.layers.Dense(32, activation='relu'),
      tf.keras.layers.Dense(10)
  ])

  if FLAGS.dpsgd:
    optimizer = DPGradientDescentGaussianOptimizer(
        l2_norm_clip=FLAGS.l2_norm_clip,
        noise_multiplier=FLAGS.noise_multiplier,
        num_microbatches=FLAGS.microbatches,
        learning_rate=FLAGS.learning_rate)
    # Compute vector of per-example loss rather than its mean over a minibatch.
    loss = tf.keras.losses.CategoricalCrossentropy(from_logits=True, reduction=tf.compat.v1.losses.Reduction.NONE)
  else:
    optimizer = GradientDescentOptimizer(learning_rate=FLAGS.learning_rate)
    loss = tf.keras.losses.CategoricalCrossentropy(from_logits=True)

  # Compile model with Keras
  model.compile(optimizer=optimizer, loss=loss, metrics=['accuracy'])

  # Train model with Keras
  eps_callback = EpsilonPrintingCallback()
  fit_history = model.fit(train_data, train_labels, epochs=FLAGS.epochs, validation_data=(test_data, test_labels), batch_size=FLAGS.batch_size, callbacks=[eps_callback])
  eps_history = eps_callback.eps_history
  val_acc_history = fit_history.history['val_accuracy']
  with open('delta_{}_lr_{}.txt'.format(delta, FLAGS.learning_rate), 'w') as f:
        f.write('eps: {}\n'.format(eps_history))
        f.write('validation acc: {}\n'.format(val_acc_history))

if __name__ == '__main__':
  app.run(main)