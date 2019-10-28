# Copyright 2018, The TensorFlow Authors.
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
"""Differentially private optimizers for TensorFlow."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf

from privacy.analysis import privacy_ledger
from privacy.dp_query import gaussian_query

def make_optimizer_class(cls):
  """Constructs a DP optimizer class from an existing one."""
  parent_code = tf.optimizers.Optimizer._compute_gradients.__code__
  child_code = cls._compute_gradients.__code__
  if child_code is not parent_code:
    tf.logging.warning(
        'WARNING: Calling make_optimizer_class() on class %s that overrides '
        'method compute_gradients(). Check to ensure that '
        'make_optimizer_class() does not interfere with overridden version.',
        cls.__name__)

  class DPOptimizerClass(cls):
    """Differentially private subclass of given class cls."""

    def __init__(
        self,
        dp_sum_query,
        num_microbatches=None,
        unroll_microbatches=False,
        *args,
        **kwargs):
      """Initialize the DPOptimizerClass.

      Args:
        dp_sum_query: DPQuery object, specifying differential privacy
          mechanism to use.
        num_microbatches: How many microbatches into which the minibatch is
          split. If None, will default to the size of the minibatch, and
          per-example gradients will be computed.
        unroll_microbatches: If true, processes microbatches within a Python
          loop instead of a tf.while_loop. Can be used if using a tf.while_loop
          raises an exception.
      """
      super(DPOptimizerClass, self).__init__(*args, **kwargs)
      ###### accountant + sanitizer ######
      self._dp_sum_query = dp_sum_query
      ######
      self._num_microbatches = num_microbatches
      self._global_state = self._dp_sum_query.initial_global_state()
      self._unroll_microbatches = unroll_microbatches

    def compute_gradients(self, loss, var_list, gate_gradients=None, aggregation_method=None, colocate_gradients_with_ops=False, grad_loss=None, gradient_tape=None):
      if not gradient_tape:
        raise ValueError('A tape needs to be passed.')

      vector_loss = loss()
      if self._num_microbatches is None:
        self._num_microbatches = tf.shape(vector_loss)[0]
      sample_state = self._dp_sum_query.initial_sample_state(var_list)
      microbatches_losses = tf.reshape(vector_loss, [self._num_microbatches, -1])
      sample_params = (self._dp_sum_query.derive_sample_params(self._global_state))

      for idx in range(self._num_microbatches):
        ###### compute gradient ######
        microbatch_loss = tf.reduce_mean(tf.gather(microbatches_losses, [idx]))
        grads = gradient_tape.gradient(microbatch_loss, var_list)
        ######

        ###### accountant ######
        sample_state = self._dp_sum_query.accumulate_record(sample_params, sample_state, grads)
        ######

      ###### sanitizer ######
      grad_sums, self._global_state = (self._dp_sum_query.get_noised_result(sample_state, self._global_state)) 
      ######

      def normalize(v):
        return v / tf.cast(self._num_microbatches, tf.float32)

      final_grads = tf.nest.map_structure(normalize, grad_sums)

      grads_and_vars = list(zip(final_grads, var_list))
      return grads_and_vars

  return DPOptimizerClass


def make_gaussian_optimizer_class(cls):
  """Constructs a DP optimizer with Gaussian averaging of updates."""

  class DPGaussianOptimizerClass(make_optimizer_class(cls)):
    """DP subclass of given class cls using Gaussian averaging."""

    def __init__(self, l2_norm_clip, noise_multiplier, num_microbatches=None, ledger=None, unroll_microbatches=False, *args, **kwargs):
      dp_sum_query = gaussian_query.GaussianSumQuery(l2_norm_clip, l2_norm_clip * noise_multiplier)

      if ledger:
        dp_sum_query = privacy_ledger.QueryWithLedger(dp_sum_query, ledger=ledger)

      super(DPGaussianOptimizerClass, self).__init__(dp_sum_query, num_microbatches, unroll_microbatches, *args, **kwargs)

    @property
    def ledger(self):
      return self._dp_sum_query.ledger

  return DPGaussianOptimizerClass

DPAdagradOptimizer = make_optimizer_class(tf.optimizers.Adagrad)
DPAdamOptimizer = make_optimizer_class(tf.optimizers.Adam)
DPGradientDescentOptimizer = make_optimizer_class(tf.optimizers.SGD)

DPAdagradGaussianOptimizer = make_gaussian_optimizer_class(tf.optimizers.Adagrad)
DPAdamGaussianOptimizer = make_gaussian_optimizer_class(tf.optimizers.Adam)
DPGradientDescentGaussianOptimizer = make_gaussian_optimizer_class(tf.optimizers.SGD)
