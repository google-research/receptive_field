# coding=utf-8
# Copyright 2019 Google LLC.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Tests for receptive_fields module."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from absl.testing import parameterized
import numpy as np
import receptive_field as rf
import tensorflow.compat.v1 as tf
import tf_slim as slim


def create_test_network_1():
  """Aligned network for test.

  Returns:
    g: Tensorflow graph object (Graph proto).
  """
  g = tf.Graph()
  with g.as_default():
    # An input test image with unknown spatial resolution.
    x = tf.placeholder(tf.float32, (None, None, None, 1), name='input_image')
    # Left branch.
    l1 = slim.conv2d(x, 1, [1, 1], stride=4, scope='L1', padding='VALID')
    # Right branch.
    l2_pad = tf.pad(x, [[0, 0], [1, 0], [1, 0], [0, 0]])
    l2 = slim.conv2d(l2_pad, 1, [3, 3], stride=2, scope='L2', padding='VALID')
    l3 = slim.conv2d(l2, 1, [1, 1], stride=2, scope='L3', padding='VALID')
    # Addition.
    tf.nn.relu(l1 + l3, name='output')
  return g


def create_test_network_2():
  """Aligned network for test.

  The graph corresponds to a variation to the network from
  "create_test_network_1". Layers 2 and 3 are changed to max-pooling operations.
  Since the functionality is the same as convolution, the network is aligned and
  the receptive field size is the same as from the network created using
  create_test_network_1().

  Returns:
    g: Tensorflow graph object (Graph proto).
  """
  g = tf.Graph()
  with g.as_default():
    # An input test image with unknown spatial resolution.
    x = tf.placeholder(tf.float32, (None, None, None, 1), name='input_image')
    # Left branch.
    l1 = slim.conv2d(x, 1, [1, 1], stride=4, scope='L1', padding='VALID')
    # Right branch.
    l2_pad = tf.pad(x, [[0, 0], [1, 0], [1, 0], [0, 0]])
    l2 = slim.max_pool2d(l2_pad, [3, 3], stride=2, scope='L2', padding='VALID')
    l3 = slim.max_pool2d(l2, [1, 1], stride=2, scope='L3', padding='VALID')
    # Addition.
    tf.nn.relu(l1 + l3, name='output')
  return g


def create_test_network_3():
  """Misaligned network for test.

  Returns:
    g: Tensorflow graph object (Graph proto).
  """
  g = tf.Graph()
  with g.as_default():
    # An input test image with unknown spatial resolution.
    x = tf.placeholder(tf.float32, (None, None, None, 1), name='input_image')
    # Left branch.
    l1_pad = tf.pad(x, [[0, 0], [2, 1], [2, 1], [0, 0]])
    l1 = slim.conv2d(l1_pad, 1, [5, 5], stride=2, scope='L1', padding='VALID')
    # Right branch.
    l2 = slim.conv2d(x, 1, [3, 3], stride=1, scope='L2', padding='VALID')
    l3 = slim.conv2d(l2, 1, [3, 3], stride=1, scope='L3', padding='VALID')
    # Addition.
    tf.nn.relu(l1 + l3, name='output')
  return g


def create_test_network_4():
  """Misaligned network for test.

  The graph corresponds to a variation to the network from
  "create_test_network_1". Layer 2 uses 'SAME' padding, which makes its padding
  dependent on the input image dimensionality. In this case, the effective
  padding will be undetermined, and the utility is not able to check the network
  alignment.

  Returns:
    g: Tensorflow graph object (Graph proto).
  """
  g = tf.Graph()
  with g.as_default():
    # An input test image with unknown spatial resolution.
    x = tf.placeholder(tf.float32, (None, None, None, 1), name='input_image')
    # Left branch.
    l1 = slim.conv2d(x, 1, [1, 1], stride=4, scope='L1', padding='VALID')
    # Right branch.
    l2 = slim.conv2d(x, 1, [3, 3], stride=2, scope='L2', padding='SAME')
    l3 = slim.conv2d(l2, 1, [1, 1], stride=2, scope='L3', padding='VALID')
    # Addition.
    tf.nn.relu(l1 + l3, name='output')
  return g


def create_test_network_5():
  """Single-path network for testing non-square kernels.

  The graph is similar to the right branch of the graph from
  create_test_network_1(), except that the kernel sizes are changed to be
  non-square.

  Returns:
    g: Tensorflow graph object (Graph proto).
  """
  g = tf.Graph()
  with g.as_default():
    # An input test image with unknown spatial resolution.
    x = tf.placeholder(tf.float32, (None, None, None, 1), name='input_image')
    # Two convolutional layers, where the first one has non-square kernel.
    l1 = slim.conv2d(x, 1, [3, 5], stride=2, scope='L1', padding='VALID')
    l2 = slim.conv2d(l1, 1, [3, 1], stride=2, scope='L2', padding='VALID')
    # ReLU.
    tf.nn.relu(l2, name='output')
  return g


def create_test_network_6():
  """Aligned network with dropout for test.

  The graph is similar to create_test_network_1(), except that the right branch
  has dropout normalization.

  Returns:
    g: Tensorflow graph object (Graph proto).
  """
  g = tf.Graph()
  with g.as_default():
    # An input test image with unknown spatial resolution.
    x = tf.placeholder(tf.float32, (None, None, None, 1), name='input_image')
    # Left branch.
    l1 = slim.conv2d(x, 1, [1, 1], stride=4, scope='L1', padding='VALID')
    # Right branch.
    l2_pad = tf.pad(x, [[0, 0], [1, 0], [1, 0], [0, 0]])
    l2 = slim.conv2d(l2_pad, 1, [3, 3], stride=2, scope='L2', padding='VALID')
    l3 = slim.conv2d(l2, 1, [1, 1], stride=2, scope='L3', padding='VALID')
    dropout = slim.dropout(l3)
    # Addition.
    tf.nn.relu(l1 + dropout, name='output')
  return g


def create_test_network_7():
  """Aligned network for test, with a control dependency.

  The graph is similar to create_test_network_1(), except that it includes an
  assert operation on the left branch.

  Returns:
    g: Tensorflow graph object (Graph proto).
  """
  g = tf.Graph()
  with g.as_default():
    # An 8x8 test image.
    x = tf.placeholder(tf.float32, (1, 8, 8, 1), name='input_image')
    # Left branch.
    l1 = slim.conv2d(x, 1, [1, 1], stride=4, scope='L1', padding='VALID')
    l1_shape = tf.shape(l1)
    assert_op = tf.Assert(tf.equal(l1_shape[1], 2), [l1_shape], summarize=4)
    # Right branch.
    l2_pad = tf.pad(x, [[0, 0], [1, 0], [1, 0], [0, 0]])
    l2 = slim.conv2d(l2_pad, 1, [3, 3], stride=2, scope='L2', padding='VALID')
    l3 = slim.conv2d(l2, 1, [1, 1], stride=2, scope='L3', padding='VALID')
    # Addition.
    with tf.control_dependencies([assert_op]):
      tf.nn.relu(l1 + l3, name='output')
  return g


def create_test_network_8():
  """Aligned network for test, including an intermediate addition.

  The graph is similar to create_test_network_1(), except that it includes a few
  more layers on top. The added layers compose two different branches whose
  receptive fields are different. This makes this test case more challenging; in
  particular, this test fails if a naive DFS-like algorithm is used for RF
  computation.

  Returns:
    g: Tensorflow graph object (Graph proto).
  """
  g = tf.Graph()
  with g.as_default():
    # An input test image with unknown spatial resolution.
    x = tf.placeholder(tf.float32, (None, None, None, 1), name='input_image')
    # Left branch before first addition.
    l1 = slim.conv2d(x, 1, [1, 1], stride=4, scope='L1', padding='VALID')
    # Right branch before first addition.
    l2_pad = tf.pad(x, [[0, 0], [1, 0], [1, 0], [0, 0]])
    l2 = slim.conv2d(l2_pad, 1, [3, 3], stride=2, scope='L2', padding='VALID')
    l3 = slim.conv2d(l2, 1, [1, 1], stride=2, scope='L3', padding='VALID')
    # First addition.
    l4 = tf.nn.relu(l1 + l3)
    # Left branch after first addition.
    l5 = slim.conv2d(l4, 1, [1, 1], stride=2, scope='L5', padding='VALID')
    # Right branch after first addition.
    l6_pad = tf.pad(l4, [[0, 0], [1, 0], [1, 0], [0, 0]])
    l6 = slim.conv2d(l6_pad, 1, [3, 3], stride=2, scope='L6', padding='VALID')
    # Final addition.
    tf.nn.relu(l5 + l6, name='output')

  return g


def create_test_network_8_keras():
  """Aligned network for test, including an intermediate addition, using Keras.

  It is exactly the same network as for the "create_test_network_8" function.

  Returns:
    g: Tensorflow graph object (Graph proto).
  """
  g = tf.Graph()
  with g.as_default():
    x = tf.keras.Input([None, None, 1], name='input_image')
    l1 = tf.keras.layers.Conv2D(
        filters=1, kernel_size=1, strides=4, padding='valid', name='L1')(
            x)
    l2_pad = tf.keras.layers.ZeroPadding2D(
        padding=[[1, 0], [1, 0]], name='L2_pad')(
            x)
    l2 = tf.keras.layers.Conv2D(
        filters=1, kernel_size=3, strides=2, padding='valid', name='L2')(
            l2_pad)
    l3 = tf.keras.layers.Conv2D(
        filters=1, kernel_size=1, strides=2, name='L3', padding='same')(
            l2)
    l4 = tf.keras.layers.ReLU(name='L4_relu')(l1 + l3)
    l5 = tf.keras.layers.Conv2D(
        filters=1, kernel_size=1, strides=2, padding='valid', name='L5')(
            l4)
    l6_pad = tf.keras.layers.ZeroPadding2D(padding=[[1, 0], [1, 0]])(l4)
    l6 = tf.keras.layers.Conv2D(
        filters=1, kernel_size=3, strides=2, padding='valid', name='L6')(
            l6_pad)
    l7 = tf.keras.layers.ReLU(name='output')(l5 + l6)
    tf.keras.models.Model(x, l7)

  return g


def create_test_network_9():
  """Aligned network for test, including an intermediate addition.

  The graph is the same as create_test_network_8(), except that VALID padding is
  changed to SAME.

  Returns:
    g: Tensorflow graph object (Graph proto).
  """
  g = tf.Graph()
  with g.as_default():
    # An input test image with unknown spatial resolution.
    x = tf.placeholder(tf.float32, (None, None, None, 1), name='input_image')
    # Left branch before first addition.
    l1 = slim.conv2d(x, 1, [1, 1], stride=4, scope='L1', padding='SAME')
    # Right branch before first addition.
    l2 = slim.conv2d(x, 1, [3, 3], stride=2, scope='L2', padding='SAME')
    l3 = slim.conv2d(l2, 1, [1, 1], stride=2, scope='L3', padding='SAME')
    # First addition.
    l4 = tf.nn.relu(l1 + l3)
    # Left branch after first addition.
    l5 = slim.conv2d(l4, 1, [1, 1], stride=2, scope='L5', padding='SAME')
    # Right branch after first addition.
    l6 = slim.conv2d(l4, 1, [3, 3], stride=2, scope='L6', padding='SAME')
    # Final addition.
    tf.nn.relu(l5 + l6, name='output')

  return g


class ReceptiveFieldTest(tf.test.TestCase, parameterized.TestCase):

  def testComputeRFFromGraphDefAligned(self):
    graph_def = create_test_network_1().as_graph_def()
    input_node = 'input_image'
    output_node = 'output'
    (receptive_field_x, receptive_field_y, effective_stride_x,
     effective_stride_y, effective_padding_x, effective_padding_y) = (
         rf.compute_receptive_field_from_graph_def(graph_def, input_node,
                                                   output_node))
    self.assertEqual(receptive_field_x, 3)
    self.assertEqual(receptive_field_y, 3)
    self.assertEqual(effective_stride_x, 4)
    self.assertEqual(effective_stride_y, 4)
    self.assertEqual(effective_padding_x, 1)
    self.assertEqual(effective_padding_y, 1)

  def testComputeRFFromGraphDefAligned2(self):
    graph_def = create_test_network_2().as_graph_def()
    input_node = 'input_image'
    output_node = 'output'
    (receptive_field_x, receptive_field_y, effective_stride_x,
     effective_stride_y, effective_padding_x, effective_padding_y) = (
         rf.compute_receptive_field_from_graph_def(graph_def, input_node,
                                                   output_node))
    self.assertEqual(receptive_field_x, 3)
    self.assertEqual(receptive_field_y, 3)
    self.assertEqual(effective_stride_x, 4)
    self.assertEqual(effective_stride_y, 4)
    self.assertEqual(effective_padding_x, 1)
    self.assertEqual(effective_padding_y, 1)

  def testComputeRFFromGraphDefUnaligned(self):
    graph_def = create_test_network_3().as_graph_def()
    input_node = 'input_image'
    output_node = 'output'
    with self.assertRaises(ValueError):
      rf.compute_receptive_field_from_graph_def(graph_def, input_node,
                                                output_node)

  def testComputeRFFromGraphDefUndefinedPadding(self):
    graph_def = create_test_network_4().as_graph_def()
    input_node = 'input_image'
    output_node = 'output'
    (receptive_field_x, receptive_field_y, effective_stride_x,
     effective_stride_y, effective_padding_x, effective_padding_y) = (
         rf.compute_receptive_field_from_graph_def(graph_def, input_node,
                                                   output_node))
    self.assertEqual(receptive_field_x, 3)
    self.assertEqual(receptive_field_y, 3)
    self.assertEqual(effective_stride_x, 4)
    self.assertEqual(effective_stride_y, 4)
    self.assertEqual(effective_padding_x, None)
    self.assertEqual(effective_padding_y, None)

  def testComputeRFFromGraphDefFixedInputDim(self):
    graph_def = create_test_network_4().as_graph_def()
    input_node = 'input_image'
    output_node = 'output'
    (receptive_field_x, receptive_field_y, effective_stride_x,
     effective_stride_y, effective_padding_x, effective_padding_y) = (
         rf.compute_receptive_field_from_graph_def(
             graph_def, input_node, output_node, input_resolution=[9, 9]))
    self.assertEqual(receptive_field_x, 3)
    self.assertEqual(receptive_field_y, 3)
    self.assertEqual(effective_stride_x, 4)
    self.assertEqual(effective_stride_y, 4)
    self.assertEqual(effective_padding_x, 1)
    self.assertEqual(effective_padding_y, 1)

  def testComputeRFFromGraphDefUnalignedFixedInputDim(self):
    graph_def = create_test_network_4().as_graph_def()
    input_node = 'input_image'
    output_node = 'output'
    with self.assertRaises(ValueError):
      rf.compute_receptive_field_from_graph_def(
          graph_def, input_node, output_node, input_resolution=[8, 8])

  def testComputeRFFromGraphDefNonSquareRF(self):
    graph_def = create_test_network_5().as_graph_def()
    input_node = 'input_image'
    output_node = 'output'
    (receptive_field_x, receptive_field_y, effective_stride_x,
     effective_stride_y, effective_padding_x, effective_padding_y) = (
         rf.compute_receptive_field_from_graph_def(graph_def, input_node,
                                                   output_node))
    self.assertEqual(receptive_field_x, 5)
    self.assertEqual(receptive_field_y, 7)
    self.assertEqual(effective_stride_x, 4)
    self.assertEqual(effective_stride_y, 4)
    self.assertEqual(effective_padding_x, 0)
    self.assertEqual(effective_padding_y, 0)

  def testComputeRFFromGraphDefStopPropagation(self):
    graph_def = create_test_network_6().as_graph_def()
    input_node = 'input_image'
    output_node = 'output'
    # Compute the receptive field but stop the propagation for the random
    # uniform variable of the dropout.
    (receptive_field_x, receptive_field_y, effective_stride_x,
     effective_stride_y, effective_padding_x, effective_padding_y) = (
         rf.compute_receptive_field_from_graph_def(
             graph_def, input_node, output_node,
             ['Dropout/dropout_1/random_uniform']))
    self.assertEqual(receptive_field_x, 3)
    self.assertEqual(receptive_field_y, 3)
    self.assertEqual(effective_stride_x, 4)
    self.assertEqual(effective_stride_y, 4)
    self.assertEqual(effective_padding_x, 1)
    self.assertEqual(effective_padding_y, 1)

  def testComputeCoordinatesRoundtrip(self):
    graph_def = create_test_network_1()
    input_node = 'input_image'
    output_node = 'output'
    receptive_field = rf.compute_receptive_field_from_graph_def(
        graph_def, input_node, output_node)

    x = np.random.randint(0, 100, (50, 2))
    y = receptive_field.compute_feature_coordinates(x)
    x2 = receptive_field.compute_input_center_coordinates(y)

    self.assertAllEqual(x, x2)

  def testComputeRFFromGraphDefAlignedWithControlDependencies(self):
    graph_def = create_test_network_7().as_graph_def()
    input_node = 'input_image'
    output_node = 'output'
    (receptive_field_x, receptive_field_y, effective_stride_x,
     effective_stride_y, effective_padding_x, effective_padding_y) = (
         rf.compute_receptive_field_from_graph_def(graph_def, input_node,
                                                   output_node))
    self.assertEqual(receptive_field_x, 3)
    self.assertEqual(receptive_field_y, 3)
    self.assertEqual(effective_stride_x, 4)
    self.assertEqual(effective_stride_y, 4)
    self.assertEqual(effective_padding_x, 1)
    self.assertEqual(effective_padding_y, 1)

  @parameterized.named_parameters(('', False), ('UsingKeras', True))
  def testComputeRFFromGraphDefWithIntermediateAddNode(self, use_keras_network):
    input_node = 'input_image'
    output_node = 'output'
    if use_keras_network:
      graph_def = create_test_network_8_keras().as_graph_def()
      output_node += '/Relu'
    else:
      graph_def = create_test_network_8().as_graph_def()

    (receptive_field_x, receptive_field_y, effective_stride_x,
     effective_stride_y, effective_padding_x, effective_padding_y) = (
         rf.compute_receptive_field_from_graph_def(graph_def, input_node,
                                                   output_node))
    self.assertEqual(receptive_field_x, 11)
    self.assertEqual(receptive_field_y, 11)
    self.assertEqual(effective_stride_x, 8)
    self.assertEqual(effective_stride_y, 8)
    self.assertEqual(effective_padding_x, 5)
    self.assertEqual(effective_padding_y, 5)

  def testComputeRFFromGraphDefWithIntermediateAddNodeSamePaddingFixedInputDim(
      self):
    graph_def = create_test_network_9().as_graph_def()
    input_node = 'input_image'
    output_node = 'output'
    (receptive_field_x, receptive_field_y, effective_stride_x,
     effective_stride_y, effective_padding_x, effective_padding_y) = (
         rf.compute_receptive_field_from_graph_def(
             graph_def, input_node, output_node, input_resolution=[17, 17]))
    self.assertEqual(receptive_field_x, 11)
    self.assertEqual(receptive_field_y, 11)
    self.assertEqual(effective_stride_x, 8)
    self.assertEqual(effective_stride_y, 8)
    self.assertEqual(effective_padding_x, 5)
    self.assertEqual(effective_padding_y, 5)


if __name__ == '__main__':
  tf.test.main()
