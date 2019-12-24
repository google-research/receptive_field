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

"""Computes Receptive Field (RF) information given a graph protobuf."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse
import sys

from absl import app
from absl import logging
import receptive_field as rf
import tensorflow as tf

from google.protobuf import text_format

cmd_args = None


def _load_graphdef(path):
  """Helper function to load GraphDef from file.

  Args:
    path: Path to pbtxt file.

  Returns:
    graph_def: A GraphDef object.
  """
  graph_def = tf.GraphDef()
  pbstr = tf.gfile.Open(path).read()
  text_format.Parse(pbstr, graph_def)
  return graph_def


def main(unused_argv):

  graph_def = _load_graphdef(cmd_args.graph_path)

  (receptive_field_x, receptive_field_y, effective_stride_x, effective_stride_y,
   effective_padding_x,
   effective_padding_y) = rf.compute_receptive_field_from_graph_def(
       graph_def, cmd_args.input_node, cmd_args.output_node)

  logging.info('Receptive field size (horizontal) = %s', receptive_field_x)
  logging.info('Receptive field size (vertical) = %s', receptive_field_y)
  logging.info('Effective stride (horizontal) = %s', effective_stride_x)
  logging.info('Effective stride (vertical) = %s', effective_stride_y)
  logging.info('Effective padding (horizontal) = %s', effective_padding_x)
  logging.info('Effective padding (vertical) = %s', effective_padding_y)

  f = tf.gfile.GFile('%s' % cmd_args.output_path, 'w')
  f.write('Receptive field size (horizontal) = %s\n' % receptive_field_x)
  f.write('Receptive field size (vertical) = %s\n' % receptive_field_y)
  f.write('Effective stride (horizontal) = %s\n' % effective_stride_x)
  f.write('Effective stride (vertical) = %s\n' % effective_stride_y)
  f.write('Effective padding (horizontal) = %s\n' % effective_padding_x)
  f.write('Effective padding (vertical) = %s\n' % effective_padding_y)
  f.close()


if __name__ == '__main__':
  parser = argparse.ArgumentParser()
  parser.register('type', 'bool', lambda v: v.lower() == 'true')
  parser.add_argument(
      '--graph_path', type=str, default='', help='Graph path (pbtxt format).')
  parser.add_argument(
      '--output_path',
      type=str,
      default='',
      help='Path to output text file where RF information will be written to.')
  parser.add_argument(
      '--input_node', type=str, default='', help='Name of input node.')
  parser.add_argument(
      '--output_node', type=str, default='', help='Name of output node.')
  cmd_args, unparsed = parser.parse_known_args()
  app.run(main=main, argv=[sys.argv[0]] + unparsed)
