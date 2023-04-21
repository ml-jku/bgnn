#Source codes of the publication "Boundary Graph Neural Networks for 3D Simulations"



#The code is based on code published in the repository https://github.com/deepmind/graph_nets.
#The following copyright and license terms hold for the code taken from this repository:

# Copyright 2018 The GraphNets Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#    http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or  implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ============================================================================

#The license file and the contribution information file of their original repository 
#are available in our derived repository at:
#- https://github.com/ml-jku/bgnn/blob/master/licenses/graph_nets/LICENSE and 
#- https://github.com/ml-jku/bgnn/blob/master/licenses/graph_nets/CONTRIBUTING.md



#We have applied several modifications to the code, on which the code below is based.
#If compatible to Apache License, Version 2.0, the GNU General Public License v3.0 
#(see https://github.com/ml-jku/bgnn/blob/master/licenses/own/LICENSE_GPL3) holds for 
#the resulting code in this file.
#In case of incompatibilities, Apache License, Version 2.0 should hold or given priority.
#
#IN ANY CASE, THE FOLLOWING HOLDS FOR THIS SOFTWARE:
#
#THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
#IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
#FITNESS FOR A PARTICULAR PURPOSE, TITLE AND NON-INFRINGEMENT. IN NO EVENT
#SHALL THE COPYRIGHT HOLDERS OR ANYONE DISTRIBUTING THE SOFTWARE BE LIABLE
#FOR ANY DAMAGES OR OTHER LIABILITY, WHETHER IN CONTRACT, TORT OR OTHERWISE,
#ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER
#DEALINGS IN THE SOFTWARE.
#
#License Information:
#Copyright (C) 2023  Andreas Mayr
#
#This program is free software: you can redistribute it and/or modify
#it under the terms of the GNU General Public License as published by
#the Free Software Foundation, version 3 of the License.
#
#This program is distributed in the hope that it will be useful,
#but WITHOUT ANY WARRANTY; without even the implied warranty of
#MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
#GNU General Public License for more details.
#
#You should have received a copy of the GNU General Public License
#along with this program.  If not, see <https://www.gnu.org/licenses/>.
#
#The license file for GNU General Public License v3.0 is available here:
#https://github.com/ml-jku/bgnn/blob/master/licenses/own/LICENSE_GPL3


#----------------------------------------------------------------------------




def broadcast_sender_nodes_to_edges(graph, startF, name="broadcast_sender_nodes_to_edges"):
  blocks._validate_broadcasted_graph(graph, blocks.NODES, blocks.SENDERS)
  with tf.name_scope(name):
    return tf.gather(graph.nodes[:, startF:], graph.senders)


def broadcast_receiver_nodes_to_edges(graph, startF, name="broadcast_receiver_nodes_to_edges"):
  blocks._validate_broadcasted_graph(graph, blocks.NODES, blocks.RECEIVERS)
  with tf.name_scope(name):
    return tf.gather(graph.nodes[:, startF:], graph.receivers)


class EdgesToGlobalsAggregator(_base.AbstractModule):
  def __init__(self, reducer, name="edges_to_globals_aggregator"):
    super(EdgesToGlobalsAggregator, self).__init__(name=name)
    self._reducer = reducer

  def _build(self, graph):
    blocks._validate_graph(graph, (blocks.EDGES,), additional_message="when aggregating from edges.")
    num_graphs = utils_tf.get_num_graphs(graph)
    graph_index = tf.range(num_graphs)
    indices = utils_tf.repeat(graph_index, graph.n_edge, axis=0)
    return self._reducer(graph.edges, indices, num_graphs)


class NodesToGlobalsAggregator(_base.AbstractModule):
  def __init__(self, reducer, name="nodes_to_globals_aggregator"):
    super(NodesToGlobalsAggregator, self).__init__(name=name)
    self._reducer = reducer

  def _build(self, graph):
    blocks._validate_graph(graph, (blocks.NODES,), additional_message="when aggregating from nodes.")
    num_graphs = utils_tf.get_num_graphs(graph)
    graph_index = tf.range(num_graphs)
    indices = utils_tf.repeat(graph_index, graph.n_node, axis=0)
    return self._reducer(graph.nodes, indices, num_graphs)


class _EdgesToNodesAggregator(_base.AbstractModule):
  def __init__(self,
               edgeFeatures,
               nodeFeatures,
               reducer,
               use_sent_edges=False,
               name="edges_to_nodes_aggregator"):
    super(_EdgesToNodesAggregator, self).__init__(name=name)
    self.edgeFeatures=edgeFeatures
    self.nodeFeatures=nodeFeatures
    self._reducer = reducer
    self._use_sent_edges = use_sent_edges

  def _build(self, graph):
    blocks._validate_graph(graph, (blocks.EDGES, blocks.SENDERS, blocks.RECEIVERS,), additional_message="when aggregating from edges.")
    if graph.nodes is not None and graph.nodes.shape.as_list()[0] is not None:
      num_nodes = graph.nodes.shape.as_list()[0]
    else:
      num_nodes = tf.reduce_sum(graph.n_node)
    indices = graph.senders if self._use_sent_edges else graph.receivers
    return self._reducer(graph.edges[:,self.edgeFeatures:], indices, num_nodes)

class SentEdgesToNodesAggregator(_EdgesToNodesAggregator):
  def __init__(self,
               edgeFeatures,
               nodeFeatures,
               reducer,
               name="sent_edges_to_nodes_aggregator"):
    super(SentEdgesToNodesAggregator, self).__init__(edgeFeatures=edgeFeatures, nodeFeatures=nodeFeatures, use_sent_edges=True, reducer=reducer, name=name)

class ReceivedEdgesToNodesAggregator(_EdgesToNodesAggregator):
  def __init__(self,
               edgeFeatures,
               nodeFeatures,
               reducer,
               name="received_edges_to_nodes_aggregator"):
    super(ReceivedEdgesToNodesAggregator, self).__init__(edgeFeatures=edgeFeatures, nodeFeatures=nodeFeatures, use_sent_edges=False, reducer=reducer, name=name)





class MyEdgeBlock(_base.AbstractModule):
  def __init__(self,
               staticEdgeFeatures,
               staticNodeFeatures,
               useStaticEdgeFeatures,
               useStaticNodeFeatures,
               residual,
               edge_model_fn,
               use_edges=True,
               use_receiver_nodes=True,
               use_sender_nodes=True,
               use_globals=True,
               name="edge_block"):
    super(MyEdgeBlock, self).__init__(name=name)

    if not (use_edges or use_sender_nodes or use_receiver_nodes or use_globals):
      raise ValueError("At least one of use_edges, use_sender_nodes, use_receiver_nodes or use_globals must be True.")

    self.staticEdgeFeatures=staticEdgeFeatures
    self.staticNodeFeatures=staticNodeFeatures
    self.useStaticEdgeFeatures=useStaticEdgeFeatures
    self.useStaticNodeFeatures=useStaticNodeFeatures
    self.residual=residual
    self._use_edges = use_edges
    self._use_receiver_nodes = use_receiver_nodes
    self._use_sender_nodes = use_sender_nodes
    self._use_globals = use_globals

    with self._enter_variable_scope():
      self._edge_model = edge_model_fn()

  def _build(self, graph):
    blocks._validate_graph(graph, (blocks.SENDERS, blocks.RECEIVERS, blocks.N_EDGE), " when using an EdgeBlock")

    edges_to_collect = []

    if self._use_edges:
      blocks._validate_graph(graph, (blocks.EDGES,), "when use_edges == True")
      if self.useStaticEdgeFeatures:
        edges_to_collect.append(graph.edges)
      else:
        edges_to_collect.append(graph.edges[:,self.staticEdgeFeatures:])

    if self._use_receiver_nodes:
      if self.useStaticNodeFeatures:
        edges_to_collect.append(broadcast_receiver_nodes_to_edges(graph, 0))
      else:
        edges_to_collect.append(broadcast_receiver_nodes_to_edges(graph, self.staticNodeFeatures))

    if self._use_sender_nodes:
      if self.useStaticNodeFeatures:
        edges_to_collect.append(broadcast_sender_nodes_to_edges(graph, 0))
      else:
        edges_to_collect.append(broadcast_sender_nodes_to_edges(graph, self.staticNodeFeatures))

    if self._use_globals:
      edges_to_collect.append(blocks.broadcast_globals_to_edges(graph))

    collected_edges = tf.concat(edges_to_collect, axis=-1)
    updated_edges = self._edge_model(collected_edges)
    #return graph.replace(edges=updated_edges)
    
    self.debugEdgesInput=collected_edges
    self.debugPreOutput=updated_edges
    self.debugResidual=graph.edges
    if self.residual:
      self.debugEdgesOutput=tf.concat([graph.edges[:,0:self.staticEdgeFeatures], graph.edges[:,self.staticEdgeFeatures:]+updated_edges],axis=-1)
    else:
      self.debugEdgesOutput=tf.concat([graph.edges[:,0:self.staticEdgeFeatures], updated_edges],axis=-1)
    
    if self.residual:
      return graph.replace(edges=tf.concat([graph.edges[:,0:self.staticEdgeFeatures], graph.edges[:,self.staticEdgeFeatures:]+updated_edges],axis=-1))
    else:
      return graph.replace(edges=tf.concat([graph.edges[:,0:self.staticEdgeFeatures], updated_edges],axis=-1))



class MyNodeBlock(_base.AbstractModule):
  def __init__(self,
               staticEdgeFeatures,
               staticNodeFeatures,
               useStaticEdgeFeatures,
               useStaticNodeFeatures,
               residual,
               node_model_fn,
               use_received_edges=True,
               use_sent_edges=False,
               use_nodes=True,
               use_globals=True,
               received_edges_reducer=tf.math.unsorted_segment_sum,
               sent_edges_reducer=tf.math.unsorted_segment_sum,
               name="node_block"):
    super(MyNodeBlock, self).__init__(name=name)

    if not (use_nodes or use_sent_edges or use_received_edges or use_globals):
      raise ValueError("At least one of use_received_edges, use_sent_edges, use_nodes or use_globals must be True.")

    self.staticEdgeFeatures=staticEdgeFeatures
    self.staticNodeFeatures=staticNodeFeatures
    self.useStaticEdgeFeatures=useStaticEdgeFeatures
    self.useStaticNodeFeatures=useStaticNodeFeatures
    self.residual=residual
    self._use_received_edges = use_received_edges
    self._use_sent_edges = use_sent_edges
    self._use_nodes = use_nodes
    self._use_globals = use_globals

    with self._enter_variable_scope():
      self._node_model = node_model_fn()
      if self._use_received_edges:
        if received_edges_reducer is None:
          raise ValueError("If `use_received_edges==True`, `received_edges_reducer` should not be None.")
        #self._received_edges_aggregator = ReceivedEdgesToNodesAggregator(self.staticEdgeFeatures, self.staticNodeFeatures, received_edges_reducer)
        if self.useStaticEdgeFeatures:
          self._received_edges_aggregator = ReceivedEdgesToNodesAggregator(0, self.staticNodeFeatures, received_edges_reducer)
        else:
          self._received_edges_aggregator = ReceivedEdgesToNodesAggregator(self.staticEdgeFeatures, self.staticNodeFeatures, received_edges_reducer)
      if self._use_sent_edges:
        if sent_edges_reducer is None:
          raise ValueError("If `use_sent_edges==True`, `sent_edges_reducer` should not be None.")
        if self.useStaticEdgeFeatures:
          self._sent_edges_aggregator = SentEdgesToNodesAggregator(0, self.staticNodeFeatures, sent_edges_reducer)
        else:
          self._sent_edges_aggregator = SentEdgesToNodesAggregator(self.staticEdgeFeatures, self.staticNodeFeatures, sent_edges_reducer)

  def _build(self, graph):
    nodes_to_collect = []

    if self._use_received_edges:
      nodes_to_collect.append(self._received_edges_aggregator(graph))

    if self._use_sent_edges:
      nodes_to_collect.append(self._sent_edges_aggregator(graph))

    if self._use_nodes:
      blocks._validate_graph(graph, (blocks.NODES,), "when use_nodes == True")
      if self.useStaticNodeFeatures:
        nodes_to_collect.append(graph.nodes)
      else:
        nodes_to_collect.append(graph.nodes[:,self.staticNodeFeatures:])

    if self._use_globals:
      nodes_to_collect.append(broadcast_globals_to_nodes(graph))

    collected_nodes = tf.concat(nodes_to_collect, axis=-1)
    updated_nodes = self._node_model(collected_nodes)
    #return graph.replace(nodes=updated_nodes)
    
    self.debugNodesInput=collected_nodes
    self.debugNodesOutput=updated_nodes
    
    if self.residual:
      return graph.replace(nodes=tf.concat([graph.nodes[:,0:self.staticNodeFeatures], graph.nodes[:,self.staticNodeFeatures:]+updated_nodes],axis=-1))
    else:
      return graph.replace(nodes=tf.concat([graph.nodes[:,0:self.staticNodeFeatures], updated_nodes],axis=-1))
