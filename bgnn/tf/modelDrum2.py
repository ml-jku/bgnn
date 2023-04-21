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

class MyMLP(snt.Module):
  def __init__(self, actFun="relu", layerSz=[128, 128], lastAct=False, useB=True, layerN=False, multI=1.0, nrVirtI=0, name=None, normeps=1e-5):
    super(MyMLP, self).__init__(name=name)
    if actFun=="relu":
      self.actFun=tf.nn.relu
      self.init=2.0
    elif actFun=="selu":
      self.actFun=tf.nn.selu
      self.init=1.0
    elif actFun=="relu":
      self.actFun=tf.math.tanh
      self.init=1.0
    self.layerSz=layerSz
    self.lastAct=lastAct
    self.useB=useB
    self.layerN=layerN
    self.multI=multI
    self.nrVirtI=nrVirtI
    
    self.W=[]
    self.b=[]
    self.ln=snt.LayerNorm(axis=-1, create_offset=False, create_scale=False, eps=normeps)

  @snt.once
  def _initialize(self, x):
    inputSize=x.shape[1]
    for i in range(0, len(self.layerSz)):
      outputSize=self.layerSz[i]
      if i==0:
        self.W.append(tf.Variable(tf.random.normal([inputSize, outputSize])*(self.init/(inputSize+self.nrVirtI))))
      else:
        self.W.append(tf.Variable(tf.random.normal([inputSize, outputSize])*(self.init/(inputSize))))
      if self.useB:
        self.b.append(tf.Variable(tf.zeros([outputSize])))
      inputSize=self.layerSz[i]

  def __call__(self, x):
    self._initialize(x)
    x=x*self.multI
    for i in range(0, len(self.layerSz)-1):
      if self.useB:
        x=self.actFun(tf.matmul(x, self.W[i])+self.b[i])
      else:
        x=self.actFun(tf.matmul(x, self.W[i]))
    if self.useB:
      x=tf.matmul(x, self.W[-1])+self.b[-1]
    else:
      x=tf.matmul(x, self.W[-1])
    if self.lastAct:
      x=self.actFun(x)
    if self.layerN:
      x=self.ln(x)
    return x

def createNetwork(networkParameters):
  def createNetworkFunction():
    return MyMLP(**networkParameters)
  return createNetworkFunction



class MLPGraphIndependentInput(snt.Module):
  def __init__(self, edgep=None, nodep=None, name="MLPGraphIndependentInput"):
    super(MLPGraphIndependentInput, self).__init__(name=name)
    
    myEdgeParameters={
      "actFun": "relu",
      "layerSz": [128, 128],
      "lastAct": False,
      "useB": True,
      "layerN": False,
      "nrVirtI": 0
    }
    myEdgeParameters.update({} if edgep is None else edgep)
    
    myNodeParameters={
      "actFun": "relu",
      "layerSz": [128, 128],
      "lastAct": False,
      "useB": True,
      "layerN": False,
      "nrVirtI": 0
    }
    myNodeParameters.update({} if nodep is None else nodep)
    
    self._network=modules.GraphIndependent(edge_model_fn=createNetwork(myEdgeParameters), node_model_fn=createNetwork(myNodeParameters), global_model_fn=None)
  
  def __call__(self, inputs):
    return self._network(inputs)

class MLPGraphIndependentOutput(snt.Module):
  def __init__(self, edgep=None, nodep=None, name="MLPGraphIndependentOutput"):
    super(MLPGraphIndependentOutput, self).__init__(name=name)
    
    myEdgeParameters={
      "actFun": "relu",
      "layerSz": [128, 128],
      "lastAct": False,
      "useB": True,
      "layerN": False,
      "nrVirtI": 0
    }
    myEdgeParameters.update({} if edgep is None else edgep)
    
    myNodeParameters={
      "actFun": "relu",
      "layerSz": [128, 3],
      "lastAct": False,
      "useB": True,
      "layerN": False,
      "nrVirtI": 0
    }
    myNodeParameters.update({} if nodep is None else nodep)
    
    #self._network=modules.GraphIndependent(edge_model_fn=createNetwork(myEdgeParameters), node_model_fn=createNetwork(myNodeParameters), global_model_fn=None)
    self._network=modules.GraphIndependent(edge_model_fn=None, node_model_fn=createNetwork(myNodeParameters), global_model_fn=None)
  
  def __call__(self, inputs):
    return self._network(inputs)

class MLPGraphNetwork(_base.AbstractModule):
  def __init__(self,
               edgep=None,
               nodep=None,
               sef=64,
               snf=64,
               useffeu=False,
               usnffeu=False,
               useffnu=False,
               usnffnu=False,
               rese=True,
               resn=True,
               name="graph_network"):
    super(MLPGraphNetwork, self).__init__(name="MLPGraphNetwork")
    
    with self._enter_variable_scope():
      myEdgeParameters={
        "actFun": "relu",
        "layerSz": [128, 128],
        "lastAct": False,
        "useB": True,
        "layerN": False,
        "nrVirtI": 0
      }
      myEdgeParameters.update({} if edgep is None else edgep)
      myEdgeParameters["layerSz"][-1]=myEdgeParameters["layerSz"][-1]-sef if rese else myEdgeParameters["layerSz"][-1]
      edge_block_opt = {"use_edges": True, "use_receiver_nodes": True, "use_sender_nodes": True, "use_globals": False}
      edge_block_opt=modules._make_default_edge_block_opt(edge_block_opt)
      self._edge_block=MyEdgeBlock(staticEdgeFeatures=sef,
                                   staticNodeFeatures=snf,
                                   useStaticEdgeFeatures=useffeu,
                                   useStaticNodeFeatures=usnffeu,
                                   residual=rese,
                                   edge_model_fn=createNetwork(myEdgeParameters),
                                   **edge_block_opt)
      
      myNodeParameters={
        "actFun": "relu",
        "layerSz": [128, 128],
        "lastAct": False,
        "useB": True,
        "layerN": False,
        "nrVirtI": 0
      }
      myNodeParameters.update({} if nodep is None else nodep)
      myNodeParameters["layerSz"][-1]=myNodeParameters["layerSz"][-1]-snf if resn else myNodeParameters["layerSz"][-1]
      node_block_opt = {"use_received_edges": True, "use_sent_edges": False, "use_nodes": True, "use_globals": False}
      node_block_opt=modules._make_default_node_block_opt(node_block_opt, tf.math.unsorted_segment_mean)
      self._node_block=MyNodeBlock(staticEdgeFeatures=sef,
                                   staticNodeFeatures=snf,
                                   useStaticEdgeFeatures=useffnu,
                                   useStaticNodeFeatures=usnffnu,
                                   residual=resn,
                                   node_model_fn=createNetwork(myNodeParameters),
                                   **node_block_opt)
  
  def _build(self, graph):
    return self._node_block(self._edge_block(graph))

class EncodeProcessDecode(snt.Module):
  def __init__(self, 
               nrSteps=3,
               shared=False,
               embedInit=1.0,
               nrEmbed=16,
               inputPar=None,
               processPar=None,
               outputPar=None,
               rese=False,
               resn=False,
               name="EncodeProcessDecode",
               nrTypes=2):
    super(EncodeProcessDecode, self).__init__(name=name)
    
    self.nrSteps=nrSteps
    self.shared=shared
    self.rese=rese
    self.resn=resn
    self.processPar=processPar
    
    
    
    self.matEmb=snt.Embed(nrTypes, nrEmbed, initializer=snt.initializers.RandomNormal(0.0, embedInit))
    
    inputPar={} if inputPar is None else inputPar
    processPar={} if processPar is None else processPar
    outputPar={} if outputPar is None else outputPar
    
    inputPar["nodep"]={} if "nodep" not in inputPar else inputPar["nodep"]
    inputPar["nodep"]["nrVirtI"]=0 if "nrVirtI" not in inputPar["nodep"] else inputPar["nodep"]["nrVirtI"]
    inputPar["nodep"].update({"nrVirtI": inputPar["nodep"]["nrVirtI"]+(embedInit*embedInit-1.0)*nrEmbed})
    self._encoder=MLPGraphIndependentInput(**inputPar)
    
    self._core=[]
    glayer=MLPGraphNetwork(**processPar)
    for i in range(self.nrSteps):
      self._core.append(glayer)
      if not shared:
        glayer=MLPGraphNetwork(**processPar)
    
    self._decoder=MLPGraphIndependentOutput(**outputPar)
  
  def __call__(self, input_op, debug=-2):
    features=input_op.nodes[:,:-1]
    particleType=tf.cast(input_op.nodes[:,-1]+0.5, tf.int32)
    core_input=input_op.replace(nodes=features)
    
    core_input=core_input.replace(nodes=tf.concat([core_input.nodes, self.matEmb(particleType)], 1))
    
    if debug==-1:
      return core_input
    core_input=self._encoder(core_input)
    
    for i in range(self.nrSteps):
      if debug==i:
        return core_input
      
      core_output=self._core[i](core_input)
      
      if self.rese:
        core_output.replace(edges=tf.concat([core_output.edges[:,0:self.processPar["sef"]], (core_output.edges[:,self.processPar["sef"]:]+core_input.edges[:,self.processPar["sef"]:]*float(i+1))/float(i+2)],axis=-1))
      elif self.processPar["sef"]>0.5:
        core_output.replace(edges=tf.concat([core_output.edges[:,0:self.processPar["sef"]], core_output.edges[:,self.processPar["sef"]:]],axis=-1))
      
      if self.resn:
        core_output.replace(nodes=tf.concat([core_output.nodes[:,0:self.processPar["snf"]], (core_output.nodes[:,self.processPar["snf"]:]+core_input.nodes[:,self.processPar["snf"]:]*float(i+1))/float(i+2)],axis=-1))
      elif self.processPar["snf"]>0.5:
        core_output.replace(nodes=tf.concat([core_output.nodes[:,0:self.processPar["snf"]], core_output.nodes[:,self.processPar["snf"]:]],axis=-1))
      
      core_input=core_output
    
    if debug==self.nrSteps:
      return core_input
    decoded_op=self._decoder(core_input)
    
    if debug==self.nrSteps+1:
      return decoded_op
    
    return decoded_op
