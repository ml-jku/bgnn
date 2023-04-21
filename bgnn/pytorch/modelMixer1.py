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





import torch
import torch.nn
import torch_geometric
import torch_geometric.nn

class MyMLP(torch.nn.Module):
  def __init__(self, inputSize, actFun="relu", layerSz=[128, 128], lastAct=False, useB=True, layerN=False, multI=1.0, nrVirtI=0, normeps=1e-5):
    super(MyMLP, self).__init__()
    if actFun=="relu":
      self.actFun=torch.nn.functional.relu
      self.init=2.0
    elif actFun=="selu":
      self.actFun=torch.nn.functional.selu
      self.init=1.0
    elif actFun=="relu":
      self.actFun=torch.nn.functional.tanh
      self.init=1.0
    self.inputSize=inputSize
    self.layerSz=layerSz
    self.lastAct=lastAct
    self.useB=useB
    self.layerN=layerN
    self.multI=multI
    self.nrVirtI=nrVirtI
    
    self.W=torch.nn.ParameterList()
    self.b=torch.nn.ParameterList()
    
    for i in range(0, len(self.layerSz)):
      outputSize=self.layerSz[i]
      if i==0:
        self.W.append(torch.nn.Parameter(torch.normal(mean=0.0, std=1.0, size=(inputSize, outputSize))*(self.init/(inputSize+self.nrVirtI))))
      else:
        self.W.append(torch.nn.Parameter(torch.normal(mean=0.0, std=1.0, size=(inputSize, outputSize))*(self.init/(inputSize))))
      if self.useB:
        self.b.append(torch.nn.Parameter(torch.zeros(outputSize)))
      inputSize=self.layerSz[i]
    
    self.ln=torch.nn.LayerNorm(outputSize, elementwise_affine=False, eps=normeps)
  
  def forward(self, x):
    x=x*self.multI
    for i in range(0, len(self.layerSz)-1):
      if self.useB:
        x=self.actFun(torch.matmul(x, self.W[i])+self.b[i])
      else:
        x=self.actFun(torch.matmul(x, self.W[i]))
    if self.useB:
      x=torch.matmul(x, self.W[-1])+self.b[-1]
    else:
      x=torch.matmul(x, self.W[-1])
    if self.lastAct:
      x=self.actFun(x)
    if self.layerN:
      x=self.ln(x)
    return x



class MLPGraphNetwork(torch_geometric.nn.MessagePassing):
  def __init__(self,
               nrNodeFeatures,
               nrEdgeFeatures,
               edgep=None,
               nodep=None,
               sef=64,
               snf=64,
               useffeu=False,
               usnffeu=False,
               useffnu=False,
               usnffnu=False,
               rese=True,
               resn=True):
    super(MLPGraphNetwork, self).__init__(aggr="mean", flow="source_to_target")
    
    self.nrNodeFeatures=nrNodeFeatures
    self.nrEdgeFeatures=nrEdgeFeatures
    self.edgep=edgep
    self.nodep=nodep
    self.sef=sef  #staticEdgeFeatures
    self.snf=snf  #staticNodeFeatures
    self.useffeu=useffeu  #useStaticEdgeFeatures - Edge Update
    self.usnffeu=usnffeu  #useStaticNodeFeatures - Edge Update
    self.useffnu=useffnu  #useStaticEdgeFeatures - Node Update
    self.usnffnu=usnffnu  #useStaticNodeFeatures - Node Update
    self.rese=rese
    self.resn=resn
    
    myEdgeParameters={}
    myEdgeParameters.update({} if edgep is None else edgep)
    edgeInputNr=0
    if self.useffeu:
      edgeInputNr=edgeInputNr+nrEdgeFeatures
    else:
      edgeInputNr=edgeInputNr+nrEdgeFeatures-self.sef
    if self.usnffeu:
      edgeInputNr=edgeInputNr+nrNodeFeatures
      edgeInputNr=edgeInputNr+nrNodeFeatures
    else:
      edgeInputNr=edgeInputNr+nrNodeFeatures-self.snf
      edgeInputNr=edgeInputNr+nrNodeFeatures-self.snf
    
    myNodeParameters={}
    myNodeParameters.update({} if nodep is None else nodep)
    nodeInputNr=0
    if self.useffnu:
      nodeInputNr=nodeInputNr+myEdgeParameters["layerSz"][-1]
    else:
      nodeInputNr=nodeInputNr+myEdgeParameters["layerSz"][-1]-self.sef
    if self.usnffnu:
      nodeInputNr=nodeInputNr+nrNodeFeatures
    else:
      nodeInputNr=nodeInputNr+nrNodeFeatures-self.snf
    
    myEdgeParameters["layerSz"][-1]=myEdgeParameters["layerSz"][-1]-sef if rese else myEdgeParameters["layerSz"][-1]
    self.myEdgeMLP=MyMLP(edgeInputNr, **myEdgeParameters)
    
    myNodeParameters["layerSz"][-1]=myNodeParameters["layerSz"][-1]-snf if resn else myNodeParameters["layerSz"][-1]
    self.myNodeMLP=MyMLP(nodeInputNr, **myNodeParameters)
  
  
  
  def message(self, nodeFeatures_i, nodeFeatures_j, edgeFeatures):
    fcollect=[]
    
    if self.useffeu:
      fcollect.append(edgeFeatures)
    else:
      fcollect.append(edgeFeatures[:,self.sef:])
    
    if self.usnffeu:
      fcollect.append(nodeFeatures_i)
      fcollect.append(nodeFeatures_j)
    else:
      fcollect.append(nodeFeatures_i[:,self.snf:])
      fcollect.append(nodeFeatures_j[:,self.snf:])
    
    edgeInput=torch.cat(fcollect, -1)
    edgeOutput=self.myEdgeMLP(edgeInput)
    
    if self.rese:
      messageOutput=torch.cat([edgeFeatures[:,0:self.sef], edgeFeatures[:,self.sef:]+edgeOutput], -1)
    else:
      messageOutput=torch.cat([edgeFeatures[:,0:self.sef], edgeOutput], -1)
    
    self.edgeFeaturesNew=messageOutput
    return messageOutput
  
  
  
  def update(self, collMessages, nodeFeatures):
    fcollect=[]
    
    if self.useffnu:
      fcollect.append(collMessages)
    else:
      fcollect.append(collMessages[:,self.sef:])
    
    if self.usnffnu:
      fcollect.append(nodeFeatures)
    else:
      fcollect.append(nodeFeatures[:,self.snf:])
    
    nodeInput=torch.cat(fcollect, -1)
    nodeOutput=self.myNodeMLP(nodeInput)
    
    if self.resn:
      updateOutput=torch.cat([nodeFeatures[:,0:self.snf], nodeFeatures[:,self.snf:]+nodeOutput], -1)
    else:
      updateOutput=torch.cat([nodeFeatures[:,0:self.snf], nodeOutput], -1)
    
    self.nodeFeaturesNew=updateOutput
    return updateOutput
  
  
  
  def forward(self, edge_index, nodeFeatures, edgeFeatures):
    out = self.propagate(edge_index, nodeFeatures=nodeFeatures, edgeFeatures=edgeFeatures)
    return out




class EncodeProcessDecode(torch.nn.Module):
  def __init__(self,
               nrNodeFeatures,
               nrEdgeFeatures,
               nrSteps=3,
               shared=False,
               embedInit=1.0,
               nrEmbed=16,
               inputPar=None,
               processPar=None,
               outputPar=None,
               rese=False,
               resn=False,
               nrTypes=2):
    super(EncodeProcessDecode, self).__init__()
    
    self.nrSteps=nrSteps
    self.shared=shared
    self.rese=rese
    self.resn=resn
    self.processPar=processPar
    self.matEmb=torch.nn.Embedding(nrTypes, nrEmbed)
    
    
    
    inputPar={} if inputPar is None else inputPar
    inputPar["nodep"]={} if "nodep" not in inputPar else inputPar["nodep"]
    inputPar["edgep"]={} if "edgep" not in inputPar else inputPar["edgep"]
    
    inputEdgeParameters={
      "actFun": "relu",
      "layerSz": [128, 128],
      "lastAct": False,
      "useB": True,
      "layerN": False,
      "nrVirtI": 0
    }
    inputEdgeParameters.update(inputPar["nodep"])
    
    inputNodeParameters={
      "actFun": "relu",
      "layerSz": [128, 128],
      "lastAct": False,
      "useB": True,
      "layerN": False,
      "nrVirtI": 0
    }
    inputNodeParameters.update(inputPar["edgep"])
    inputNodeParameters.update({"nrVirtI": inputPar["nodep"]["nrVirtI"]+(embedInit*embedInit-1.0)*nrEmbed})
    
    self.nodeEncoder=MyMLP(nrNodeFeatures-1+nrEmbed, **inputNodeParameters)
    self.edgeEncoder=MyMLP(nrEdgeFeatures, **inputEdgeParameters)
    
    
    
    processPar={} if processPar is None else processPar
    processPar["nodep"]={} if "nodep" not in processPar else processPar["nodep"]
    processPar["edgep"]={} if "edgep" not in processPar else processPar["edgep"]
    
    processEdgeParameters={
      "actFun": "relu",
      "layerSz": [128, 128],
      "lastAct": False,
      "useB": True,
      "layerN": False,
      "nrVirtI": 0
    }
    processEdgeParameters.update(processPar["edgep"])
    processEdgeParameters["layerSz"][-1]=processEdgeParameters["layerSz"][-1]-sef if rese else processEdgeParameters["layerSz"][-1]
    processPar["edgep"]=processEdgeParameters
    
    processNodeParameters={
      "actFun": "relu",
      "layerSz": [128, 128],
      "lastAct": False,
      "useB": True,
      "layerN": False,
      "nrVirtI": 0
    }
    processNodeParameters.update(processPar["nodep"])
    processNodeParameters["layerSz"][-1]=processNodeParameters["layerSz"][-1]-snf if resn else processNodeParameters["layerSz"][-1]
    processPar["nodep"]=processNodeParameters
    
    self.core=torch.nn.ModuleList([])
    glayer=MLPGraphNetwork(inputNodeParameters["layerSz"][-1], inputEdgeParameters["layerSz"][-1], **processPar)
    for i in range(self.nrSteps):
      self.core.append(glayer)
      if not shared:
        glayer=MLPGraphNetwork(processNodeParameters["layerSz"][-1], processEdgeParameters["layerSz"][-1], **processPar)
    
    
    
    outputPar={} if outputPar is None else outputPar
    outputPar["nodep"]={} if "nodep" not in outputPar else outputPar["nodep"]
    outputPar["edgep"]={} if "edgep" not in outputPar else outputPar["edgep"]
    
    outputNodeParameters={
      "actFun": "relu",
      "layerSz": [128, 3],
      "lastAct": False,
      "useB": True,
      "layerN": False,
      "nrVirtI": 0
    }
    outputNodeParameters.update(outputPar["nodep"])
    
    self.nodeDecoder=MyMLP(processNodeParameters["layerSz"][-1], **outputNodeParameters)
  
  
  
  def forward(self, edge_index, nodeFeatures, edgeFeatures, debug=-2):
    particleType=(nodeFeatures[:,-1]+0.5).to(torch.int32)
    nodeInput=torch.cat([nodeFeatures[:,:-1], self.matEmb(particleType)], 1)
    edgeInput=edgeFeatures
    
    if debug==-1:
      return nodeInput
    nodeInput=self.nodeEncoder(nodeInput)
    edgeInput=self.edgeEncoder(edgeInput)
    
    for i in range(self.nrSteps):
      if debug==i:
        return nodeInput
      
      self.core[i](edge_index, nodeInput, edgeInput)
      
      if self.rese:
        edgeInput=torch.cat([edgeInput[:,0:self.processPar["sef"]], (edgeInput[:,self.processPar["sef"]:]+self.core[i].edgeFeaturesNew[:,self.processPar["sef"]:]*float(i+1))/float(i+2)],axis=-1)
      elif self.processPar["sef"]>0.5:
        edgeInput=torch.cat([edgeInput[:,0:self.processPar["sef"]], self.core[i].edgeFeaturesNew[:,self.processPar["sef"]:]],axis=-1)
      else:
        edgeInput=self.core[i].edgeFeaturesNew[:,self.processPar["sef"]:]
      
      if self.resn:
        nodeInput=torch.cat([nodeInput[:,0:self.processPar["snf"]], (nodeInput[:,self.processPar["snf"]:]+self.core[i].nodeFeaturesNew[:,self.processPar["snf"]:]*float(i+1))/float(i+2)],axis=-1)
      elif self.processPar["snf"]>0.5:
        nodeInput=torch.cat([nodeInput[:,0:self.processPar["snf"]], self.core[i].nodeFeaturesNew[:,self.processPar["snf"]:]],axis=-1)
      else:
        nodeInput=self.core[i].nodeFeaturesNew[:,self.processPar["snf"]:]
    
    if debug==self.nrSteps:
      return nodeInput
    nodeInput=self.nodeDecoder(nodeInput)
    
    if debug==self.nrSteps+1:
      return nodeInput
    
    return nodeInput

