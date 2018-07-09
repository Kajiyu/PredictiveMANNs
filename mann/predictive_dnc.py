#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import torch.nn as nn
import torch as T
import torch.nn.functional as F
from torch.autograd import Variable as var
import numpy as np

from torch.nn.utils.rnn import pad_packed_sequence as pad
from torch.nn.utils.rnn import pack_padded_sequence as pack
from torch.nn.utils.rnn import PackedSequence

from .util import *
from .memory import *
from .dis import *
from .gen import *

from torch.nn.init import orthogonal, xavier_uniform


class PDNC(nn.Module):

  def __init__(
      self,
      hidden_size,
      whole_input_size,
      whole_output_size,
      rnn_type='lstm',
      num_hidden_layers=2,
      bias=True,
      batch_first=True,
      dropout=0,
      bidirectional=False,
      nr_cells=5,
      read_heads=2,
      cell_size=10,
      nonlinearity='tanh',
      gpu_id=-1,
      independent_linears=False,
      share_memory=True,
      debug=False,
      clip=20,
      skip_hidden_to_output = True,
      skip_memory_reset = False
  ):
    super(PDNC, self).__init__()
    # todo: separate weights and RNNs for the interface and output vectors
    self.hidden_size = hidden_size
    self.whole_input_size = whole_input_size
    self.whole_output_size = whole_output_size
    self.rnn_type = rnn_type
    self.num_hidden_layers = num_hidden_layers
    self.bias = bias
    self.batch_first = batch_first
    self.dropout = dropout
    self.bidirectional = bidirectional
    self.nr_cells = nr_cells
    self.read_heads = read_heads
    self.cell_size = cell_size
    self.nonlinearity = nonlinearity
    self.gpu_id = gpu_id
    self.independent_linears = independent_linears
    self.share_memory = share_memory
    self.debug = debug
    self.clip = clip
    self.skip_hidden_to_output = skip_hidden_to_output
    self.skip_memory_reset = skip_memory_reset

    self.w = self.cell_size
    self.r = self.read_heads

    self.read_vectors_size = self.r * self.w
    self.output_size = self.hidden_size

    self.nn_input_size = self.whole_output_size + 2 + self.read_vectors_size
    self.nn_output_size = self.output_size + self.read_vectors_size

    self.rnns = []
    self.memories = []

    self.mem_hidden = None

    self.c_encoder = ConvEncoder(self.whole_input_size, z_dim=self.whole_output_size)
    self.net_g = Generator(self.whole_input_size, z_dim=self.whole_output_size)
    self.net_d = Discriminator(self.whole_input_size)
    setattr(self, 'conv_encoder', self.c_encoder)
    setattr(self, 'conv_generator', self.net_g)
    setattr(self, 'conv_discriminator', self.net_d)

    if self.rnn_type.lower() == 'rnn':
      self.controller = nn.RNN(
        self.nn_input_size, self.output_size, bias=self.bias,
        nonlinearity=self.nonlinearity, batch_first=True, dropout=self.dropout,
        num_layers=self.num_hidden_layers
      )
    elif self.rnn_type.lower() == 'gru':
      self.controller = nn.GRU(
        self.nn_input_size, self.output_size, bias=self.bias,
        batch_first=True, dropout=self.dropout, num_layers=self.num_hidden_layers
      )
    elif self.rnn_type.lower() == 'lstm':
      self.controller = nn.LSTM(
        self.nn_input_size, self.output_size, bias=self.bias,
        batch_first=True, dropout=self.dropout, num_layers=self.num_hidden_layers
      )
    setattr(self, self.rnn_type.lower() + '_controller', self.controller)
    self.memory = Memory(
      input_size=self.output_size,
      mem_size=self.nr_cells,
      cell_size=self.w,
      read_heads=self.r,
      gpu_id=self.gpu_id,
      independent_linears=self.independent_linears
    )
    setattr(self, 'memory', self.memory)

    # final output layer
    self.output = nn.Linear(self.nn_output_size, self.whole_output_size)
    orthogonal(self.output.weight)

    if self.gpu_id != -1:
      [x.cuda(self.gpu_id) for x in [self.controller, self.memory]]

  def _init_hidden(self, hx, batch_size, reset_experience):
    # create empty hidden states if not provided
    if hx is None:
      hx = (None, None, None)
    (chx, mhx, last_read) = hx

    # initialize hidden state of the controller RNN
    if chx is None:
      h = cuda(T.zeros(self.num_hidden_layers, batch_size, self.output_size), gpu_id=self.gpu_id)
      xavier_uniform(h)
      if self.rnn_type.lower() == 'lstm':
        chx = (h, h)
      else:
        chx = h

    # Last read vectors
    if last_read is None:
      last_read = cuda(T.zeros(batch_size, self.w * self.r), gpu_id=self.gpu_id)

    # memory states
    if mhx is None:
      mhx = self.memory.reset(batch_size, erase=reset_experience)
    else:
      mhx = self.memory.reset(batch_size, mhx, erase=reset_experience)

    return chx, mhx, last_read

  def _debug(self, mhx, debug_obj):
    if not debug_obj:
      debug_obj = {
          'memory': [],
          'link_matrix': [],
          'precedence': [],
          'read_weights': [],
          'write_weights': [],
          'usage_vector': [],
      }

    debug_obj['memory'].append(mhx['memory'][0].data.cpu().numpy())
    debug_obj['link_matrix'].append(mhx['link_matrix'][0][0].data.cpu().numpy())
    debug_obj['precedence'].append(mhx['precedence'][0].data.cpu().numpy())
    debug_obj['read_weights'].append(mhx['read_weights'][0].data.cpu().numpy())
    debug_obj['write_weights'].append(mhx['write_weights'][0].data.cpu().numpy())
    debug_obj['usage_vector'].append(mhx['usage_vector'][0].unsqueeze(0).data.cpu().numpy())
    return debug_obj

  def _step_forward(self, input, hx=(None, None), pass_through_memory=True):
    (chx, mhx) = hx
    _chx = chx
    zero_chx = (
      Variable(torch.zeros(_chx[0].size()[0], _chx[0].size()[1], _chx[0].size()[2])),
      Variable(torch.zeros(_chx[1].size()[0], _chx[1].size()[1], _chx[1].size()[2]))
    )
    if self.gpu_id != -1:
      zero_chx = (zero_chx[0].cuda(), zero_chx[1].cuda())
    _input = input
    input, chx = self.controller(input.unsqueeze(1), chx)
    input = input.squeeze(1)
    if self.skip_hidden_to_output:
      tmp_output, _chx = self.rnns[layer](_input.unsqueeze(1), zero_chx)
      tmp_output = tmp_output.squeeze(1)
    else:
      tmp_output = input

    # clip the controller output
    if self.clip != 0:
      output = T.clamp(input, -self.clip, self.clip)
      tmp_output = T.clamp(tmp_output, -self.clip, self.clip)
    else:
      output = input

    # the interface vector
    ξ = output

    # pass through memory
    if pass_through_memory:
      read_vecs, mhx = self.memory(ξ, mhx)
      # the read vectors
      read_vectors = read_vecs.view(-1, self.w * self.r)
    else:
      read_vectors = None

    return tmp_output, (chx, mhx, read_vectors)

  def forward(self, input, hx=(None, None, None), reset_experience=False, pass_through_memory=True, flag_list=None, additiona_vec=T.from_numpy(np.array([0,1]).astype("float32")), ctr_penalty=0.2):
    # handle packed data
    is_packed = type(input) is PackedSequence
    if is_packed:
      input, lengths = pad(input)
      max_length = lengths[0]
    else:
      max_length = input.size(1) if self.batch_first else input.size(0)
      lengths = [input.size(1)] * max_length if self.batch_first else [input.size(0)] * max_length

    batch_size = input.size(0) if self.batch_first else input.size(1)

    if not self.batch_first:
      input = input.transpose(0, 1)
    # make the data time-first

    controller_hidden, mem_hidden, last_read = self._init_hidden(hx, batch_size, reset_experience)
    if self.skip_memory_reset:
      if self.mem_hidden is not None:
        for key in mem_hidden.keys():
          mem_hidden[key].data = self.mem_hidden[key].data

    # concat input with last read (or padding) vectors
    inputs = input
    # inputs = [T.cat([input[:, x, :], last_read], 1) for x in range(max_length)]

    # batched forward pass per element / word / etc
    if self.debug:
      viz = None

    outs = [None] * max_length
    read_vectors = None

    # pass through time
    for time in range(max_length):
      flag_tsr = None
      # pass thorugh layers
      if flag_list is not None:
        if flag_list[time] == 0:
          flag_tsr = T.eye(2)[0].repeat(batch_size, 1)
          if self.gpu_id != -1:
            flag_tsr = flag_tsr.cuda()
        else:
          flag_tsr = T.eye(2)[1].repeat(batch_size, 1)
          if self.gpu_id != -1:
            flag_tsr = flag_tsr.cuda()
        if flag_list[time] == 2:
          # print(time, inputs[:,time,:,:,:])
          inputs[:,time,:,:,:] = outs[time-1]
          # print(time, inputs[:,time,:,:,:])
          print("\n")
      else:
        flag_tsr = T.eye(2)[1].repeat(batch_size, 1)
        if self.gpu_id != -1:
          flag_tsr = flag_tsr.cuda()
      # this layer's hidden states
      chx = controller_hidden
      m = mem_hidden
      __input = inputs[:,time,:,:,:]
      t_input = self.c_encoder(__input)
      t_input = T.cat([t_input, flag_tsr], 1)
      if read_vectors is None:
        t_input = T.cat([t_input, last_read], 1)
      else:
        t_input = T.cat([t_input, read_vectors], 1)
      # pass through controller
      outs[time], (chx, m, read_vectors) = \
        self._step_forward(t_input, (chx, m), pass_through_memory)
      # debug memory
      if self.debug:
        viz = self._debug(m, viz)

      # store the memory back (per layer or shared)
      mem_hidden = m
      controller_hidden = chx

      if read_vectors is not None:
        # the controller output + read vectors go into next layer
        outs[time] = T.cat([outs[time]*ctr_penalty, read_vectors], 1)
      else:
        outs[time] = T.cat([outs[time]*ctr_penalty, last_read], 1)
      outs[time] = self.output(outs[time])
      outs[time] = self.net_g(outs[time]) # generate image
      # print(time, outs[time].size())
      # inputs[time] = outs[time]

    if self.debug:
      viz = {k: np.array(v) for k, v in viz.items()}
      viz = {k: v.reshape(v.shape[0], v.shape[1] * v.shape[2]) for k, v in viz.items()}
    
    self.mem_hidden = mem_hidden
    # pass through final output layer
    # inputs = [self.output(i) for i in inputs]
    outputs = T.stack(outs, 1 if self.batch_first else 0)

    if is_packed:
      outputs = pack(output, lengths)

    if self.debug:
      return outputs, (controller_hidden, mem_hidden, read_vectors), viz
    else:
      return outputs, (controller_hidden, mem_hidden, read_vectors)

  def __repr__(self):
    s = "\n----------------------------------------\n"
    s += '{name}({hidden_size}'
    if self.rnn_type != 'lstm':
      s += ', rnn_type={rnn_type}'
    if self.num_hidden_layers != 2:
      s += ', num_hidden_layers={num_hidden_layers}'
    if self.bias != True:
      s += ', bias={bias}'
    if self.batch_first != True:
      s += ', batch_first={batch_first}'
    if self.dropout != 0:
      s += ', dropout={dropout}'
    if self.bidirectional != False:
      s += ', bidirectional={bidirectional}'
    if self.nr_cells != 5:
      s += ', nr_cells={nr_cells}'
    if self.read_heads != 2:
      s += ', read_heads={read_heads}'
    if self.cell_size != 10:
      s += ', cell_size={cell_size}'
    if self.nonlinearity != 'tanh':
      s += ', nonlinearity={nonlinearity}'
    if self.gpu_id != -1:
      s += ', gpu_id={gpu_id}'
    if self.independent_linears != False:
      s += ', independent_linears={independent_linears}'
    if self.share_memory != True:
      s += ', share_memory={share_memory}'
    if self.debug != False:
      s += ', debug={debug}'
    if self.clip != 20:
      s += ', clip={clip}'

    s += ")\n" + super(PDNC, self).__repr__() + \
      "\n----------------------------------------\n"
    return s.format(name=self.__class__.__name__, **self.__dict__)


class ConvEncoder(nn.Module):
    def __init__(self, size, dim=64, z_dim=128):
        super(ConvEncoder, self).__init__()
        main = nn.Sequential(
            nn.Conv2d(3, dim, 3, 2, padding=1),
            nn.LeakyReLU(),
            nn.Conv2d(dim, 2 * dim, 3, 2, padding=1),
            nn.LeakyReLU(),
            nn.Conv2d(2 * dim, 4 * dim, 3, 2, padding=1),
            nn.LeakyReLU(),
        )
        self.main = main
        self._w = size[1] / 8
        self._h = size[2] / 8
        self.z_dim = z_dim
        self.dim = dim
        self.linear1 = nn.Linear(int(self._w*self._h*4*dim), int(self._w*self._h*2*dim))
        self.linear2 = nn.Linear(int(self._w*self._h*2*dim), int(self._w*self._h*dim))
        self.linear3 = nn.Linear(int(self._w*self._h*dim), self.z_dim)
        self.tanh = nn.Tanh()
    
    def forward(self, input):
        output = self.main(input)
        output = output.view(-1, self._w*self._h*4*self.dim)
        output = self.linear1(output)
        output = self.linear2(output)
        output = self.linear3(output)
        output = self.tanh(output)
        return output
