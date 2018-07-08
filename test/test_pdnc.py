#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import warnings
warnings.filterwarnings('ignore')

import numpy as np
import getopt
import sys
import os
import math
import time
import argparse
from visdom import Visdom

sys.path.insert(0, os.path.join('..', '..'))

import torch as T
from torch.autograd import Variable as var
import torch.nn.functional as F
import torch.optim as optim

from torch.nn.utils import clip_grad_norm

from mann.predictive_dnc import PDNC
from mann.util import *

parser = argparse.ArgumentParser(description='PyTorch Differentiable Neural Computer')
parser.add_argument('-rnn_type', type=str, default='lstm', help='type of recurrent cells to use for the controller')
parser.add_argument('-nhid', type=int, default=64, help='number of hidden units of the inner nn')
parser.add_argument('-dropout', type=float, default=0, help='controller dropout')
parser.add_argument('-memory_type', type=str, default='dnc', help='dense or sparse memory: dnc | sdnc | sam')

parser.add_argument('-nhlayer', type=int, default=2, help='number of hidden layers')
parser.add_argument('-lr', type=float, default=1e-4, help='initial learning rate')
parser.add_argument('-optim', type=str, default='adam', help='learning rule, supports adam|rmsprop')
parser.add_argument('-clip', type=float, default=50, help='gradient clipping')
parser.add_argument('--skip_h', action='store_true')

parser.add_argument('-batch_size', type=int, default=100, metavar='N', help='batch size')
parser.add_argument('-mem_size', type=int, default=80, help='memory dimension')
parser.add_argument('-mem_slot', type=int, default=500, help='number of memory slots')
parser.add_argument('-read_heads', type=int, default=4, help='number of read heads')
parser.add_argument('-sparse_reads', type=int, default=10, help='number of sparse reads per read head')
parser.add_argument('-temporal_reads', type=int, default=2, help='number of temporal reads')

parser.add_argument('-sequence_max_length', type=int, default=20, metavar='N', help='sequence_max_length')
parser.add_argument('-z_dim', type=int, default=100, metavar='N', help='latent vector dimension')
parser.add_argument('-curriculum_increment', type=int, default=0, metavar='N', help='sequence_max_length incrementor per 1K iterations')
parser.add_argument('-curriculum_freq', type=int, default=1000, metavar='N', help='sequence_max_length incrementor per 1K iterations')
parser.add_argument('-cuda', type=int, default=-1, help='Cuda GPU ID, -1 for CPU')

parser.add_argument('-iterations', type=int, default=100000, metavar='N', help='total number of iteration')
parser.add_argument('-summarize_freq', type=int, default=100, metavar='N', help='summarize frequency')
parser.add_argument('-check_freq', type=int, default=100, metavar='N', help='check point frequency')
parser.add_argument('-visdom', action='store_true', help='plot memory content on visdom per -summarize_freq steps')

args = parser.parse_args()
print(args)

viz = Visdom()
# assert viz.check_connection()

if args.cuda != -1:
    print('Using CUDA.')
    T.manual_seed(1111)
else:
    print('Using CPU.')

def llprint(message):
    sys.stdout.write(message)
    sys.stdout.flush()

def generate_data(batch_size, length, size=(3, 150, 100), cuda=-1, input_length=5, loop_mode=False):
    if loop_mode:
        flag_list = [2] * length
    else:
        flag_list = [1] * length
    for i in range(input_length):
        flag_list[i] = 0
    if input_length < length:
        flag_list[input_length] = 1
    total_data = T.empty(batch_size, length, size[0], size[1], size[2]).uniform_(0, 1)
    input_data = total_data[:,:-1,:,:,:]
    target_output = total_data[:,1:,:,:,:]
    if cuda != -1:
        input_data = input_data.cuda()
        target_output = target_output.cuda()
    return var(input_data), var(target_output), flag_list

def criterion(predictions, targets):
    loss = (predictions.tanh() - targets)**2
    return T.mean(loss)

if __name__ == '__main__':

    dirname = os.path.dirname(__file__)
    ckpts_dir = os.path.join(dirname, 'checkpoints')
    if not os.path.isdir(ckpts_dir):
        os.mkdir(ckpts_dir)

    batch_size = args.batch_size
    sequence_max_length = args.sequence_max_length
    iterations = args.iterations
    summarize_freq = args.summarize_freq
    check_freq = args.check_freq
    whole_input_size = (3,40,32)
    
    # input_size = output_size = args.input_size
    mem_slot = args.mem_slot
    mem_size = args.mem_size
    read_heads = args.read_heads
    # args.output_size = args.sequence_max_length

    rnn = PDNC(
        hidden_size=args.nhid,
        whole_input_size=whole_input_size,
        whole_output_size=args.z_dim,
        rnn_type=args.rnn_type,
        num_hidden_layers=args.nhlayer,
        dropout=args.dropout,
        nr_cells=mem_slot,
        cell_size=mem_size,
        read_heads=read_heads,
        gpu_id=args.cuda,
        debug=True,
        batch_first=True,
        independent_linears=True,
        skip_hidden_to_output = args.skip_h
    )
    print(rnn)

    if args.cuda != -1:
        rnn = rnn.cuda(args.cuda)

    last_save_losses = []

    if args.optim == 'adam':
        optimizer = optim.Adam(rnn.parameters(), lr=args.lr, eps=1e-9, betas=[0.9, 0.98]) # 0.0001
    elif args.optim == 'adamax':
        optimizer = optim.Adamax(rnn.parameters(), lr=args.lr, eps=1e-9, betas=[0.9, 0.98]) # 0.0001
    elif args.optim == 'rmsprop':
        optimizer = optim.RMSprop(rnn.parameters(), lr=args.lr, momentum=0.9, eps=1e-10) # 0.0001
    elif args.optim == 'sgd':
        optimizer = optim.SGD(rnn.parameters(), lr=args.lr) # 0.01
    elif args.optim == 'adagrad':
        optimizer = optim.Adagrad(rnn.parameters(), lr=args.lr)
    elif args.optim == 'adadelta':
        optimizer = optim.Adadelta(rnn.parameters(), lr=args.lr)
    
    (chx, mhx, rv) = (None, None, None)
    for epoch in range(iterations + 1):
        llprint("\rIteration {ep}/{tot}".format(ep=epoch, tot=iterations))
        optimizer.zero_grad()
        input_data, target_output, flag_list = generate_data(batch_size, args.sequence_max_length, size=whole_input_size, cuda=args.cuda)

        if rnn.debug:
            output, (chx, mhx, rv), v = rnn(input_data, (None, mhx, None), reset_experience=True, pass_through_memory=True, flag_list=flag_list)
        else:
            output, (chx, mhx, rv) = rnn(input_data, (None, mhx, None), reset_experience=True, pass_through_memory=True, flag_list=flag_list)
        loss = criterion((output), target_output)
        loss.backward()

        T.nn.utils.clip_grad_norm(rnn.parameters(), args.clip)
        optimizer.step()
        loss_value = loss.data[0]

        summarize = (epoch % summarize_freq == 0)
        take_checkpoint = (epoch != 0) and (epoch % check_freq == 0)
        increment_curriculum = (epoch != 0) and (epoch % args.curriculum_freq == 0)
        
        # detach memory from graph
        mhx = { k : (v.detach() if isinstance(v, var) else v) for k, v in mhx.items() }

        last_save_losses.append(loss_value)

        if summarize:
            loss = np.mean(last_save_losses)
            llprint("\n\tAvg. Logistic Loss: %.4f\n" % (loss))
            last_save_losses = []