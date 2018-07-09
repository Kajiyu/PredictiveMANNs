#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import warnings
warnings.filterwarnings('ignore')

import numpy as np
import getopt
import sys
import os
import json
import math
import random
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

parser.add_argument('-batch_size', type=int, default=64, metavar='N', help='batch size')
parser.add_argument('-mem_size', type=int, default=80, help='memory dimension')
parser.add_argument('-mem_slot', type=int, default=500, help='number of memory slots')
parser.add_argument('-read_heads', type=int, default=4, help='number of read heads')
parser.add_argument('-sparse_reads', type=int, default=10, help='number of sparse reads per read head')
parser.add_argument('-temporal_reads', type=int, default=2, help='number of temporal reads')

parser.add_argument('-sequence_max_length', type=int, default=20, metavar='N', help='sequence_max_length')
parser.add_argument('-input_length', type=int, default=10, metavar='N', help='input_length')
parser.add_argument('-z_dim', type=int, default=100, metavar='N', help='latent vector dimension')
parser.add_argument('-curriculum_increment', type=int, default=0, metavar='N', help='sequence_max_length incrementor per 1K iterations')
parser.add_argument('-curriculum_freq', type=int, default=1000, metavar='N', help='sequence_max_length incrementor per 1K iterations')
parser.add_argument('-cuda', type=int, default=-1, help='Cuda GPU ID, -1 for CPU')

parser.add_argument('-iterations', type=int, default=100000, metavar='N', help='total number of iteration')
parser.add_argument('-summarize_freq', type=int, default=50, metavar='N', help='summarize frequency')
parser.add_argument('-check_freq', type=int, default=2000, metavar='N', help='check point frequency')
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

def generate_data(
    batch_size, total_length, input_length, npy_idx,
    base_path='/home/kaji/data/frames_cleanpass/np_out/',
    cuda=-1, loop_mode=False):
    if loop_mode:
        flag_list = [2] * total_length
    else:
        flag_list = [1] * total_length
    for i in range(input_length):
        flag_list[i] = 0
    if input_length < total_length:
        flag_list[input_length] = 1
    npy_path = base_path + str(npy_idx) + ".npy"
    npy_values = np.load(npy_path)
    input_data = []
    target_output = []
    for i in range(batch_size):
        start_idx = random.randrange(0, len(npy_values)-total_length-3)
        in_data = npy_values[start_idx:start_idx+total_length]
        out_data = npy_values[(start_idx+1):(start_idx+1)+total_length]
        input_data.append(in_data)
        target_output.append(out_data)
    input_data = np.stack(input_data, axis=0).astype("float32")
    target_output = np.stack(target_output, axis=0).astype("float32")
    input_data = T.from_numpy(input_data)
    target_output = T.from_numpy(target_output)
    if cuda != -1:
        input_data = input_data.cuda()
        target_output = target_output.cuda()
    return var(input_data), var(target_output), flag_list

def criterion(predictions, targets, input_length):
    print(predictions.size(), targets.size())
    loss = (predictions[:,input_length:,:,:,:].tanh() - targets[:,input_length:,:,:,:])**2
    return T.mean(loss)

if __name__ == '__main__':

    dirname = os.path.dirname(__file__)
    ckpts_dir = os.path.join(dirname, 'checkpoints')
    if not os.path.isdir(ckpts_dir):
        os.mkdir(ckpts_dir)
    args_json_dir = os.path.join(dirname, 'args_json')
    if not os.path.isdir(args_json_dir):
        os.mkdir(args_json_dir)

    batch_size = args.batch_size
    sequence_max_length = args.sequence_max_length
    input_length = args.input_length
    iterations = args.iterations
    summarize_freq = args.summarize_freq
    check_freq = args.check_freq
    whole_input_size = (3,80,40)
    
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
    loss_history = []
    for epoch in range(iterations + 1):
        llprint("\rIteration {ep}/{tot}".format(ep=epoch, tot=iterations))
        optimizer.zero_grad()
        npy_idx = random.randrange(0, 16)
        input_data, target_output, flag_list = generate_data(batch_size, sequence_max_length, input_length, npy_idx, cuda=args.cuda)
        print(input_data.size())
        if rnn.debug:
            output, (chx, mhx, rv), v = rnn(input_data, (None, mhx, None), reset_experience=False, pass_through_memory=True, flag_list=flag_list)
        else:
            output, (chx, mhx, rv) = rnn(input_data, (None, mhx, None), reset_experience=False, pass_through_memory=True, flag_list=flag_list)
        loss = criterion((output), target_output, input_length)
        loss.backward()

        T.nn.utils.clip_grad_norm(rnn.parameters(), args.clip)
        optimizer.step()
        loss_value = loss.data[0]
        loss_history.append(loss_value)

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
        
        if take_checkpoint:
            llprint("\nSaving Checkpoint ... "),
            check_ptr = os.path.join(ckpts_dir, 'cleanpass_step_{}.pth'.format(epoch))
            np.save('./args_json/read_weights/cleanpass_step_{}.npy'.format(epoch), v["read_weights"])
            np.save('./args_json/link_matrix/cleanpass_step_{}.npy'.format(epoch), v["link_matrix"])
            np.save('./args_json/precedence/cleanpass_step_{}.npy'.format(epoch), v["precedence"])
            np.save('./args_json/usage_vector/cleanpass_step_{}.npy'.format(epoch), v["usage_vector"])
            np.save('./args_json/write_weights/cleanpass_step_{}.npy'.format(epoch), v["write_weights"])
            np.save('./args_json/memory/cleanpass_step_{}.npy'.format(epoch), v["memory"])
            cur_weights = rnn.state_dict()
            T.save(cur_weights, check_ptr)
            llprint("Done!\n")
    
    with open('./args_json/losses.json','w') as f:
        json.dump({"losses":loss_history},f)
    input_data, target_output, flag_list = generate_data(batch_size, sequence_max_length, input_length, npy_idx, cuda=args.cuda)
    output, (chx, mhx, rv), v = rnn(input_data, (None, mhx, None), reset_experience=False, pass_through_memory=True, flag_list=flag_list)
    loss = criterion((output), target_output)
    loss_value = loss.data[0]
    print("Finally Loss: ", loss)
    np.save('./args_json/input/cleanpass_step_result.npy', input_data.data[0].cpu().numpy())
    np.save('./args_json/target/cleanpass_step_result.npy', target_output.data[0].cpu().numpy())
    np.save('./args_json/output/cleanpass_step_result.npy', output.tanh().data[0].cpu().numpy())
    np.save('./args_json/read_weights/cleanpass_step_result.npy', v["read_weights"])
    np.save('./args_json/link_matrix/cleanpass_step_result.npy', v["link_matrix"])
    np.save('./args_json/precedence/cleanpass_step_result.npy', v["precedence"])
    np.save('./args_json/usage_vector/cleanpass_step_result.npy', v["usage_vector"])
    np.save('./args_json/write_weights/cleanpass_step_result.npy', v["write_weights"])
    np.save('./args_json/memory/cleanpass_step_result.npy', v["memory"])