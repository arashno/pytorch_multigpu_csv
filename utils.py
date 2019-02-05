from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import time
import numpy as np
import torch
import sys
import argparse

from datetime import datetime

"""
This methods counts the number of examples in an input file and calculates the number of batches for each epoch.
Args:
    filename: the name of the input file
    batch_size: batch size
Returns:
    number of samples and number of batches
"""
def count_input_records(filename, batch_size):
    with open(filename) as f:
        num_samples = sum(1 for line in f)
    num_batches = num_samples / batch_size
    return num_samples, int(num_batches) if num_batches.is_integer() else int(num_batches) + 1

"""
This methods parses an input string to determine details of a learning rate policy.

Args:
    policy_type: Type of the policy
    details_str: the string to parse
"""
def get_policy(policy_type, details_str):
    if policy_type == 'constant':

        def custom_policy(step):
            return float(details_str)

        return custom_policy

    if policy_type == 'piecewise_linear':
        details = [float(x) for x in details_str.split(',')]
        length = len(details)
        assert length % 2 ==1 , 'Invalid policy details'
        assert all(item.is_integer() for item in details[0:int((length - 1) / 2)]), 'Invalid policy details'

        def custom_policy(step):
            for i,x in enumerate(details[0:int((length - 1) / 2)]):
                if step <= int(x):
                    return details[int((length - 1) / 2) + i]
            return details[-1]

        return custom_policy

    if policy_type == 'exponential':
        details = [float(x) for x in details_str.split(',')]
        assert len(details) == 3, 'Invalid policy details'
        assert details[1].is_integer(), 'Invalid policy details'

        def custom_policy(step):
            return details[0] * details[2] ^ (step / details[1])

        return custom_policy


def adjust_param(optimizer, param_name, policy, step):
    param = policy(step)
    print('%s is set to %f' % (param_name , param))
    for param_group in optimizer.param_groups:
        param_group[param_name] = param

"""
this method return an instance of a type of optimization algorithms based on the arguments.
Args:
    opt_type: type of the algorithm
    lr: learning rate policy
"""
def get_optimizer(opt_type, params, init_lr):
    if opt_type.lower() == 'momentum':
        return torch.optim.SGD(params, init_lr, momentum = 0.9)
    elif opt_type.lower() == 'adam':
        return torch.optim.Adam(params)
    elif opt_type.lower() == 'adadelta':
        return torch.optim.Adadelta(params)
    elif opt_type.lower() == 'adagrad':
        return torch.optim.Adagrad(init_lr)
    elif opt_type.lower() == 'rmsprop':
        return torch.optim.RMSProp(init_lr)
    elif opt_type.lower() == 'sgd':
        return torch.optim.SGD(params, init_lr)
    else:
        print('invalid optimizer')
        return None

class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

def accuracy(output, target, topk=(1,)):
    """Computes the accuracy over the k top predictions for the specified values of k"""
    with torch.no_grad():
        maxk = max(topk)
        batch_size = target.size(0)

        _, pred = output.topk(maxk, 1, True, True)
        pred = pred.t()
        correct = pred.eq(target.view(1, -1).expand_as(pred))

        res = []
        for k in topk:
            correct_k = correct[:k].view(-1).float().sum(0, keepdim=True)
            res.append(correct_k.mul_(100.0 / batch_size))
        return res

def smart_save(path, max_to_keep = -1):
    dir_name = os.path.dirname(path)
    cache_file = os.path.join(dir_name, 'checkpoint.txt')
    if not os.path.exists(cache_file):
        open(cache_file, 'a').close()
    with open(cache_file) as f:
        lines = [line.strip() for line in f]
    lines.insert(0, os.path.basename(path))
    if  max_to_keep < len(lines):
        to_delete = lines.pop()
        os.remove(os.path.join(dir_name, to_delete))
    with open(cache_file, 'w') as f:
        for line in lines:
            f.write('%s\n' % line)
    return path

def smart_load(path):
    assert os.path.exists(path)
    if os.path.isdir(path):
        cache_file = os.path.join(path, 'checkpoint.txt')
        assert os.path.exists(cache_file), 'No cache file found'
        with open(cache_file) as f:
            lines = [line.strip() for line in f]
        return os.path.join(path, lines[0])
    else:
        return path
