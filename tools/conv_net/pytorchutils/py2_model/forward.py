#!/usr/bin/env python
__doc__ = """

Inference functions

Nicholas Turner <nturner.cs@princeton.edu>, 2017-8
"""

import sys,time

import torch
import numpy as np

import utils


def forward(net, scanner, scan_spec, activation=None):

    with torch.no_grad():
        
        start = time.time()
        inputs = scanner.pull()

        while inputs:
            inputs = make_variables(inputs)
    
            outputs = run_forward_pass(net, inputs, activation)
    
            push_outputs(scanner, outputs, scan_spec)
    
            end = time.time()
            sys.stdout.write("Elapsed: %3f\n" % (end-start)); sys.stdout.flush()

            start = end
            inputs = scanner.pull()

    return scanner


def make_variables(inputs):
    expanded = [np.expand_dims(arr, axis=0) for (k,arr) in inputs.items()]
    return [utils.to_torch(arr) for arr in expanded]


def run_forward_pass(net, inputs, activation=None):
    
    outputs = net(*inputs)

    if activation is not None:
        outputs = list(map(activation, outputs))

    return outputs


def push_outputs(scanner, outputs, scan_spec):

    fmt_outputs = dict()
    for (i,k) in enumerate(scan_spec.keys()):
        fmt_outputs[k] = extract_data(outputs[i])

    scanner.push(fmt_outputs)


def extract_data(expanded_variable):
    return np.squeeze(expanded_variable.data.cpu().numpy(), axis=(0,))
