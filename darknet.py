from __future__ import division
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import numpy as np
from torchviz import make_dot
from graphviz import Digraph
from utils import *


class Darknet(nn.Module):
    def __init__(self, cfgfile):
        self.blocks = parse_cfg(cfgfile)
        self.net_info, self.module_list = create_modules(self.blocks)

    def forward(self, x, CUDA):
        modules = self.blocks[1:]
        outputs = {}  # persist the outputs for the route layer

        write = 0

        for i, module in enumerate(modules):
            module_type = (module["type"])

            if module_type == "convolutional" or module_type == "upsample":
                x = self.module_list[i](x)

            elif module_type == "route":
                layers = module["layers"]
                layers = [int(a) for a in layers]

                if (layers[0]) > 0:
                    layers[0] = layers[0] - i

                if len(layers) == 1:
                    x = outputs[i + (layers[0])]
                else:
                    if (layers[1] > 0):
                        layers[1] = layers[1] - i

                    map1 = outputs[i + layers[0]]
                    map2 = outputs[i + layers[1]]

                    x = torch.cat((map1, map2), 1)

            elif module_type == "shortcut":
                from_ = int(module["from"])
                x = outputs[i - 1] + outputs[i + from_]

            elif module_type == "yolo":

                anchors = self.module_list[i][0].anchors

                # Get the input dimensions
                inp_dim = int(self.net_info["height"])

                # Get the number of classes
                num_classes  = int(module["classes"])

                #Transform
                x = x.data
                x = predict_transform(x, inp_dim, anchors, num_classes, CUDA)
                if not write:
                    detections = x
                    write = 1
                else:
                    detections = torch.cat((detections, x), 1)

            outputs[i] = x




class EmptyLayer(nn.Module):
    def __init__(self):
        super(EmptyLayer, self).__init__()


class DetectionLayer(nn.Module):
    def __init__(self, anchors):
        super(DetectionLayer, self).__init__()
        self.anchors = anchors





def parse_cfg(cfgfile):
    """
    TODO - add err handling and logging.
    :param: cfgfile - a yolo cfg file
    :return: a list of blocks. Each block describes a block in the neural network to be built.

        A block is represented as a dictionary in the list of blocks.
        This is just a list of blocks - not the actual torch.nn layers (modules)
        Yolo blocks can have more than one layer! i.e Conv = convolutional + ReLU
    """

    file = open(cfgfile, 'r')
    lines = file.read().split('\n')                 # store the lines in a list
    lines = [x for x in lines if len(x) > 0]        # eliminate empty lines
    lines = [x for x in lines if x[0] != '#']       # eliminate the comments
    lines = [x.strip().lstrip() for x in lines]     # eliminate leading and trailing whitespace

    block = {}
    blocks = []

    for line in lines:
        if line[0] == "[":                           # bracket indicates start of a new block
            if len(block) != 0:                      # if block is not empty - this is a previous block
                blocks.append(block)
                block = {}                           # make a new block
            block["type"] = line[1:-1].rstrip()      # pull the name of the CFG block out of the cfg file
        else:
            key, value = line.split("=")
            block[key.rstrip()] = value.lstrip()

    blocks.append(block)

    return blocks


def create_modules(blocks):
    """
    TODO - add err handling and logging

    :param blocks: list of blocks in the cfg file (created by parse_cfg())
    :return: nn.ModuleList - a list of native Torch.nn modules for each block from the cfg file
    """
    net_info = blocks[0]                            # First block of .cfg is not a layer, but metadata about the network
    module_list = nn.ModuleList()                   # get a list of native Torch modules
    prev_filters = 3                                # each Conv layer needs number of filters from previous layer
    output_filters = []                             # persisted filters from previous layers for blocks of type ROUTE

    for index, x in enumerate(blocks[1:]):
        module = nn.Sequential()

        if x["type"] == "convolutional":

            # Get the info about the layer

            activation = x["activation"]
            try:                                    # Test for existence of batch_normalize
                batch_normalize = int(x["batch_normalize"])
                bias = False
            except:
                batch_normalize = 0
                bais = True

            filters = int(x["filters"])
            padding = int(x["pad"])
            kernel_size = int(x["size"])
            stride = int(x["stride"])

            if padding:
                pad = (kernel_size - 1) // 2
            else:
                pad = 0

            # Add the convolutional layer

            conv = nn.Conv2d(prev_filters, filters, kernel_size, stride, pad, bias=bias)
            module.add_module("conv_{0}".format(index), conv)

            # Add the batch normalization layer
            if batch_normalize:
                bn = nn.BatchNorm2d(filters)
                module.add_module("batch_norm_{0}".format(index), bn)


            # Check the activation
            # It is either Linear or leaky ReLU for YOLO
            if activation == "leaky":
                activn = nn.LeakyReLU(0.1, inplace=True)
                module.add_module("leaky_{0}".format(index), activn)

        elif x["type"] == "upsample":                 # Use Bilinear2dUpSampling for upsampling type Blocks
            stride = int(x["stride"])
            upsample = nn.Upsample(scale_factor=2, mode="bilinear")
            module.add_module("upsample_{}".format(index), upsample)

        elif x["type"] == "route":                    # It is a route layer
            x["layers"] = x["layers"].split(',')

            # Start of a route
            start = int(x["layers"][0])

            # if there is an end
            try:
                end = int(x["layers"][1])
            except:
                end = 0

            # Positive annotation
            if start > 0:
                start = start - index
            if end > 0:
                end = end - index

            route = EmptyLayer()

            module.add_module("route_{0}".format(index), route)
            if end < 0:
                filters = output_filters[index + start] + output_filters[index + end]
            else:
                filters = output_filters[index + start]

        elif x["type"] == "shortcut":                   # shortcut block is a skip connection
            shortcut = EmptyLayer()
            module.add_module("shortcut_{}".format(index), shortcut)

        elif x["type"] == "yolo":
            mask = x["mask"].split(",")
            mask = [int(x) for x in mask]

            anchors = x["anchors"].split(",")
            anchors = [int(x) for x in anchors]
            anchors = [(anchors[i], anchors[i+1]) for i in range(0, len(anchors), 2)]
            anchors = [anchors[i] for i in mask]

            detection = DetectionLayer(anchors)
            module.add_module("Detection_{}".format(index), detection)

        module_list.append(module)
        prev_filters = filters
        output_filters.append(filters)

    return (net_info, module_list)


