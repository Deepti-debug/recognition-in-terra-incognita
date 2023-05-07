import json
import numpy as np
#from domainbed import datasets
#from domainbed import algorithms
#from domainbed.lib.fast_data_loader import FastDataLoader
#from domainbed import networks
import torch
import torch.nn as nn
import os
import argparse
import time

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.autograd as autograd
from torch.autograd import Variable

import copy
import numpy as np
from collections import defaultdict, OrderedDict

import networks

class EoA(torch.nn.Module):
    def __init__(self, input_shape, hparams, num_classes):
        super(EoA, self).__init__()
        self.hparams = hparams
        if input_shape[1:3] == (224, 224):
            self.featurizer = networks.ResNet(input_shape, hparams)
        self.classifier = networks.Classifier(
            self.featurizer.n_outputs,
            num_classes,
            hparams['nonlinear_classifier'])

        self.network = nn.Sequential(self.featurizer, self.classifier)
        
        if input_shape[1:3] == (224, 224):
            self.featurizer_mo = networks.ResNet(input_shape, hparams)
        self.classifier_mo = networks.Classifier(
            self.featurizer.n_outputs,
            num_classes,
            hparams['nonlinear_classifier'])
        
        self.network = self.network.cuda()
        self.network = torch.nn.parallel.DataParallel(self.network).cuda()

        self.network_sma = nn.Sequential(self.featurizer_mo, self.classifier_mo)
        self.network_sma = self.network_sma.cuda()
        self.network_sma = torch.nn.parallel.DataParallel(self.network_sma).cuda()
        
    def predict(self, x):
        if self.hparams['SMA']:
            return self.network_sma(x)
        else:
            return self.network(x)

class Algorithm(torch.nn.Module):

    def __init__(self, input_shape, num_classes, num_domains, hparams):
        super(Algorithm, self).__init__()
        self.hparams = hparams

    def update(self, minibatches, unlabeled=None):

        raise NotImplementedError

    def predict(self, x):
        raise NotImplementedError

class ERM(Algorithm):
    """
    Empirical Risk Minimization (ERM)
    """

    def __init__(self, input_shape, num_classes, num_domains, hparams):
        super(ERM, self).__init__(input_shape, num_classes, num_domains,hparams)
        
        if input_shape[1:3] == (224, 224):
            self.featurizer = networks.ResNet(input_shape, hparams)
        self.classifier = networks.Classifier(
            self.featurizer.n_outputs,
            num_classes,
            self.hparams['nonlinear_classifier'])

        self.network = nn.Sequential(self.featurizer, self.classifier)
        self.optimizer = torch.optim.Adam(
            self.network.parameters(),
            lr=self.hparams["lr"],
            weight_decay=self.hparams['weight_decay']
        )

    def update(self, minibatches, unlabeled=None):
        all_x = torch.cat([x for x,y in minibatches])
        all_y = torch.cat([y for x,y in minibatches])
        loss = F.cross_entropy(self.predict(all_x), all_y)

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        return {'loss': loss.item()}

    def predict(self, x):
        return self.network(x)

class MovingAvg:
    def __init__(self, network):
        self.network = network
        self.network_sma = copy.deepcopy(network)
        self.network_sma.eval()
        self.sma_start_iter = 100
        self.global_iter = 0
        self.sma_count = 0
    def update_sma(self):
        self.global_iter += 1
        new_dict = {}
        if self.global_iter>=self.sma_start_iter:
            self.sma_count += 1
            for (name,param_q), (_,param_k) in zip(self.network.state_dict().items(), self.network_sma.state_dict().items()):
                if 'num_batches_tracked' not in name:
                   new_dict[name] = ((param_k.data.detach().clone()* self.sma_count + param_q.data.detach().clone())/(1.+self.sma_count))
        else:
            for (name,param_q), (_,param_k) in zip(self.network.state_dict().items(), self.network_sma.state_dict().items()):
                if 'num_batches_tracked' not in name:
                    new_dict[name] = param_q.detach().data.clone()
        self.network_sma.load_state_dict(new_dict)

class ERM_SMA(Algorithm, MovingAvg):
    """
    Empirical Risk Minimization (ERM) with Simple Moving Average (SMA) prediction model
    """
    def __init__(self, input_shape, num_classes, num_domains, hparams):
        Algorithm.__init__(self, input_shape, num_classes, num_domains, hparams)

        if input_shape[1:3] == (224, 224):
            self.featurizer = networks.ResNet(input_shape, hparams)
        self.classifier = networks.Classifier(
                    self.featurizer.n_outputs,
                    num_classes,
                    self.hparams['nonlinear_classifier'])
        self.network = nn.Sequential(self.featurizer, self.classifier)
        self.optimizer = torch.optim.Adam(
                        self.network.parameters(),
                        lr=self.hparams["lr"],
                        weight_decay=self.hparams['weight_decay']
                        )
        MovingAvg.__init__(self, self.network)

    def update(self, minibatches, unlabeled=None):
        all_x = torch.cat([x for x,y in minibatches])
        all_y = torch.cat([y for x,y in minibatches])
        loss = F.cross_entropy(self.network(all_x), all_y)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        self.update_sma()
        return {'loss': loss.item()}

    def predict(self, x):
        self.network_sma.eval()
        return self.network_sma(x)