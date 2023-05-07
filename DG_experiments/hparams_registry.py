# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
import numpy as np
import misc


def _define_hparam(hparams, hparam_name, default_val, random_val_fn):
    hparams[hparam_name] = (hparams, hparam_name, default_val, random_val_fn)


def _hparams(algorithm, dataset, random_seed):


    hparams = {}

    def _hparam(name, default_val, random_val_fn):

        assert(name not in hparams)
        random_state = np.random.RandomState(
            misc.seed_hash(random_seed, name)
        )
        hparams[name] = (default_val, random_val_fn(random_state))

    # Unconditional hparam definitions.

    _hparam('data_augmentation', True, lambda r: True)
    _hparam('arch', 'resnet50', lambda r: 'resnet50')
    _hparam('resnet_dropout', 0., lambda r: r.choice([0., 0.1, 0.5]))
    _hparam('nonlinear_classifier', False,
            lambda r: bool(r.choice([False, False])))

    _hparam('lr', 5e-5, lambda r: 10**r.uniform(-5, -3.5))


    _hparam('weight_decay', 0., lambda r: 10**r.uniform(-6, -2))

    _hparam('batch_size', 32, lambda r: int(2**r.uniform(3, 5.5)))


    ### hyper-params used in our experiments
    del hparams['batch_size']
    del hparams['lr']
    del hparams['weight_decay'],
    _hparam('batch_size', 32, lambda r: 32)
    _hparam('lr', 5e-5, lambda r: 5e-5)
    _hparam('weight_decay', 0, lambda r: 10**r.uniform(-6, -4))


    return hparams


def default_hparams(algorithm, dataset):
    return {a: b for a, (b, c) in _hparams(algorithm, dataset, 0).items()}


def random_hparams(algorithm, dataset, seed):
    return {a: c for a, (b, c) in _hparams(algorithm, dataset, seed).items()}
