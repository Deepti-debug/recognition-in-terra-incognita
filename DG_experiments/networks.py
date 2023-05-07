
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models

#from domainbed.lib import wide_resnet
import copy




class Identity(nn.Module):
	"""An identity layer"""
	def __init__(self):
		super(Identity, self).__init__()

	def forward(self, x):
		return x

class ResNet(torch.nn.Module):
	"""ResNet with the softmax chopped off and the batchnorm frozen"""
	def __init__(self, input_shape, hparams):
		super(ResNet, self).__init__()

		if hparams['arch']=='resnet50':
			self.network = torchvision.models.resnet50(pretrained=True)
			self.n_outputs = 2048

		nc = input_shape[0]
		if nc != 3:
			tmp = self.network.conv1.weight.data.clone()

			self.network.conv1 = nn.Conv2d(
				nc, 64, kernel_size=(7, 7),
				stride=(2, 2), padding=(3, 3), bias=False)

			for i in range(nc):
				self.network.conv1.weight.data[:, i, :, :] = tmp[:, i % 3, :, :]

		del self.network.fc
		self.network.fc = Identity()

		self.freeze_bn()
		self.hparams = hparams
		self.dropout = nn.Dropout(hparams['resnet_dropout'])

	def forward(self, x):
		return self.dropout(self.network(x))

	def train(self, mode=True):
		super().train(mode)
		self.freeze_bn()

	def freeze_bn(self):
		for m in self.network.modules():
			if isinstance(m, nn.BatchNorm2d):
				m.eval()

# def Featurizer(input_shape, hparams):
# 	if input_shape[1:3] == (224, 224):
# 		return ResNet(input_shape, hparams)
# 	else:
# 		raise NotImplementedError


def Classifier(in_features, out_features, is_nonlinear=False):
	if is_nonlinear:
		return torch.nn.Sequential(
			torch.nn.Linear(in_features, in_features // 2),
			torch.nn.ReLU(),
			torch.nn.Linear(in_features // 2, in_features // 4),
			torch.nn.ReLU(),
			torch.nn.Linear(in_features // 4, out_features))
	else:
		return torch.nn.Linear(in_features, out_features)