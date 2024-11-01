# %%writefile going_modular/model_builder.py

"""
Contains PyTorch model code to instantiate a TinyVGG model.
"""

import torch
from torch import nn

"""Creates the TinyVGG architecture.

  Replicates the TinyVGG architecture from the CNN explainer website in PyTorch.
  See the original architecture here: https://poloclub.github.io/cnn-explainer/

  Args:
    input_shape: An integer indicating number of input channels.
    hidden_units: An integer indicating number of hidden units between layers.
    output_shape: An integer indicating number of output units.
  """

class TinyVGG(nn.Module):
	"""
    	Model architecture copying TinyVGG from: 
    	https://poloclub.github.io/cnn-explainer/
	"""
	def __init__(self, input_shape : int, hidden_units : int, output_shape : int):
		super().__init__()
		self.conv_block_1 = nn.Sequential(
			nn.Conv2d(
				in_channels=input_shape,
				out_channels=hidden_units,
				kernel_size=1,
				stride=1,
				padding=1
			),
			nn.ReLU(),
			nn.Conv2d(
				in_channels=hidden_units,
				out_channels=hidden_units,
				kernel_size=1,
				stride=1,
				padding=1
			),
			nn.ReLU(),
			nn.MaxPool2d(
				kernel_size=1,
				stride=2
			)
		)

		self.conv_block_2 = nn.Sequential(
			nn.Conv2d(
				in_channels=hidden_units,
				out_channels=hidden_units,
				kernel_size=1,
				stride=1,
				padding=1
			),
			nn.ReLU(),
						nn.Conv2d(
				in_channels=hidden_units,
				out_channels=hidden_units,
				kernel_size=1,
				stride=1,
				padding=1
			),
			nn.ReLU(),
			nn.MaxPool2d(
				kernel_size=1,
				stride=2
			)
		)

		self.classifier = nn.Sequential(
			nn.Flatten(),
			# Where did this in_features shape come from? 
            # It's because each layer of our network compresses and changes the shape of our inputs data.
            nn.Linear(in_features=hidden_units*361,
						out_features=output_shape)
		)

	def forward(self, x : torch.Tensor):
		x = self.conv_block_1(x)
		x = self.conv_block_2(x)
		x = self.classifier(x)
		return x