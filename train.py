"""
	Trains a PyTorch image classification model using device-agnosic code
"""
import argparse
import os
import torch
import data_setup, engine, model_builder, utils

from torchvision import transforms

parser = argparse.ArgumentParser()

# Arguments
parser.add_argument("-td", "--train_dir",default="data/pizza_steak_sushi/train")
parser.add_argument("-tsd", "--test_dir", default="data/pizza_steak_sushi/test")
parser.add_argument("-lr", "--learning_rate", type=float, default=0.001)
parser.add_argument("-batch_size", "--batch_size", type=int, default=32)
parser.add_argument("-eps", "--epochs", type=int, default=5)
parser.add_argument("-hu", "--hidden_units", type=int, default=10)

arguments = parser.parse_args()

# Hyperparameters
NUM_EPOCHS = arguments.epochs
BATCH_SIZE = arguments.batch_size
HIDDEN_UNITS = arguments.hidden_units
LEARNING_RATE = arguments.learning_rate
# Setup directories
train_dir = arguments.train_dir
test_dir = arguments.test_dir

def main():
	# Setup target device
	device = "cuda" if torch.cuda.is_available() else "cpu"

	data_transform = transforms.Compose([
		transforms.Resize((64, 64)),
		transforms.ToTensor()
	])

	train_dataloader, test_dataloader, class_names = data_setup.create_dataloaders(
		train_dir=train_dir,
		test_dir=test_dir,
		transform=data_transform,
		batch_size=BATCH_SIZE
	)

	# model
	model = model_builder.TinyVGG(
		input_shape=3,
		hidden_units=HIDDEN_UNITS,
		output_shape=len(class_names)
	).to(device)

	# loss function and optimizer
	loss_fn = torch.nn.CrossEntropyLoss()
	optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)

	# Start training with help from engine.py
	engine.train(
		model=model,
		train_dataloader=train_dataloader,
		test_dataloader=test_dataloader,
		optimizer=optimizer,
		loss_fn=loss_fn,
		epochs=NUM_EPOCHS,
		device=device
	)

	# Save a model with help from utils.py
	utils.save_model(
		model=model,
		target_dir="./model",
		model_name="Model_Sushi_Pizza_Steak.pth"
	)

if __name__ == "__main__":
	print(f"-----------------------------------------------\nStarting the training with the following hyperparameters:\nTraining directory: {train_dir}\nTesting directory: {test_dir}\nNumber of epochs: {NUM_EPOCHS}\nLearning rate: {LEARNING_RATE}\nBatch size: {BATCH_SIZE}\nNumber of hidden units: {HIDDEN_UNITS}\n-----------------------------------------------")
	main()