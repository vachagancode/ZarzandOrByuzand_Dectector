import torch
import torchvision
from torchvision import transforms
from typing import List
import matplotlib.pyplot as plt
import model_builder
import argparse

parser = argparse.ArgumentParser()

parser.add_argument("-path", "--path", default="example.jpg")

data_transform = transforms.Compose(
	[transforms.Resize((64, 64))]
)

device = "cuda" if torch.cuda.is_available() else "cpu"
class_names = ['pizza', 'steak', 'sushi']
MODEL_DICT_PATH = "model/Model_Sushi_Pizza_Steak.pth"
EXAMPLE_PATH = parser.parse_args().path
model = model_builder.TinyVGG(input_shape=3, hidden_units=10, output_shape=3).to(device)
model_state_dict = torch.load(MODEL_DICT_PATH, weights_only=True)
model.load_state_dict(model_state_dict)

def pred_and_plot_image(model: torch.nn.Module,
						image_path: str,
						class_names: List[str],
						transform=None,
						device: torch.device = device,
						):

	target_image = torchvision.io.read_image(str(image_path)).type(torch.float32)

	if transform:
		target_image = transform(target_image)

	target_image = target_image / 255

	model.to(device)

	model.eval()
	with torch.inference_mode():
		target_image = target_image.unsqueeze(dim=0)

		target_image_pred = model(target_image.to(device))
	
	target_image_pred_probs = torch.softmax(target_image_pred, dim=1)
	target_image_pred_label = torch.argmax(target_image_pred_probs, dim=1)

	plt.imshow(target_image.squeeze().permute(1, 2, 0)) # make sure it's the right size for matplotlib
	if class_names:
	    title = f"Prediction: {class_names[target_image_pred_label.cpu()]} | Probability: {target_image_pred_probs.max().cpu():.3f}/1"
	else: 
		title = f"Prediction: {target_image_pred_label} | Probability: {target_image_pred_probs.max().cpu():.3f}/1"
	
	print(target_image_pred_probs)
	plt.title(title)
	plt.axis(False)
	plt.show()

pred_and_plot_image(
	model=model,
	image_path=EXAMPLE_PATH,
	class_names=class_names,
	device=device,
	transform=data_transform,
)
