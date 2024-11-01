
import os
import requests
import zipfile 
from pathlib import Path

# setup the data path
data_path = Path("data/")
image_path = data_path / 'pizza_steak_sushi'

if image_path.is_dir():
	print(f"{image_path} already exists.")
else:
	print(f"Did not found any {image_path} directory, creating one...")
	image_path.mkdir(parents=True, exist_ok=True)
	with open(data_path / "pizza_steak_sushi.zip", "wb") as file:
		response = requests.get("https://github.com/mrdbourke/pytorch-deep-learning/raw/main/data/pizza_steak_sushi.zip")
		print(f"Downloading pizza, steak, sushi...")
		file.write(response.content)
	with zipfile.ZipFile(data_path / "pizza_steak_sushi.zip", "r") as zip_ref:
		print(f"Unzipping pizza, steak, sushi...")
		zip_ref.extractall(image_path)
	os.remove(data_path / 'pizza_steak_sushi.zip')

# Downloading pizza, sushi, steak
# with open(data_path / "pizza_steak_sushi.zip", "wb") as file:
# 	response = requests.get("https://github.com/mrdbourke/pytorch-deep-learning/raw/main/data/pizza_steak_sushi.zip")
# 	print(f"Downloading pizza, steak, sushi...")
# 	file.write(response.content)

# # Unzip pizza, steak, sushi data
# with zipfile.ZipFile(data_path / "pizza_steak_sushi.zip", "r") as zip_ref:
# 	print(f"Unzipping pizza, steak, sushi...")
# 	zip_ref.extractall(image_path)

# # remove the zip file
# os.remove(data_path / 'pizza_steak_sushi.zip')
