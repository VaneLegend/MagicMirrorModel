# This code gathers images using the picamera library, the runModelPC.py file is for testing on the pc
# this code is only meant to be run

import picamera
import time
import os
from PIL import Image
import torch
from transformers import ViTForImageClassification, ViTImageProcessor

# Directory to save the images
save_directory = "/home/pi/pictures"  # Change to your desired directory

# Create the directory if it doesn't exist
if not os.path.exists(save_directory):
    os.makedirs(save_directory)

# Initialize the camera
camera = PiCamera()

# Camera warm-up time
time.sleep(2)

try:
    # Take 5 pictures
    for i in range(5):
        # Define the file path for each image
        image_path = os.path.join(save_directory, f"image_{i+1}.jpg")
        
        # Capture the image
        camera.capture(image_path)
        print(f"Captured {image_path}")
        
        # Pause between captures (optional)
        time.sleep(1)
finally:
    # Close the camera
    camera.close()

# Path to your fine-tuned model
model_dir = "./skintelligent-acne"

# Load the model from the local directory
model = ViTForImageClassification.from_pretrained(model_dir, local_files_only=True)

# Load the preprocessor (image processor) configuration
processor = ViTImageProcessor.from_pretrained(model_dir, local_files_only=True)

# Set the model to evaluation mode
model.eval()

# Define the directory where your test images are stored
image_dir1 = "./Dermi_Acne_Dataset/test"
image_dir2 = "./Dermi_Acne_Dataset/train"
image_dir3 = "./Dermi_Acne_Dataset/valid"

# Load the id-to-label mapping from the model config file (this can be loaded from config.json if not hardcoded)
id2label = {
    0: "level -1",
    1: "level 0",
    2: "level 1",
    3: "level 2",
    4: "level 3"
}

# Iterate over your test images
count = 0
sum = 0
for img_file in os.listdir(image_dir1):
    if img_file.endswith(('.png', '.jpg', '.jpeg')):
        img_path = os.path.join(image_dir1, img_file)
        image = Image.open(img_path)
        count += 1

        # Preprocess the image using the ViT processor
        inputs = processor(images=image, return_tensors="pt")

        # Run inference on the image
        with torch.no_grad():
            outputs = model(**inputs)
            logits = outputs.logits

        # Get the predicted class
        predicted_class = logits.argmax(-1).item()
        print(f"Image: {img_file}, Predicted Class: {id2label[predicted_class]}")
        sum += predicted_class

# print the average score of the user
print(f"Average score: {id2label[(int)(sum/count)]}")
