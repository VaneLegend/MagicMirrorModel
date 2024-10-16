import os
from PIL import Image
import torch
from transformers import ViTForImageClassification, ViTImageProcessor

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
for img_file in os.listdir(image_dir1):
    if img_file.endswith(('.png', '.jpg', '.jpeg')):
        img_path = os.path.join(image_dir1, img_file)
        image = Image.open(img_path)

        # Preprocess the image using the ViT processor
        inputs = processor(images=image, return_tensors="pt")

        # Run inference on the image
        with torch.no_grad():
            outputs = model(**inputs)
            logits = outputs.logits

        # Get the predicted class
        predicted_class = logits.argmax(-1).item()
        print(f"Image: {img_file}, Predicted Class: {id2label[predicted_class]}")

for img_file in os.listdir(image_dir2):
    if img_file.endswith(('.png', '.jpg', '.jpeg')):
        img_path = os.path.join(image_dir2, img_file)
        image = Image.open(img_path)

        # Preprocess the image using the ViT processor
        inputs = processor(images=image, return_tensors="pt")

        # Run inference on the image
        with torch.no_grad():
            outputs = model(**inputs)
            logits = outputs.logits

        # Get the predicted class
        predicted_class = logits.argmax(-1).item()
        print(f"Image: {img_file}, Predicted Class: {id2label[predicted_class]}")

for img_file in os.listdir(image_dir3):
    if img_file.endswith(('.png', '.jpg', '.jpeg')):
        img_path = os.path.join(image_dir3, img_file)
        image = Image.open(img_path)

        # Preprocess the image using the ViT processor
        inputs = processor(images=image, return_tensors="pt")

        # Run inference on the image
        with torch.no_grad():
            outputs = model(**inputs)
            logits = outputs.logits

        # Get the predicted class
        predicted_class = logits.argmax(-1).item()
        print(f"Image: {img_file}, Predicted Class: {id2label[predicted_class]}")