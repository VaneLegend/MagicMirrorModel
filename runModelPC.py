# PC example for gathering data and running the model with that data
# runModelPI is used for gathering data using the raspberry pi picamera library

import cv2
import time
import os
from PIL import Image
import torch
from transformers import ViTForImageClassification, ViTImageProcessor

# Directory to save the images
save_directory = "./pictures"  # Change to your desired directory

# Create the directory if it doesn't exist
if not os.path.exists(save_directory):
    os.makedirs(save_directory)

# Open the default camera (0 usually refers to the built-in camera)
cap = cv2.VideoCapture(0)

if not cap.isOpened():
    print("Error: Could not open camera.")
    exit()

cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1920)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 1080)

# Take 5 pictures
for i in range(5):
    ret, frame = cap.read()  # Read the frame from the camera
    if ret:
        # Define the file path for each image
        image_path = os.path.join(save_directory, f"image_{i+1}.jpg")
        # Save the image
        cv2.imwrite(image_path, frame)
        print(f"Captured {image_path}")
    else:
        print("Error: Could not capture image.")
    # Pause between captures (optional)
    cv2.waitKey(1000)  # Wait for 1 second
    # TODO: while waiting, prompt user to look left, right, up, and other angles to get a full grading

# Release the camera and close any OpenCV windows
cap.release()
cv2.destroyAllWindows()

# run model on images
# Path to your fine-tuned model
model_dir = "./skintelligent-acne"

# Load the model from the local directory
model = ViTForImageClassification.from_pretrained(model_dir, local_files_only=True)

# Load the preprocessor (image processor) configuration
processor = ViTImageProcessor.from_pretrained(model_dir, local_files_only=True)

# Set the model to evaluation mode
model.eval()

# Define the directory where your test images are stored
image_dir1 = "./pictures"

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
print(f"Average score: {id2label[sum/count]}")