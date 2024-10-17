from ultralytics import YOLO
import cv2
import matplotlib.pyplot as plt
import os

# Load the trained YOLOv8 model
model = YOLO('/home/chaitu/Downloads/best.pt')  # Path to the trained model

# Load the image you want to test
image_path = '/home/chaitu/Desktop/Sample_Apple_Dataset/V&R_Insta360Cam/cam_1_000619_000331836957_reprojected_1.jpg'
image = cv2.imread(image_path)

# save directory for detection results
save_dir = '/home/chaitu/Documents/Code/AppleProject/test_results/'

# Create the directory if it doesn't exist
os.makedirs(save_dir, exist_ok=True)

# Run inference on the image
results = model.predict(source=image_path, save=True, save_txt=True, save_dir=save_dir)

# Assuming 'apple' is the 0th class in the dataset (check your data.yaml to confirm the class index)
apple_class_index = 0  # Update this index if 'apple' has a different class ID

# Count the number of detected apples
detected_apples = sum([1 for box in results[0].boxes if int(box.cls[0]) == apple_class_index])

print(f"Number of apples detected: {detected_apples}")

# Display the image with bounding boxes using OpenCV and Matplotlib
# YOLOv8 will save the predicted image in 'runs/predict/'
predicted_image_path = 'runs/predict/image.jpg'  # Update if the filename changes
img = cv2.imread(predicted_image_path)
img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

# Display the image with detected apples
plt.figure(figsize=(10, 10))
plt.imshow(img_rgb)
plt.axis('off')
plt.show()
