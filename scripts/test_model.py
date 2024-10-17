from ultralytics import YOLO
import matplotlib.pyplot as plt
import cv2


model = YOLO('/home/chaitu/Downloads/best.pt')
#model = YOLO('/home/chaitu/Downloads/yolov8n.pt')

#path to image
image_path = '/home/chaitu/Desktop/Sample_Apple_Dataset/V&R_Insta360Cam/cam_1_000619_000331836957_reprojected_1.jpg'

results = model.predict(source=image_path, save=True, save_txt=True)

#print(results)
results.show()

img = cv2.imread(image_path)
plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
plt.axis('off')
plt.show()