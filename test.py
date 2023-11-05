import cv2
import numpy as np
from ultralytics import YOLO
import MOT_func as MOT

a = np.empty((3,3,4))
b = np.empty((3,3,4))
print(a)
print(b)
#c = np.concatenate((a[0,0,:], b[0,1,:]))
c = np.stack((a[:,0,:], b[:,1,:]), axis=1)
print(c)

'''
# Load the YOLOv8 model
model = YOLO('yolov8n.pt')

# Open the video file
img = cv2.imread("test.jpg")
img_new = cv2.imread("test.jpg")

# Run YOLOv8 inference on the frame
results = model(img, conf=0.3, iou=0.5, show=False)


new_cp = []
for i in range(len(results[0].boxes)):
	if(i==20): break
	else:
		new_cp.append([[i]*4, results[0].boxes[i].xywh.tolist()[0], results[0].boxes[i].xyxy.tolist()[0]])
new_cp = np.array(new_cp)
print(np.array(new_cp))


# knn
new_y = MOT.KNN(cp[:,1,0:2], cp[:,0,0], new_cp[:,1,0:2])
print(prediction)


cp = new_cp


cv2.rectangle(img, (432, 43), (569, 396), (0, 0, 255), 3)
cv2.line(img, (432, 43), (569, 396), (0, 0, 255), 5)
cv2.line(img, (500, 219), (500, 219), (0, 255, 255), 5)
cv2.putText(img, str(1), (432, 43), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 1, cv2.LINE_AA)
cv2.imwrite("test_draw.jpg",img)'''
