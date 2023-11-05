import cv2
import numpy as np
import MOT_func as MOT
from PIL import ImageGrab
from ultralytics import YOLO

# Load the YOLOv8 model
model = YOLO('yolov8n.pt')

# Open the video file
cap = cv2.VideoCapture(0)

# Loop through the video frames
cp = np.zeros((20,3,4))
KNN_flag = 0
while True:
	# Read a frame from the screem
	img_rgb = ImageGrab.grab()
	img_bgr = cv2.cvtColor(np.array(img_rgb), cv2.COLOR_RGB2BGR)
	#frame = img_bgr
	frame = img_bgr[int(img_bgr.shape[0]/2):-1, 0:int(img_bgr.shape[1]/2), :]


	new_cp = []
	# Run YOLOv8 inference on the frame
	results = model(frame, conf=0.3, iou=0.5, show=False)


	# 將 results 處理成 new_cp
	if(len(results[0].boxes)==0): new_cp = np.zeros((1,3,4))
	else:
		for i in range(len(results[0].boxes)):
			if(i==20): break
			else:
				new_cp.append([[i+1,1,0,0], results[0].boxes[i].xywh.tolist()[0], results[0].boxes[i].xyxy.tolist()[0]])
		new_cp = np.array(new_cp)


	# KNN
	if(KNN_flag==0): 
		KNN_flag = 1
		cp = MOT.MOT_table(new_cp, new_cp, 0)
	else: 
		cp = MOT.MOT_table(cp, new_cp, 1)
	cp = cp.astype("int")


	# DQN
	


	# Visualize
	frame_plt = frame
	frame_plt = cv2.circle(frame_plt, cp[0,1,0:2], 1, (255, 0, 255), 4)
	for i in range(20):
		if(cp[i,0,0]!=0):
			frame_plt = cv2.rectangle(frame_plt, cp[i,2,0:2].tolist(), cp[i,2,2:4].tolist(), (0, 0, 255), 3)
			frame_plt = cv2.putText(frame_plt, str(cp[i,0,0].tolist()), cp[i,2,0:2].tolist(), cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 0, 255), 3, cv2.LINE_AA)

	# Display the annotated frame
	cv2.imshow("YOLOv8 Inference", frame_plt)



	# Break the loop if 'q' is pressed
	if cv2.waitKey(1) & 0xFF == ord("q"):
		break

# Release the video capture object and close the display window
cv2.destroyAllWindows()