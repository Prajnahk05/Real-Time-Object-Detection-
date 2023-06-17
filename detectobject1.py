import numpy as np
import time
import cv2
import pyttsx3
language = 'en'

con = 0.5
thresh = 0.3

LABELS = None
with open('coco.names', 'r') as f:
    LABELS = [line.strip() for line in f.readlines()]


np.random.seed(42)
COLORS = np.random.randint(0, 255, size=(len(LABELS), 3),dtype="uint8")




net = cv2.dnn.readNet('yolov2.weights', 'yolov2.cfg')
engine = pyttsx3.init()

def process():
	vs = cv2.VideoCapture(0)
	oldvr=[]
	while True:
		vr=[]
		ret,image = vs.read()



		(H, W) = image.shape[:2]
	

		ln = net.getLayerNames()
		ln = [ln[i[0] - 1] for i in net.getUnconnectedOutLayers()]
	



		blob = cv2.dnn.blobFromImage(image, 1 / 255.0, (416, 416),swapRB=True, crop=False)
		net.setInput(blob)
		start = time.time()
		layerOutputs = net.forward(ln)
		end = time.time()


		print("[INFO] YOLO took {:.6f} seconds".format(end - start))



		boxes = []
		confidences = []
		classIDs = []


		for output in layerOutputs:

			for detection in output:


				scores = detection[5:]
				classID = np.argmax(scores)
				confidence = scores[classID]



				if confidence > con:




					
					box = detection[0:4] * np.array([W, H, W, H])
					print(box)
					(centerX, centerY, width, height) = box.astype("int")
	


					x = int(centerX - (width / 2))
					y = int(centerY - (height / 2))



					boxes.append([x, y, int(width), int(height)])
					confidences.append(float(confidence))
					classIDs.append(classID)

		


		idxs = cv2.dnn.NMSBoxes(boxes, confidences, con, thresh)


		if len(idxs) > 0:

			for i in idxs.flatten():

				(x, y) = (boxes[i][0], boxes[i][1])
				(w, h) = (boxes[i][2], boxes[i][3])
	

				color = [int(c) for c in COLORS[classIDs[i]]]
				cv2.rectangle(image, (x, y), (x + w, y + h), color, 2)
				text = "{}: {:.4f}".format(LABELS[classIDs[i]], confidences[i])
				
				print("old vector")
				print(oldvr)
				labeld=LABELS[classIDs[i]]
				msg=""
				if labeld in oldvr:
					for j in range(0,len(oldvr)):
						if oldvr[j] == labeld:
							print("label availabe",oldvr[j],labeld)
							print("X axis",oldvr[j+1],"Now x value",x)
							d=oldvr[j+1]-x
							print("d value",d)
							if d >0.0:
								print(labeld,"Moving to right direction")
								msg=labeld+"Moving to right direction"
								engine.say(msg)
								engine.runAndWait()
								msg=""
														
							if d <0.0:
								print(labeld,"Moving to left direction")
								msg=labeld+"Moving to left direction"
								#engine.say(msg)
								#engine.runAndWait()
								#msg=""
				cv2.putText(image, text+msg, (x, y - 5), cv2.FONT_HERSHEY_SIMPLEX,0.5, color, 2)
				vr.append(LABELS[classIDs[i]])
				vr.append(x)
				vr.append(y)
		print(vr)
		oldvr=vr
		cv2.imshow("Surveillance Camera", image)
		key = cv2.waitKey(1) & 0xFF
		if key == ord("q"):
			break



	cv2.destroyAllWindows()
process()

