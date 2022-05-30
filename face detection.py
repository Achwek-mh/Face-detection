import numpy as np 
import cv2

cap = cv2.VideoCapture(0) 
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades+ 'haarcascade_frontalface_default.xml') #load classifier that detect features in capture
eye_cascade = cv2.CascadeClassifier(cv2.data.haarcascades+ 'haarcascade_eye.xml')
#nose_cascade = cv2.CascadeClassifier(cv2.data.haarcascades+ 'haarcascade_nose.xml')
mouth_cascade = cv2.CascadeClassifier(cv2.data.haarcascades+ 'haarcascade_smile.xml')

while True:
	ret,frame= cap.read()

	gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY) # detection works with grayscale
	faces =face_cascade.detectMultiScale(gray, 1.3, 5) #apply classifier detect face return location | 1.3: higher number, higher algo perf ,smaller accuracy | 5: at least 5 rectangles close to each other to determine its a face
	for (x,y,w,h) in faces:
		cv2.rectangle(frame, (x,y), (x+w,y+h), (255,0,0),3) # draw rectange on face (stable)
		
		#get gray face to apply eye,nose,mouth detection and color face to show rectangle
		face_gray = gray[y:y+w, x:x+w]
		face_color = frame[y:y+h, x:x+w]

		#eye detection (stable)
		eyes = eye_cascade.detectMultiScale(face_gray, 1.5, 5)
		for (ex,ey,ew,eh) in eyes:
			cv2.rectangle(face_color, (ex,ey), (ex+ew,ey+eh), (0,255,0),3)

		#nose detection (stable)
		#noses = nose_cascade.detectMultiScale(face_gray,1.3,5)
		#for (nx,ny,nw,nh) in noses:
	    #cv2.rectangle(face_color, (nx,ny), (nx+nw,ny+nh), (0,0,255),3)

		#mouth detection (not very stable)
		mouths = mouth_cascade.detectMultiScale(face_gray,1.3,6)
		for (mx,my,mw,mh) in mouths:
			cv2.rectangle(face_color, (mx,my), (mx+mw,my+mh), (0,255,255),3)
	

	
	cv2.imshow('frame', frame)

	if cv2.waitKey(1) == ord('q'):
		break

cap.release()
cv2.destroyAllWindows()