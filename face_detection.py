import cv2
import os
import sys
import numpy as np
import time
import detect_face
import tensorflow as tf
import configure

class FaceDetection(object):
	"""docstring for FaceDetection"""
	def __init__(self):
		super(FaceDetection, self).__init__()
		self.model_detect = self.load_model() 
		# print (self.model_detect.minNeighbors)

	def load_model(self, filename = "model_opencv/haarcascades/haarcascade_frontalface_default.xml"):
		model = cv2.CascadeClassifier(filename)
		return model

	def detect_faces(self, image, scaleFactor = 1.3, minNeighbors=5):
	    img_copy = np.copy(image)
	    #convert the test image to gray image as opencv face detector expects gray images
	    gray = cv2.cvtColor(img_copy, cv2.COLOR_BGR2GRAY)
	    #let's detect multiscale (some images may be closer to camera than others) images
	    faces = self.model_detect.detectMultiScale(gray, scaleFactor=scaleFactor, minNeighbors = minNeighbors)
	    return faces

	def draw_faces_image(self, image, scaleFactor = 1.3, minNeighbors = 5):
		# get bounding box for face detection
		faces = self.detect_faces(image, scaleFactor, minNeighbors = minNeighbors)

		#go over list of faces and draw them as rectangles on original colored img
		for (x, y, w, h) in faces:
			cv2.rectangle(image, (x, y), (x+w, y+h), (0, 255, 0), 2)
		return image


path_video = "folder_video/sontung.mp4"
cap = cv2.VideoCapture(path_video)
face_detect = FaceDetection()
i = 0

while(cap.isOpened()):
    # Capture frame-by-frame
    ret, frame = cap.read()
    ## only detect face
    # faces = face_detect.detect_faces(frame)
    # detect face + draw rectangle face
    frame = face_detect.draw_faces_image(frame)
    # Display the resulting frame
    cv2.imshow('frame',frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# When everything done, release the capture
cap.release()
cv2.destroyAllWindows()