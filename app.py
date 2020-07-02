from flask import Flask, render_template, redirect, url_for, request, send_file, jsonify
from flask_socketio import SocketIO, emit
from gevent.pywsgi import WSGIServer
from gevent.monkey import patch_all
import requests
import logging
import sys
import argparse
import os
from processing import Processing
import json
import io
from io import StringIO, BytesIO
from PIL import Image
import base64
import cv2
import numpy as np
import random
import copy
from face_detection import FaceDetection
from face_recognition import FaceRecognition
import pickle
from sklearn.metrics.pairwise import cosine_similarity as cs
from datetime import datetime

class RunFaceID(object):
	def __init__(self):
		super(RunFaceID, self).__init__()
		self.face_detection = FaceDetection()
		self.face_recognition = FaceRecognition()

	def predict(self, array_embeddings, embeddings_source):
		return "unknown"

	def processing(self, images, embeddings_source):
		frame = copy.deepcopy(images)
		faces = self.face_detection.detect_faces(frame)
		if faces is None or len(faces) < 1:
			return None
		data = {}
		array_img = []
		labels = []
		for x,y,w,h in faces:
			if w > 0 and h > 0:
				img_crop = frame[y:y+h, x:x+w, :]
				array_img.append(img_crop)
				# labels.append("unknown")
		array_img = np.array(array_img)
		# data["labels"] = labels
		if count >= NUMBER_FRAME:
			array_embeddings = self.face_recognition.embedding_image(array_img)
			data["labels"] = self.predict_labels(array_embeddings, embeddings_source)
		data["bounding_boxs"] = faces
		return data

# runfaceid = RunFaceID()

cap = cv2.VideoCapture(0)
face_detect = FaceDetection()
i = 0

while(True):
    # Capture frame-by-frame
    ret, frame = cap.read()
    faces = face_detect.detect_faces(frame)
    if len(faces) > 0:
    	cv2.imwrite(str(i) + ".png", frame)
    	i += 1
    	if i == 5:
    		break

    # Display the resulting frame
    cv2.imshow('frame',frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break