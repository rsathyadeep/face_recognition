import numpy as np
import pandas as pd 
import matplotlib.pyplot as plt
import cv2
import os 
import insightface
from insightface.app import FaceAnalysis
from sklearn.metrics import pairwise
from sklearn.neighbors import NearestNeighbors


#configuring face analysis 
faceapp = FaceAnalysis(name='buffalo_l',root='insightface_model',providers=['CPUExecutionProvider'])
faceapp.prepare(ctx_id=0, det_size=(640,640), det_thresh=0.5)

 #resgistration form  collecting name and role 

person_name = input('Enter your name: ')

trials = 3
for i in range(trials):
    role = input("""
    please choose
    1.developer or manager
    2.other


  enter number either 1 or 2
""")
    if role in ("1","2"):
         if role == "1":
              role = 'developer or manager'
         else:
             role = 'other'

         break
    else:
        print('invalid input')
    if i == 3:
          print('exceeds maximum trails')

key = person_name+'@'+role
print('Your name =', person_name)
print('Your role =', role)
print('Key=',key)
print(person_name,role) 

# Initialize the video capture object
cap = cv2.VideoCapture(0)
face_embeddings = []
sample = 0
while True:
    ret, frame = cap.read()
    if not ret:
        print('Unable to read from camera')
        break 

    # Perform face detection using insightface
    results = faceapp.get(frame, max_num=1)
    
    for res in results:
        sample += 1
        # Extract bounding box and convert to integers
        x1, y1, x2, y2 = res['bbox'].astype(int)
        
        # Draw a rectangle around the detected face
        cv2.rectangle(frame, (x1, y1), (x2, y2), (235, 206, 135), 2)


        # Extract facial features (embedding)
        embeddings = res['embedding']
        face_embeddings.append(embeddings)

    # Stop collecting embeddings after 100 samples
    if sample >= 200:
        break

    # Display the frame with bounding boxes
    cv2.imshow('frame', frame)

    # Break on 'q' key press
    if cv2.waitKey(1) == ord('q'):
        break
    
cap.release()  
cv2.destroyAllWindows() 











