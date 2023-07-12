#Author: Job Philip, Assert AI
#Date: June 22, 2023

from deepface import DeepFace
import pandas as pd
import argparse
import cv2
import numpy as np
import os

img1_path = "rtsp://admin:assert%40123@192.168.1.36:554/Streaming/Channels/601/?transportmode=unicast"
img2_path = "office_data/Job Philip/15.jpg"
db_path = "office_data"
#either provide folder_path or comment it out
folder_path = "office_data/Job Philip"

parser = argparse.ArgumentParser()
parser.add_argument("--detection", action="store_true", help="Perform face detection and alignment")
parser.add_argument("--verify", action="store_true", help="Perform face verification")
parser.add_argument("--similar", action="store_true", help="Perform face detection and similarity")
parser.add_argument("--embeddings", action="store_true", help="Gives embeddings")
parser.add_argument("--analyze", action="store_true", help="Gives emotion, age, gender and race")
parser.add_argument("--similarity", action="store_true", help="Perform similarity index with other images")
args = parser.parse_args()

rtsp = "rtsp://admin:assert%40123@192.168.1.36:554/Streaming/Channels/101/?transportmode=unicast"
video = cv2.VideoCapture(rtsp)

if args.detection:
    backends = ['opencv', 'ssd', 'dlib', 'mtcnn', 'retinaface', 'mediapipe']
    _, frame = video.read()
    face_objs = DeepFace.extract_faces(img_path = rtsp, target_size = (224, 224), detector_backend = backends[4])
    # face_objs = DeepFace.extract_faces(img_path = img1_path, target_size = (224, 224), detector_backend = backends[4])
    
    #very important - load the image outside the loop else only last bounding box will come
    # image = cv2.imread(frame)
    
    for face_obj in face_objs:
        region = face_obj['facial_area']  # Extract the facial area
        print (region)
        x, y, w, h = region['x'], region['y'], region['w'], region['h']
        x, y, w, h = int(x), int(y), int(w), int(h)
        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)

        confidence = face_obj['confidence']
        confidence_text = f'Confidence: {confidence*100:.2f}%'
        cv2.putText(frame, confidence_text, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)

    #very important - resize the image outside the loop else only last bounding box will come  
    frame = cv2.resize(frame, (600, 600))
    cv2.imshow('Image with Bounding Box', frame)
#   cv2.imshow('Detected Face', face_obj['face'])
    cv2.waitKey(0)

    cv2.destroyAllWindows()

#face_verification
if args.verify:
    result = DeepFace.verify(img1_path = img1_path, img2_path = img2_path, enforce_detection=False)
    print (result)

#face_similarity
if args.similar:
    dfs = DeepFace.find(img_path=img1_path, db_path=db_path, enforce_detection=False)
    print (dfs[0])
    # Check if the list of DataFrames is not empty
    if len(dfs) > 0:
        # Get the first DataFrame from the list
        first_df = dfs[0][["identity", "VGG-Face_cosine"]]

        # Save the first DataFrame to an Excel file
        first_df.to_excel('output.xlsx', index=False)

        # Get the first 3 image paths from the DataFrame
        image_paths = first_df['identity'].head(3).tolist()
        # Load and resize the images
        images = []
        for image_path in image_paths:
            image = cv2.imread(image_path)
            image = cv2.resize(image, (300, 300))
            images.append(image)

        # Create a blank canvas to display the images
        canvas = np.zeros((300, 900, 3), dtype=np.uint8)

        # Arrange the images side by side
        canvas[:, :300] = images[0]
        canvas[:, 300:600] = images[1]
        canvas[:, 600:] = images[2]

        # Display the canvas with the images
        cv2.imshow("Images", canvas)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
    
if args.embeddings:
    embedding_objs = DeepFace.represent(img_path = img1_path, model_name = "Facenet")
    #model_name options: VGG-Face, Facenet, Facenet512, OpenFace, DeepFace, DeepID, Dlib, ArcFace, SFace
    embedding = embedding_objs[0]["embedding"]
    assert isinstance(embedding, list)
    print (embedding)

if args.analyze:
    
    # For detection in a image
    objs = DeepFace.analyze(img_path=img1_path, actions=['age', 'gender', 'race', 'emotion'], enforce_detection=False)
    age = objs[0]['age']
    dominant_gender = objs[0]['dominant_gender']
    dominant_race = objs[0]['dominant_race']
    dominant_emotion = objs[0]['dominant_emotion']
    
    image = cv2.imread(img1_path)
    image = cv2.resize(image, (600, 600))
    text = f"Age: {age} \nGender: {dominant_gender} \nRace: {dominant_race} \nEmotion: {dominant_emotion}"
    lines = text.split('\n')
    y = 30
    for line in lines:
        cv2.putText(image, line, (20, y), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2)
        y += 30
    cv2.imshow("Image with Results", image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    
    # For detection in all images in a folder - iterate through each file in the folder
    if 'folder_path' in locals() or 'folder_path' in globals():
        filename_results = []
        for filename in os.listdir(folder_path):
            if filename.endswith(".jpg") or filename.endswith(".png"): 
                file_path = os.path.join(folder_path, filename)  # Get the full file path
                
                # Apply DeepFace.analyze() to each file
                objs = DeepFace.analyze(img_path=file_path, actions=['age', 'gender', 'race', 'emotion'], enforce_detection=False)
                age = objs[0]['age']
                dominant_gender = objs[0]['dominant_gender']
                dominant_race = objs[0]['dominant_race']
                dominant_emotion = objs[0]['dominant_emotion']

                filename_results.append([filename, age, dominant_gender, dominant_race, dominant_emotion])
        print (filename_results)

if args.similarity:
    metrics = ["cosine", "euclidean", "euclidean_l2"]

    #face verification
    result = DeepFace.verify(img1_path = img1_path, img2_path = img2_path, distance_metric = metrics[0])
    print (result)

    #face recognition
    dfs = DeepFace.find(img_path = img1_path, db_path = db_path, distance_metric = metrics[0])
    print (dfs)