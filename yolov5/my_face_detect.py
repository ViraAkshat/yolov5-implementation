import cv2 as cv
from deepface import DeepFace as df

rtsp = "rtsp://admin:assert%40123@192.168.1.36:554/Streaming/Channels/101/?transportmode=unicast"
video = cv.VideoCapture(rtsp)

while True:
    _, frame = video.read()

    if _:
        result = df.analyze(frame, detector_backend='opencv')

        # print(result.shape)

        detected_faces = result['region']

        for face in detected_faces:
            x, y, w, h = face['box']
        
        cv.imshow(frame)
    
    else:
        break

video.release()