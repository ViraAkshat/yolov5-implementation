import torch
import cv2
import sys
sys.path.append('sort')
from sort import *

model = torch.hub.load(   'yolov5' # Use backend for yolov5 in this folder
                        , 'custom' # to use model in this folder
                        , path='yolov5s.pt' # the name of model is this folder
                        , source='local' # to use backend from this folder
                        , force_reload=True # clear catch
                        , device = 'cpu' # I want to use CPU
                    ) 

model.conf = 0.25 # NMS confidence threshold
model.iou = 0.45  # IoU threshold
model.multi_label = False  # NMS multiple labels per box
model.max_det = 1000  # maximum number of detections per image
model.classes = [2]   # person

tracker = Sort()
colors = [(0, 255, 0), (0, 0, 255), (255, 0, 0), (255, 255, 0), (0, 255, 255), (255, 0, 255)]


# video = cv2.VideoCapture(0) # Read USB Camera
video = cv2.VideoCapture('cars_on_highway.mp4') # Read USB Camera

frame_width = int(video.get(3))
frame_height = int(video.get(4)) 
size = (frame_width, frame_height)

annotated_video = cv2.VideoWriter('filename.avi', cv2.VideoWriter_fourcc(*'MJPG'), 10, size)

while(video.isOpened()):
    # Read Frame
    ret, frame = video.read()
    if not ret:
        print('Reached the end of the video!')
        break
    # Object Detection
    results = model(frame)
    # print("results: ", results)
    # cv2.imshow('Object detector', results.render()[0])

    bounding_boxes = results.xyxy[0].cpu().numpy()
    tracked_objects = tracker.update(bounding_boxes)

    for obj in tracked_objects:
        bbox = obj[:4]
        track_id = int(obj[4])
        color = colors[track_id % len(colors)]
        x1, y1, x2, y2 = bbox.astype(int)
        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
        cv2.putText(frame, str(track_id), (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, color, 2)

    cv2.imshow('Object Tracking', frame)
    annotated_video.write(frame)

    # Press 'q' to quit
    if cv2.waitKey(1) == ord('q'): 
        break

# Clean up
video.release()
cv2.destroyAllWindows()