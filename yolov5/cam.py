import cv2 as cv

rtsp = "rtsp://admin:assert%40123@192.168.1.36:554/Streaming/Channels/101/?transportmode=unicast"
video = cv.VideoCapture(rtsp)

while True:
    _, frame = video.read()
    cv.imshow("RTSP", frame)
    k = cv.waitKey(1)
    if k == ord('q'):
        break

video.release()
cv.destroyAllWindows()

