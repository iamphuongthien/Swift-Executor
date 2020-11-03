import cv2
import numpy as np

def findLines():
    video = cv2.VideoCapture("realtimeObDec/video/road_car_view.mp4")
    while True:
        ret, frame = video.read()
        if not ret:
            video = cv2.VideoCapture("realtimeObDec/video/road_car_view.mp4")
            continue
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        low_yellow = np.array([18, 94, 140])
        up_yellow = np.array([48, 255, 255])
        mask = cv2.inRange(hsv, low_yellow, up_yellow)
        edge = cv2.Canny(mask, 75, 150)
        lines = cv2.HoughLinesP(edge, 1, np.pi/180, 50, maxLineGap=50)
        if lines is not None:
            for line in lines:
                x1, y1, x2, y2 = line[0]
                cv2.line(frame,(x1, y1), (x2, y2),(0 , 255, 0),3)
        cv2.imshow("Fame", frame)
        # cv2.imshow("Edge", edge)
        if cv2.waitKey(20) & 0xFF == ord('q'):
            break
    video.release()
    cv2.destroyAllWindows()

def init():
    findLines()