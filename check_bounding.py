import cv2
import numpy as np

point1 = ()
point2 = ()
COLOR_RED = (0,0,255)
COLOR_GREEN = (0,255,0)
COUNT_DETECT = 0

BOX_DRAW = []


def mouse_drawing(event, x, y, flags, params):
    global point1, point2
    if event == cv2.EVENT_LBUTTONDOWN:
        point1 = (x, y)
        point2 = ()

    elif event == cv2.EVENT_RBUTTONDOWN:
        point2 = (x, y)
        point1 = ()


def check_bounding_box(limit, p):

    for box in limit:
        if box[0][0] > p[0]:
            continue
        if box[0][1] > p[1]:
            continue
        if box[1][0] < p[0]:
            continue
        if box[1][1] < p[1]:
            continue
        return True, box

    return False, ()

point = [[(50, 50),(150,150)],[(90, 90),(190, 190)]]

# with open('log.txt', 'w') as f:
#     for item in point:
#         f.write(f'{item[0][0]}, {item[0][1]}, {item[1][0]}, {item[1][1]}\n')

# point = np.loadtxt('log.txt', dtype=np.int , delimiter=',')

img = cv2.imread('nguoi1.jpg', 1)
cv2.namedWindow("Frame")
cv2.setMouseCallback("Frame", mouse_drawing)
for item in point:
    cv2.rectangle(img, item[0], item[1], COLOR_GREEN, thickness=5)

while True:
    cv2.imshow('Frame', img)
    if point1:
        result = check_bounding_box(point , point1)
        if result[0]:
            if not BOX_DRAW.__contains__(result[1]):
                BOX_DRAW.append(result[1])
                COUNT_DETECT += 1

            cv2.rectangle(img, result[1][0], result[1][1], COLOR_RED, thickness=5)

            # Paint frame
            cv2.rectangle(img, (0, 0), (55, 30), (255,255,255), cv2.FILLED)
            cv2.putText(img, str(COUNT_DETECT), (0 + 15, 0 + 20), cv2.FONT_ITALIC, 1, (0, 0, 255))
            point1 = ()
    
    if point2:
        result = check_bounding_box(point , point2)
        if result[0]:
            if BOX_DRAW.__contains__(result[1]):
                BOX_DRAW.remove(result[1])
                COUNT_DETECT -= 1
            cv2.rectangle(img, result[1][0], result[1][1], COLOR_GREEN, thickness=5)
            # Paint frame
            cv2.rectangle(img, (0, 0), (55, 30), (255,255,255), cv2.FILLED)
            cv2.putText(img, str(COUNT_DETECT), (0 + 15, 0 + 20), cv2.FONT_ITALIC, 1, (0, 0, 255))
            point2 = ()
        
    if cv2.waitKey(20) & 0xFF == ord('q'):
        break

cv2.destroyAllWindows()
