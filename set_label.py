import cv2

import os
from os import listdir
from os.path import isfile, exists


def detection_keypress(k):
    for i in range(1, 10):
        if ord(str(i)) ==  k:
            return True, str(i)
    return False, None

img = cv2.imread('nguoi1.jpg', 1)


while True:
    cv2.imshow("frame", img)

    key = cv2.waitKey(2000000)

    key =  detection_keypress(key)

    if key[0]:
        with open('log_img.txt', 'a') as f:
            f.write(f'nguoi1.jpg, {key[1]}\n')

        print(key[1])

    if key == ord('q'):
        break
cv2.destroyAllWindows()