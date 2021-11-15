import cv2
import numpy as np


# VideoCapture is used to capture video input
# paramenter : filename: for using locally stored files
#              pass 0 for capturing input from camera


cap = cv2.VideoCapture('data/video.mp4')

while True:
    frame_available, frame1 = cap.read()

    if frame_available == True:
        cv2.imshow('video', frame1)
        key = cv2.waitKey(1)
        # print(key)
        if key == 13:
            print("Termination initiated")
            break
    else:
        print("Video ended")
        break

cv2.destroyAllWindows()
cap.release()
