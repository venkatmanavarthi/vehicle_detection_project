import cv2
import numpy as np

# VideoCapture is used to capture video input
# paramenter : filename: for using locally stored files
#              pass 0 for capturing input from camera


cap = cv2.VideoCapture('data/video.mp4')

vehicle_count = 0

veh_detection = cv2.bgsegm.createBackgroundSubtractorMOG()

while True:
    frame_available, frame = cap.read()

    if frame_available == True:

        grey_image = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        blur = cv2.GaussianBlur(grey_image, (3,3),5)

        img_sub = veh_detection.apply(blur)
        dilat = cv2.dilate(img_sub, np.ones((5,5)))
        kernal = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5,5))
        dilatada = cv2.morphologyEx(dilat, cv2.MORPH_CLOSE, kernal)
        dilatada = cv2.morphologyEx(dilatada, cv2.MORPH_CLOSE, kernal)
        counter,h = cv2.findContours(dilatada, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

        for (i,c) in enumerate(counter):
            # print(c)
            (x,y,w,h) = cv2.boundingRect(c)
            if w > 60 and h > 50:
                cv2.rectangle(frame, (x,y), (x+w, y+h), (0,255,0),2)
                vehicle_count += 1
            else:
                continue
        cv2.putText(frame, "count : " + str(vehicle_count),(450,70),cv2.FONT_HERSHEY_SIMPLEX,2,(0,0,255),5)
        cv2.imshow('video', frame)
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
