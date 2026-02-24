# pip install opencv-python

import cv2
import numpy as np

#using camera 1
cap = cv2.VideoCapture(0)
#cap = cv2.VideoCapture('1.mp4')
fps = cap.get(cv2.CAP_PROP_FPS);  # print ('FPS:', fps)

frame_width = int(cap.get(3))
frame_height = int(cap.get(4))

while cap.isOpened():
    ret, frame = cap.read()

    if not ret:
        print("Can't receive frame (stream end?). Exiting ...")
        break

    HSV = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)  # 1- convert frame from BGR to HSV

    # 2- define the range of
    lower = np.array([0, 0, 0])
    upper = np.array([255, 10, 255])

    # check if the HSV of the frame is lower or upper
    def_mask = cv2.inRange(HSV, lower, upper)
    result = cv2.bitwise_and(frame, frame, mask=def_mask)

    # Draw rectangular bounded line on the detected area
    (contours, hierarchy) = cv2.findContours(def_mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    for pic, contour in enumerate(contours):
        area = cv2.contourArea(contour)
        if (area > 300):  # to remove the noise
            # Constructing the size of boxes to be drawn around the detected area
            x, y, w, h = cv2.boundingRect(contour)
            frame = cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 0, 0), 2)

            cv2.putText(frame, 'DEFECT', (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

    cv2.imshow("Tracking DEFECT", frame)
    #cv2.imshow("Mask",def_mask)
    #cv2.imshow("And",result)

    # out.write(frame)

    if cv2.waitKey(25) & 0xFF == ord('f') or cv2.waitKey(25) & 0xFF == ord('F'):
        break

cap.release()
# out.release()
# cv2.destroyAllWindows()



