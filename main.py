import os
import time
import pyautogui
import cv2
import dlib
from collections import deque
import numpy as np

import imutils
from scipy.spatial import distance as dist
from imutils.video import FileVideoStream
from imutils.video import VideoStream
from imutils import face_utils

# Red color
redLower = (0, 130, 20)
redUpper = (20, 255,255)

#redLower = (155, 25, 0)
#redUpper = (179, 255, 255)

BUFFER_MAX = 32
pts = deque(maxlen=BUFFER_MAX)
 
def startGame():
    
    # locate the play and resume button and click them to start/resume the game
    playButton = pyautogui.locateOnScreen('buttons/play_button.png')
    resumeButton = pyautogui.locateOnScreen('buttons/resume_button.png')
    
    # Mark x and y coordinate to click
    if playButton:
        clickX, clickY = pyautogui.locateCenterOnScreen('buttons/play_button.png')
    elif resumeButton:
        clickX, clickY = pyautogui.locateCenterOnScreen('buttons/resume_button.png')
    else:
        print(" [INFO] GAME SCREEN NOT FOUND!")
    
    # perform 2 clicks, first to bring focus out of python window and other to actually 
    # click the play/resume button
    pyautogui.click(clickX, clickY)
    time.sleep(2.0)
    pyautogui.click(clickX, clickY)
    
 
def main():
    # init flags
    faceDetected = False
    ballDetected = False
    
    # initialize dlib's face detector (HOG-based)
    print("[INFO] loading face detector...")
    face_detector = dlib.get_frontal_face_detector()
    
    # start the video stream thread
    print("[INFO] starting video stream thread...")
    vs = cv2.VideoCapture(0)
    time.sleep(1.0)
    
    # loop over frames from the video stream
    while True:
    
        # grab the frame from the threaded video file stream
        frame = vs.read()
        frame = frame[1]
        
        # if we are viewing a video and we did not grab a frame,
        # then we have reached the end of the video
        if frame is None:
            break
        
        frame = imutils.resize(frame, width=1080)
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        
        # detect faces in the grayscale frame
        faces = face_detector(gray, 0)
        
        # convert dlib's rectangle to a OpenCV-style bounding box
        # [i.e., (x, y, w, h)], then draw the face bounding box
        # draw them on screen if you find it
        if len(faces) != 0:
            (x, y, w, h) = face_utils.rect_to_bb(faces[0])
            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
            faceDetected = True 
        
        
        blurred = cv2.GaussianBlur(frame, (11, 11), 0)
        hsv = cv2.cvtColor(blurred, cv2.COLOR_BGR2HSV)
        # construct a mask for the color "green", then perform
        # a series of dilations and erosions to remove any small
        # blobs left in the mask
        mask = cv2.inRange(hsv, redLower, redUpper)
        mask = cv2.erode(mask, None, iterations=2)
        mask = cv2.dilate(mask, None, iterations=2)
        
        # uncomment if you wanna see the mask
        #cv2.imshow("mask", mask)
        
        # find contours in the mask and initialize the current
        # (x, y) center of the ball
        contours = cv2.findContours(mask.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        contours = imutils.grab_contours(contours)
        center = None

        # only proceed if at least one contour was found
        if len(contours) > 0:
            # find the largest contour in the mask, then use
            # it to compute the minimum enclosing circle and
            # centroid
            c = max(contours, key=cv2.contourArea)
            ((x, y), radius) = cv2.minEnclosingCircle(c)
            M = cv2.moments(c)
            center = (int(M["m10"] / M["m00"]), int(M["m01"] / M["m00"]))
            # only proceed if the radius meets a minimum size
            if radius > 10:
                # draw the circle and centroid on the frame,
                # then update the list of tracked points
                cv2.circle(frame, (int(x), int(y)), int(radius), (0, 255, 255), 2)
                cv2.circle(frame, center, 5, (0, 0, 255), -1)
                ballDetected = True
        
        # update the points queue
        pts.appendleft(center)
        
        # loop over the set of tracked points
        for i in range(1, len(pts)):
            # if either of the tracked points are None, ignore them
            if pts[i - 1] is None or pts[i] is None:
                continue
            # otherwise, compute the thickness of the line and
            # draw the connecting lines
            thickness = int(np.sqrt(BUFFER_MAX / float(i + 1)) * 2.5)
            cv2.line(frame, pts[i - 1], pts[i], (0, 0, 255), thickness)
        
        # show the frame
        cv2.imshow("Frame", frame)
        key = cv2.waitKey(1) & 0xFF
        
        # if the `q` key was pressed, break from the loop
        if key == ord("q"):
            break

    #startGame()
    
    # do a bit of cleanup
    vs.release()
    cv2.destroyAllWindows()
    
if __name__ == '__main__':
    main()
