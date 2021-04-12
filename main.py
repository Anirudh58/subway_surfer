import os
import time
import pyautogui
import cv2
import dlib
from collections import deque
import numpy as np
import playsound

import imutils
from scipy.spatial import distance as dist
from imutils.video import FileVideoStream
from imutils.video import VideoStream
from imutils import face_utils
from threading import Thread

# By default horizontal movements controlled by head, enable to use football
USE_BALL = False

# Game Boundaries
UP_LIMIT_START_X = 350
UP_LIMIT_START_Y = 100
UP_LIMIT_END_X = 550
UP_LIMIT_END_Y = 100

DOWN_LIMIT_START_X = 350
DOWN_LIMIT_START_Y = 200
DOWN_LIMIT_END_X = 550
DOWN_LIMIT_END_Y = 200

LEFT_LIMIT_START_X = 350
LEFT_LIMIT_START_Y = 100
LEFT_LIMIT_END_X = 350
LEFT_LIMIT_END_Y = 200

RIGHT_LIMIT_START_X = 550
RIGHT_LIMIT_START_Y = 100
RIGHT_LIMIT_END_X = 550
RIGHT_LIMIT_END_Y = 200


# Game states
gameActuallyStarted = False
playerHorizontalPosition = 0 # 0 -> center, -1 -> left, +1 -> right
playerVerticalPosition = 0 # 0 -> normal -1 -> down +1 -> up

# human height
humanHeight = 183

# Red color range
redLower = (0, 130, 20)
redUpper = (20, 255,255)

# frame config
frameHeight = 560
frameWidth = 840
frameY = 300
frameX = 1050

# paths
READY_SOUND_PATH = "audio/ready.mp3"

# controlling length of ball trail
BUFFER_MAX = 16
pts = deque(maxlen=BUFFER_MAX)


def play_sound_ready(path):
    playsound.playsound(path)
 
def startGame():
    
    global gameActuallyStarted
    
    # locate the play and resume button and click them to start/resume the game
    playButton = pyautogui.locateOnScreen('buttons/play.png', confidence=0.5)
    resumeButton = pyautogui.locateOnScreen('buttons/resume.png', confidence=0.5)
    
    # Mark x and y coordinate of center of the button to click
    if playButton:
        clickX = playButton.left + playButton.width // 2
        clickY = playButton.top + playButton.height // 2
    elif resumeButton:
        clickX = playButton.left + playButton.width // 2
        clickY = playButton.top + playButton.height // 2
    else:
        print(" [INFO] GAME SCREEN NOT FOUND! Exiting.. ")
        return 

    
    print(" [INFO] Starting game in 10 seconds!! ")
    t = Thread(target=play_sound_ready, args=(READY_SOUND_PATH,))
    t.deamon = True
    t.start()
    
    time.sleep(8.0)
    
    # perform 2 clicks, first to bring focus out of python window and other to actually 
    # click the play/resume button
    pyautogui.click(clickX, clickY)
    time.sleep(2.0)
    pyautogui.click(clickX, clickY)
    gameActuallyStarted = True
        
 
def main():
    # init flags
    faceDetected = False
    ballDetected = False
    gameStarted = False
    
    # initialize dlib's face detector (HOG-based)
    print("[INFO] loading face detector...")
    #face_detector = dlib.get_frontal_face_detector()
    face_cascade = cv2.CascadeClassifier('models/haarcascade_face.xml')
    
    # start the video stream thread
    print("[INFO] starting video stream thread...")
    vs = cv2.VideoCapture(0)
    time.sleep(1.0)
    
    # test image to set the window position
    test_frame = np.zeros(shape=(frameHeight, frameWidth, 3)).astype('uint8')
    cv2.imshow('Frame', test_frame)
    cv2.moveWindow('Frame', frameX, frameY)
    
    # loop over frames from the video stream
    while True:
                
        face_x = 0
        face_y = 0
        face_w = 0
        face_h = 0
        
        ball_x = 0
        ball_y = 0
        ball_radius = 0
    
        # grab the frame from the threaded video file stream
        frame = vs.read()
        frame = frame[1]
        
        # if we are viewing a video and we did not grab a frame,
        # then we have reached the end of the video
        if frame is None:
            break
        
        frame = imutils.resize(frame, width=frameWidth, height=frameHeight)
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        
        # visualize boundary lines first
        cv2.line(frame, (UP_LIMIT_START_X, UP_LIMIT_START_Y), 
                 (UP_LIMIT_END_X, UP_LIMIT_END_Y), (255, 0, 0), 2)
        cv2.line(frame, (DOWN_LIMIT_START_X, DOWN_LIMIT_START_Y), 
                 (DOWN_LIMIT_END_X, DOWN_LIMIT_END_Y), (255, 0, 0), 2)
        cv2.line(frame, (LEFT_LIMIT_START_X, LEFT_LIMIT_START_Y), 
                 (LEFT_LIMIT_END_X, LEFT_LIMIT_END_Y), (255, 0, 0), 2)
        cv2.line(frame, (RIGHT_LIMIT_START_X, RIGHT_LIMIT_START_Y), 
                 (RIGHT_LIMIT_END_X, RIGHT_LIMIT_END_Y), (255, 0, 0), 2)
        
        # detect faces in the grayscale frame
        #faces = face_detector(gray, 0)
        faces = face_cascade.detectMultiScale(gray, 1.1, 4)
        
        # convert dlib's rectangle to a OpenCV-style bounding box
        # [i.e., (x, y, w, h)], then draw the face bounding box
        # draw them on screen if you find it
        if len(faces) != 0:
            #(x, y, w, h) = face_utils.rect_to_bb(faces[0])
            (face_x, face_y, face_w, face_h) = faces[0]
            cv2.rectangle(frame, (face_x, face_y), (face_x + face_w, face_y + face_h), 
                          (0, 255, 0), 2)
            faceDetected = True 
        
        # if user wants to use football
        if USE_BALL:
        
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
                ((ball_x, ball_y), ball_radius) = cv2.minEnclosingCircle(c)
                M = cv2.moments(c)
                center = (int(M["m10"] / M["m00"]), int(M["m01"] / M["m00"]))
                
                # only proceed if the radius falls roughly within the size of a standard football
                if ball_radius > 10 and ball_radius < 80:
                    # draw the circle and centroid on the frame,
                    # then update the list of tracked points
                    #print("radius = ", radius)
                    cv2.circle(frame, (int(ball_x), int(ball_y)), int(ball_radius), (0, 255, 255), 2)
                    cv2.circle(frame, center, 5, (0, 0, 255), -1)
                    ballDetected = True
            
            # update the points queue
            pts.appendleft(center)
            
            # loop over the set of tracked points for the trail effect
            # decide later if the trail effect is actually needed. 
            for i in range(1, len(pts)):
                # if either of the tracked points are None, ignore them
                if pts[i - 1] is None or pts[i] is None:
                    continue
                # otherwise, compute the thickness of the line and
                # draw the connecting lines
                thickness = int(np.sqrt(BUFFER_MAX / float(i + 1)) * 2.5)
                cv2.line(frame, pts[i - 1], pts[i], (0, 0, 255), thickness)
                #print(pts[i-1], pts[i])
        
        
        if ((USE_BALL and ballDetected) or (not USE_BALL)) and faceDetected and not gameStarted:
            t = Thread(target=startGame, args=())
            t.deamon = True
            t.start()
            gameStarted = True
        
        # Add game logic here
        if gameActuallyStarted:
            
            global playerVerticalPosition
            global playerHorizontalPosition
                        
            # horizontal movements controlled by the football
            
            if USE_BALL:
                # player moves right from center
                if playerHorizontalPosition == 0 and ball_x < LEFT_LIMIT_START_X:
                    playerHorizontalPosition = 1 
                    pyautogui.press("right")
                    
                # player moves left from center
                if playerHorizontalPosition == 0 and ball_x > RIGHT_LIMIT_START_X:
                    playerHorizontalPosition = -1 
                    pyautogui.press("left")
                    
                # player comes to center from right
                if playerHorizontalPosition == 1 and ball_x > LEFT_LIMIT_START_X:
                    playerHorizontalPosition = 0 
                    pyautogui.press("left")
                    
                # player comes to center from left
                if playerHorizontalPosition == -1 and ball_x < RIGHT_LIMIT_START_X:
                    playerHorizontalPosition = 0 
                    pyautogui.press("right")
            else:
                # player moves right from center
                if playerHorizontalPosition == 0 and (face_x + face_w // 2) < LEFT_LIMIT_START_X:
                    playerHorizontalPosition = 1 
                    pyautogui.press("right")
                    
                # player moves left from center
                if playerHorizontalPosition == 0 and (face_x + face_w // 2) > RIGHT_LIMIT_START_X:
                    playerHorizontalPosition = -1 
                    pyautogui.press("left")
                    
                # player comes to center from right
                if playerHorizontalPosition == 1 and (face_x + face_w // 2) > LEFT_LIMIT_START_X:
                    playerHorizontalPosition = 0 
                    pyautogui.press("left")
                    
                # player comes to center from left
                if playerHorizontalPosition == -1 and (face_x + face_w // 2) < RIGHT_LIMIT_START_X:
                    playerHorizontalPosition = 0 
                    pyautogui.press("right")
                
            # vertical movements controlled by the face position
            
            # player jumps up from center
            if playerVerticalPosition == 0 and (face_y + face_h // 2) < UP_LIMIT_START_Y:
                playerVerticalPosition = 1
                pyautogui.press("up")
            
            # player return to center from jump
            if playerVerticalPosition == 1 and (face_y + face_h // 2) > UP_LIMIT_START_Y:
                playerVerticalPosition = 0
                
            # player squats from center
            if playerVerticalPosition == 0 and (face_y + face_h // 2) > DOWN_LIMIT_START_Y:
                playerVerticalPosition = -1
                pyautogui.press("down")
                
            # player returns to center from squat
            if playerVerticalPosition == -1 and (face_y + face_h // 2) > DOWN_LIMIT_START_Y:
                playerVerticalPosition = 0
            
        # show the frame
        cv2.imshow("Frame", frame)
        key = cv2.waitKey(1) & 0xFF
        
        # if the `q` key was pressed, break from the loop
        if key == ord("q") or key == ord("Q"):
            break
        
    # cleanup
    vs.release()
    cv2.destroyAllWindows()
    
if __name__ == '__main__':
    time.sleep(5)
    main()
