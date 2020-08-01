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
    startGame()
    
if __name__ == '__main__':
    main()
