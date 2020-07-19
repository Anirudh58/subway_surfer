from PIL import ImageGrab
import os
import time
import pyautogui
import cv2
 
def screenGrab():
    # take screenshot using pyautogui 
    image = pyautogui.screenshot() 
       
    # since the pyautogui takes as a  
    # PIL(pillow) and in RGB we need to  
    # convert it to numpy array and BGR  
    # so we can write it to the disk 
    image = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR) 
       
    # writing it to the disk using opencv 
    cv2.imwrite(os.getcwd() + '\\full_snap__' + str(int(time.time())) + '.png', image) 
    
    #box = ()
    #im = ImageGrab.grab()
    #im.save(os.getcwd() + '\\full_snap__' + str(int(time.time())) + '.png', 'PNG')
 
def main():
    screenGrab()
 
if __name__ == '__main__':
    main()