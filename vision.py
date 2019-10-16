import numpy as np
import matplotlib
import cv2

# from https://techtutorialsx.com/2018/06/02/python-opencv-converting-an-image-to-gray-scale/ 
def grey_image():
    #print(cv2.__version__)
    image = cv2.imread("drivingpov.jpg")
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    cv2.imshow('Original image', image)
    cv2.imshow('Gray image', gray)

    cv2.waitKey(0)
    cv2.destroyAllWindows()

def grey_video():
    print("Making a grey video")

    capture = cv2.VideoCapture('test2.mp4')

    while True:
        ret, frame = capture.read()
 
        grayFrame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
     
        cv2.imshow('video gray', grayFrame)
        cv2.imshow('video original', frame)
          
        if cv2.waitKey(1) == 27:
            break
  
    capture.release()
    cv2.destroyAllWindows()


if __name__ == '__main__':
    print("Ampeaters")
    while True:
        x = input("Quit? ")
        if x in ["q", "quit", "Quit", "yes"]:
            quit()
        else:
            #grey_image()
            grey_video() # press 'esc' to quit the video


    print("Done")


'''
from https://pypi.org/project/opencv-python/ 

a. Packages for standard desktop environments (Windows, macOS, almost any GNU/Linux distribution)

run pip install opencv-python if you need only main modules
run pip install opencv-contrib-python if you need both main and
    contrib modules (check extra modules listing from OpenCV documentation)

'''
