# This is the  file that will get called

# source: https://pythonforundergradengineers.com/python-arduino-LED.html

import sys
import keras
import cv2
import numpy as np
import rccortex
import serial           # serial imported for serial communication
import time             # required to use delay functions
from tesla import *     # the nn
import matplotlib.pyplot as plt

import sys, os # to get exception line number
###############################################################################
dir_dict = \
  { "FORWARD"   : ( 0, 1, 1),  # straight forward
    "BACKWARD"  : (-1, 0, 2),  # straight back
    "LEFT"      : (-1, 1, 3),  # forward left
    "RIGHT"     : ( 1, 1, 4)}  # forward right

# third variable is for sending sommands to the arudino
def init_arduino(connected):
    uno = None
    if (connected):
        uno = serial.Serial('COM5', 9600)  # check com number
    return uno

def update_direction(tesla, path, uno):
    print("In update_direction(),", path)
    dirx, diry = tesla.get_direction()
    newdirx, newdiry, arduino_instruction = dir_dict[path]
    ################################################################
    # tesla object doesnt really need to know the arduino instruction, it would be nice
    # send turn instruction to arduino, not sure if this works, does work for strings
    if (uno is not None):
        # ARDUINO INSTRUCTION IS AN INTEGRE FROM DIRECTION DICTIONARY
        uno.write(arduino_instruction.encode()) # this is it!
    ################################################################
    tesla.set_direction(newdirx, newdiry)

###############################################################################
def analyze_view(frame, command):        # VITAL
    cannyimg = rccortex.canny(frame)
    if (command == "canny"):
        cv2.imshow("Ampeater View", cannyimg)
        cv2.waitKey(5)

    cropped_image = rccortex.region_of_interest(cannyimg)
    if (command == "cropped"):
        cv2.imshow("Ampeater View", cropped_image)
        cv2.waitKey(0)

    ##############################  OBJECT DETECTION  ########################



    ##############################  MAJOR KEY  ###########################

    # # XXX: ended here
    # return cropped_image, None
    # for video1.mp4
    lines = cv2.HoughLinesP(cropped_image, 2, np.pi/180, 100,
            np.array([]), minLineLength=40, maxLineGap=5) # works with one middle line
    # lines = cv2.HoughLinesP(cropped_image, 2, np.pi/180, 100,
    #         np.array([]), minLineLength=10, maxLineGap=10) # works with one middle line

    ##############################  MAJOR KEY  ###########################
    # return lines, "FORWARD"
    # optimized lines
    averaged_lines, path = rccortex.average_slope_intercept(frame, lines)
    # path is which direction to go, as a str
    ##############################  MAJOR KEY  Above #####################
    return averaged_lines, path

#######################################################################
def detect_lane_from_video(video, tesla, uno=None, command=None, detect=False):

    cap = cv2.VideoCapture("../../videos/"+video)

    while (cap.isOpened()):
        _, frame = cap.read()
        if (command == "show"):
            plt.imshow(frame)
            plt.show()
        averaged_lines, path = analyze_view(frame, command)

        #################### MAJOR KEY ################################
        update_direction(tesla, path, uno)
        ##################################################################
        line_image = rccortex.display_lines(frame, averaged_lines, (255, 0, 0))
        combo_img = cv2.addWeighted(frame, 0.8, line_image, 1, 1) # gamma value at end
        cv2.imshow("Ampeater View", combo_img)
        if cv2.waitKey(1) == ord('q'): # waits 1 millisecond between frames (if 0, then video will freeze)
            break

    cap.release()
    cv2.destroyAllWindows()

def main(tesla, uno, command=None):
    print("Starting rcbellum.py ...")
    video = "video1.mp4"
    #video = "croppedcustom11.mp4"
    # video = "croppedcustom12.mp4"
    # img = "test1.jpg" #from croppedcustom11

    try:
        tesla.test_run(tesla, uno, command)
        # detect_lane_from_video(video, tesla, uno, command)
    except Exception as exc:
        exc_type, exc_obj, exc_tb = sys.exc_info()
        fname = os.path.split(exc_tb.tb_frame.f_code.co_filename)[1]
        print("Error at rcbellum.main():", exc)
        print(exc_type, fname, exc_tb.tb_lineno)
        cv2.destroyAllWindows()
    finally:
        print("Goodbye, Thank You!")

if __name__ == '__main__':
    command = sys.argv[1]

    tesla = Tesla() # would like to add serial object into Tesla

    connected = False
    uno = init_arduino(connected)

    main(tesla, uno, command)


# for debugging
# def detect_lane_from_image(image, tesla, uno):
#
#     img = cv2.imread("../../pics/"+image, 1)
#     averaged_lines, path = analyze_view(img)
#     #  path is which direction to go, as a str
#     line_image = rccortex.display_lines(img, averaged_lines, (0, 255, 0))
#
#     combo_img = cv2.addWeighted(img, 0.8, line_image, 1, 1) # gamma value at end
#
#     cv2.imshow("Ampeater View", combo_img)
#     # plt.imshow(img)
#     # plt.show()
#     if cv2.waitKey(0) == ord('q'): # waits 1 millisecond between frames (if 0, then video will freeze)
#         cv2.destroyAllWindows()
# def get_path(averaged_lines):   # NOT USED
#     middle_line = []
#     try:
#         middle_line = rccortex.get_middle_line(averaged_lines)
#     except Exception as exc:
#         print("Error in rcbellum.get_path():", exc)
#     return middle_line
