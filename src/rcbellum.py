# This is the  file that will get called

# source: https://pythonforundergradengineers.com/python-arduino-LED.html

import cv2
import numpy as np
import rccortex
import serial           # serial imported for serial communication
import time             # required to use delay functions
from tesla import *     # the nn
import matplotlib.pyplot as plt

import sys, os # to get exception line number

dir_dict = \
  { "FORWARD"   : ( 0, 1),  # straight forward
    "BACKWARD"  : (-1, 0), # straight back
    "LEFT"      : (-1, 1),  # forward left
    "RIGHT"     : ( 1, 1)}  # forward right

def update_direction(tesla, path):
    dirx, diry = tesla.get_direction()
    print(tesla)

    newdirx, newdiry = dir_dict[path]

    tesla.set_direction(newdirx, newdiry)
    print(tesla)

def get_path(averaged_lines):
    middle_line = []
    try:
        middle_line = rccortex.get_middle_line(averaged_lines)
    except Exception as exc:
        print("Error in rcbellum.get_path():", exc)
    return middle_line

def analyze_view(frame):
    cannyimg = rccortex.canny(frame)
    # print("two")
    cropped_image = rccortex.region_of_interest(cannyimg)
    # return cropped_image
    # print("three")
    # lines = cv2.HoughLinesP(cropped_image, 1, np.pi/180, 100,
    #     np.array([]), minLineLength=5, maxLineGap=5)
    lines = cv2.HoughLinesP(cropped_image, 2, np.pi/180, 100,
            np.array([]), minLineLength=10, maxLineGap=10) # works with one middle line
    # lines = cv2.HoughLinesP(cropped_image, 2, np.pi/180,
    #         100, np.array([]), minLineLength=50, maxLineGap=50)
    # lines = cv2.HoughLinesP(cropped_image, 2, np.pi/180,
    #     100, np.array([]), minLineLength=100, maxLineGap=100)

    # print("LINES:", lines)
    averaged_lines, path = rccortex.average_slope_intercept(frame, lines)
    # averaged_lines = rccortex.average_slope_intercept(frame, np.array(lines[0], lines[1]))
    return averaged_lines, path

#######################################################################
def detect_lane_from_video(video, tesla, detect=False):

    # previous = np.array([np.array(np.zeros(4, int)) , np.array(np.zeros(4, int))])

    cap = cv2.VideoCapture("../../videos/"+video)

    while (cap.isOpened()):
        _, frame = cap.read()
        # print("one")
        # if (detect):
        #     rccortex.detect_objects(cap, video)
        #     break
        # previous = [np.zeros(4, dtype=int), np.zeros(4, dtype=int)]
        # temp = [np.zeros(4, dtype=int), np.zeros(4, dtype=int)]
        averaged_lines, path = analyze_view(frame)
        # print("averaged_lines:", averaged_lines)
        #################### major key ################################
        # print("five")
        # path = get_path(averaged_lines)
        update_direction(tesla, path)
        ##################################################################
        # line_image = rccortex.display_lines(frame, path, (0, 255, 0))
        # print("six")
        line_image = rccortex.display_lines(frame, averaged_lines, (0, 255, 0))
        # print("seven")
        combo_img = cv2.addWeighted(frame, 0.8, line_image, 1, 1) # gamma value at end
        # print("eight")

        cv2.imshow("Ampeater View", combo_img)
        # plt.imshow(combo_img)
        # plt.show()

        if cv2.waitKey(1) == ord('q'): # waits 1 millisecond between frames (if 0, then video will freeze)
            break

    cap.release()
    cv2.destroyAllWindows()

def detect_lane_from_image(image, tesla):
    # previous = np.array([np.array(np.zeros(4, int)) , np.array(np.zeros(4, int))])

    img = cv2.imread("../../pics/"+image, 1)

    averaged_lines = analyze_view(img)
    # print("one")
    line_image = rccortex.display_lines(img, averaged_lines, (0, 255, 0))
    # print("two")
    combo_img = cv2.addWeighted(img, 0.8, line_image, 1, 1) # gamma value at end
    # print("three")
    cv2.imshow("Ampeater View", combo_img)
    # plt.imshow(img)
    # plt.show()
    if cv2.waitKey(0) == ord('q'): # waits 1 millisecond between frames (if 0, then video will freeze)
        cv2.destroyAllWindows()

###############################################################################
def init_arduino():
    ser1 = serial.Serial('COM5', 9600)  # check com number

    # while True:
    # for i in range(30):
    ser1.write('H'.encode())           # turns on the led
    ser1.write('L'.encode())           # turns off the led

###############################################################################

def main(tesla):
    print("Starting rcbellum.py ...")
    # video = "video1.mp4"
    # video = "video2.mp4"  # DONT USE THIS ONE
    # video = "croppedvideo3.mp4"
    # video = "video4.mp4"
    # video = "video5.mp4"
    # video = "custom2.mp4"
    # video = "custom3.p4"
    # video = "custom4.mp4" #DONT USE THIS ONE
    # video = "custom5.mp4"
    # video = "custom6.mp4"
    # video = "custom7.mp4"
    # video = "croppedcustom7.mp4"
    # video = "croppedcustom8.mp4"
    # video = "croppedcustom10.mp4"
    video = "croppedcustom11.mp4"
    # video = "croppedcustom12.mp4"

    # img = "lanes1.jpg"
    # img = "lanes2.jpg"
    # img = "lanes3.jpg"
    img = "test1.jpg" #from croppedcustom11

    try:
        # detect_lane_from_image(img, tesla)
        detect_lane_from_video(video, tesla)
        # detect_lane_from_video(video, tesla, True)
    except Exception as exc:
        exc_type, exc_obj, exc_tb = sys.exc_info()
        fname = os.path.split(exc_tb.tb_frame.f_code.co_filename)[1]
        print("Error at rcbellum.main():", exc)
        print(exc_type, fname, exc_tb.tb_lineno)
        cv2.destroyAllWindows()
    finally:
        print("Goodbye, Thank You!")

if __name__ == '__main__':
    tesla = Tesla()
    # init_arduino()
    main(tesla)
