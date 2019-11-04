# This is the  file that will get called

# XXX: this will get info from rccortex.py
# cerebellum controlls movement and coordination
# this will communicate with the arduino that will either contorl the rc or car itself
# source: https://pythonforundergradengineers.com/python-arduino-LED.html
# XXX: NEED TO TAKE BETTER VIDEOS
#  lanes must be centered, on a smooth background

import cv2
import numpy as np
import rccortex
import serial   # serial imported for serial communication
import time     # required to use delay functions
from tesla import *    # the nn
import matplotlib.pyplot as plt


# state will be the cars current state (up, stop, left, right)
def handle_stop(state):
    print("IN handle_stop")

    cv2.imshow("Result", state)
    cv2.waitKey(0)
    if cv2.waitKey(1) == ord('q'): # waits 1 millisecond between frames (if 0, then video will freeze)
        cv2.destroyAllWindows()


def handle_traffic(state):
    print("IN handle_traffic")

    cv2.imshow("Result", state)
    cv2.waitKey(0)
    if cv2.waitKey(1) == ord('q'): # waits 1 millisecond between frames (if 0, then video will freeze)
        cv2.destroyAllWindows()


foo = {'stop sign':handle_stop,
        'traffic light': handle_traffic}

###############################################################################
def drive_forward():
    print("Starting drive_forwards() ... ")
    # NEEDS TO BE CONNECTED TO ARDUINO HERE I GUESS
    Arduino = serial.Serial('com18', 9600)
    time.sleep(2)

def drive_left(tesla):
    pass
# might merge these two
def drive_right(tesla):
    pass
###############################################################################
def update_direction(tesla, path):
    dirx, diry = tesla.get_direction()
    print("Direction=", dirx, diry)

    left = path[0]
    right = path[1]
    # will have to do some kind of math here
    # x = Acos(O)
    # y = Asin(0)

    # XXX: NEED TO DETECT A TURN!
    # if (turn):
    #     drive_newdirection()
    # else:
    #     drive_forward()

    newdirx, newdiry = dirx, diry

    tesla.set_direction(newdirx, newdiry)

def get_path(averaged_lines):
    middle_line = []
    try:
        middle_line = rccortex.get_middle_line(averaged_lines)
    except Exception as exc:
        print("Error in get_path():", exc)
    return middle_line

def analyze_view(frame):
    cannyimg = rccortex.canny(frame)
    print("two")
    cropped_image = rccortex.region_of_interest(cannyimg)
    print("three")
    lines = cv2.HoughLinesP(cropped_image, 1, np.pi/180,
            100, np.array([]), minLineLength=50, maxLineGap=1)
    # lines = cv2.HoughLinesP(cropped_image, 2, np.pi/180,
    #     100, np.array([]), minLineLength=100, maxLineGap=5)
    print("four")
    averaged_lines = rccortex.average_slope_intercept(frame, lines)

    # return cropped_image
    return averaged_lines

def detect_lane_from_video(video, tesla, detect=False):

    cap = cv2.VideoCapture("../../videos/"+video)

    while (cap.isOpened()):
        _, frame = cap.read()

        # print("one")
        # if (detect):
        #     rccortex.detect_objects(cap, video)
        #     break

        # previous = [np.zeros(4, dtype=int), np.zeros(4, dtype=int)]
        # temp = [np.zeros(4, dtype=int), np.zeros(4, dtype=int)]
        averaged_lines = analyze_view(frame)

        # print("five")
        path = get_path(averaged_lines)
        # print("six")
        #################### major key ################################
        update_direction(tesla, path)
        ##################################################################
        # line_image = rccortex.display_lines(frame, path, (0, 255, 0))
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
    video = "croppedcustom8.mp4"
    # video = "croppedcustom10.mp4"
    # video = "croppedcustom11.mp4"
    try:
        detect_lane_from_video(video, tesla)
        # detect_lane_from_video(video, tesla, True)
    except Exception as exc:
        print("ERROR:,", exc)
        cv2.destroyAllWindows()
    finally:
        print("Done")

if __name__ == '__main__':
    tesla = Tesla()
    main(tesla)
