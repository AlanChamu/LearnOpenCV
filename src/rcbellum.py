# This will probably be the main file that gets called

# XXX: this will get info from rccortex.py
# cerebellum controlls movement and coordination
# this will communicate with the arduino that will either contorl the rc or car itself

# source: https://pythonforundergradengineers.com/python-arduino-LED.html

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
def detect_lane_from_video(video, tesla):

    cap = cv2.VideoCapture("../../videos/"+video)

    while (cap.isOpened()):
        _, frame = cap.read()
        # print("one")
        previous = []
        temp = []
        cannyimg = rccortex.canny(frame)
        # print("two")
        cropped_image = rccortex.region_of_interest(cannyimg)
        # print("three")
        # CHANGED maxLineGap=5 TO maxLineGap=10 AND THE VIDEO DIDNT CRASH!
        lines = cv2.HoughLinesP(cropped_image, 2, np.pi/180,
            100, np.array([]), minLineLength=40, maxLineGap=10)

        print("four")

        averaged_lines = rccortex.average_slope_intercept(frame, lines)
        previous = averaged_lines

        if averaged_lines is None:
            averaged_lines = previous

        print("five")
        middle_line = rccortex.get_middle_line(averaged_lines)
        temp = middle_line

        if middle_line is None:
            middle_line = temp

        print("six")
        #################### major key ################################
        dirx, diry = tesla.get_direction()
        print("Direction=", dirx, diry)

        # something with middle_line
        left = middle_line[0]
        right = middle_line[1]
        print(left, right)
        newdirx, newdiry = dirx, diry

        tesla.set_direction(newdirx, newdiry)

        # XXX: NEED TO DETECT A TURN!

        # if (turn):
        #     drive_newdirection()
        # else:
        #     drive_forward()

        ##################################################################
        # line_image = rccortex.display_lines(frame, middle_line, (0, 255, 0))
        line_image = rccortex.display_lines(frame, averaged_lines, (0, 255, 0))
        print("seven")
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
    # video = "custom2.mp4"
    # video = "custom3.mp4"
    # video = "custom4.mp4"
    # video = "custom5.mp4"
    # video = "custom6.mp4"
    # video = "custom7.mp4"
    video = "croppedcustom8.mp4"
    # video = "custom8.mp4"
    try:
        print(tesla)
        detect_lane_from_video(video, tesla)
    except Exception as exc:
        print("Noooo,", exc)
        cv2.destroyAllWindows()
    finally:
        print("Done")

if __name__ == '__main__':
    tesla = Tesla()
    main(tesla)
