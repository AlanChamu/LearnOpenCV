# This is the  file that will get called

# source: https://pythonforundergradengineers.com/python-arduino-LED.html


from picamera.array import PiRGBArray
from picamera import PiCamera
import time
import cv2
import numpy as np
import rccortex
#import serial           # serial imported for serial communication
import time             # required to use delay functions
from tesla import *     # the nn
#import matplotlib.pyplot as plt
import sys, os # to get exception line number

###############################################################################

dir_dict = \
  { "FORWARD"   : ( 0, 1, 1),  # straight forward
    "BACKWARD"  : (-1, 0, 2), # straight back
    "LEFT"      : (-1, 1, 3),  # forward left
    "RIGHT"     : ( 1, 1, 4)}  # forward right

def init_arduino(connected):
    uno = None

    if (connected):
        uno = serial.Serial('COM5', 9600)  # check com number

    # uno.write('H'.encode())            # turns on the led
    # uno.write('L'.encode())          # turns off the led

    return uno

def update_direction(tesla, path, uno=None):
    dirx, diry = tesla.get_direction()
    print(tesla)

    newdirx, newdiry, arduino_instruction = dir_dict[path]
    ################################################################
    # tesla object doesnt really need to know the arduino instruction, it would be nice tho
    # send turn instruction to arduino
    if (uno is not None):
        uno.write(arduino_instruction.encode())
    ################################################################
    tesla.set_direction(newdirx, newdiry)
    print(tesla)

###############################################################################
def get_path(averaged_lines):   # NOT USED
    middle_line = []
    try:
        middle_line = rccortex.get_middle_line(averaged_lines)
    except Exception as exc:
        print("Error in rcbellum.get_path():", exc)
    return middle_line

def analyze_view(frame):        # VITAL
    cannyimg = rccortex.canny(frame)

    cropped_image = rccortex.region_of_interest(cannyimg)
    # return cropped_image, None
    # for video1.mp4
    lines = cv2.HoughLinesP(cropped_image, 2, np.pi/180, 100,
            np.array([]), minLineLength=100, maxLineGap=5) # works with one middle line
    # lines = cv2.HoughLinesP(cropped_image, 2, np.pi/180, 100,
    #         np.array([]), minLineLength=10, maxLineGap=10) # works with one middle line

    averaged_lines, path = rccortex.average_slope_intercept(frame, lines)
    # path is which direction to go, as a str
    return averaged_lines, path

#######################################################################
def detect_lane_from_video(tesla, uno=None, detect=False):
    camera = PiCamera();
    camera.resolution = (640,480)
    camera.framerate = 32
    rawCapture = PiRGBArray(camera, size=(640,480))
    time.sleep(0.1)

    for bigframe in camera.capture_continuous(rawCapture, format="bgr", use_video_port=True):
        frame = bigframe.array
#       # _, frame = cap.read()
#        # if (detect):
#        #     rccortex.detect_objects(cap, video)
#        #     break
#        averaged_lines, path = analyze_view(frame)
#        # cv2.imshow("Ampeater View/", averaged_lines)
#        # plt.imshow(averaged_lines)
#        # plt.show()
#
#        # print("HELLO")
#        #################### major key ################################
#        update_direction(tesla, path, uno)
#        ##################################################################
#        line_image = rccortex.display_lines(frame, averaged_lines, (0, 255, 0))
#
#        combo_img = cv2.addWeighted(frame, 0.8, line_image, 1, 1) # gamma value at end

#        cv2.imshow("Ampeater View", combo_img)
        cv2.imshow("Ampeater View", frame)

        #plt.imshow(combo_img)
        # plt.show()

        rawCapture.truncate(0)

        if cv2.waitKey(1) == ord('q'): # waits 1 millisecond between frames (if 0, then video will freeze)
            break
    #cap.release()
    cv2.destroyAllWindows()

def main(tesla, uno):
    print("Starting rcbellum.py ...")
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
    # video = "croppedcustom8.mp4"  # BAD VIDEO, NEEDS A CLEAR BACKGROUND
    # video = "croppedcustom10.mp4"
    # video = "croppedcustom11.mp4"
    # video = "croppedcustom12.mp4"

    # img = "lanes1.jpg"
    try:
        # detect_lane_from_image(img, tesla)
        detect_lane_from_video(tesla, uno)
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
    tesla = Tesla() # would like to add serial object into Tesla

    connected = False
    uno = init_arduino(connected)

    main(tesla, uno)

# USED FOR DEBUGGING
# def detect_lane_from_image(image, tesla, uno):
#
#     img = cv2.imread("../../pics/"+image, 1)
#
#     averaged_lines, path = analyze_view(img)
#     #  path is which direction to go, as a str
#
#     line_image = rccortex.display_lines(img, averaged_lines, (0, 255, 0))
#
#     combo_img = cv2.addWeighted(img, 0.8, line_image, 1, 1) # gamma value at end
#
#     cv2.imshow("Ampeater View", combo_img)
#     # plt.imshow(img)
#     # plt.show()
#     if cv2.waitKey(0) == ord('q'): # waits 1 millisecond between frames (if 0, then video will freeze)
#         cv2.destroyAllWindows()
