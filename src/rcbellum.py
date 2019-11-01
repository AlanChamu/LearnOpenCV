# This will probably be the main file that gets called

# XXX: this will get info from rccortex.py
# cerebellum controlls movement and coordination
# this will communicate with the arduino that will either contorl the rc or car itself

# source: https://pythonforundergradengineers.com/python-arduino-LED.html

import cv2
import rccortex
import serial   # serial imported for serial communication
import time     # required to use delay functions
from tesla import *    # the nn

def move_forwards():
    print("Starting move_forwards() ... ")

    # NEEDS TO BE CONNECTED TO ARDUINO HERE I GUESS
    Arduino = serial.Serial('com18', 9600)

    time.sleep(2)

def main(tesla):
    print("Starting rcbellum.py ...")
    # video = "video1.mp4"
    # video = "custom1.mp4"  # not good
    # video = "custom2.mp4"
    video = "custom3.mp4"
    try:
        print(tesla)
        rccortex.detect_lane_from_video(video)
    except Exception as exc:
        print("Noooo,", exc)
        # cap.release()
        cv2.destroyAllWindows()
    finally:
        print("Done")

if __name__ == '__main__':
    x = input("Quit? ")
    if x in ["q", "quit", "Quit", "yes"]:
        quit()
    else:
        tesla = Tesla()
        main(tesla)
