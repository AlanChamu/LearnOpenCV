# This will probably be the main file that gets called

# XXX: this will get info from rccortex.py
# cerebellum controlls movement and coordination
# this will communicate with the arduino that will either contorl the rc or car itself

# source: https://pythonforundergradengineers.com/python-arduino-LED.html

import os
import time
import serial
# from rccortex import*
# import rccortex
import lanedetection
# import tesla

def main():
    print("Starting rcbellum.py ...")
    video = "video1.mp4"
    # video = "custom1.mp4"
    lanedetection.detect_lane_from_video(video)

if __name__ == '__main__':
    while True:
        x = input("Quit? ")
        if x in ["q", "quit", "Quit", "yes"]:
            quit()
        else:
            main()
