# This will probably be the main file that gets called

# XXX: this will get info from rccortex.py
# cerebellum controlls movement and coordination
# this will communicate with the arduino that will either contorl the rc or car itself

# source: https://pythonforundergradengineers.com/python-arduino-LED.html

import rccortex
import serial   # serial imported for serial communication
import time     # required to use delay functions
from tesla import *    # the nn

def move_forwards():
    print("Starting move_forwards() ... ")

    # NEEDS TO BE CONNECTED TO ARDUINO HERE I GUESS
    Arduino = serial.Serial('com18', 9600)

    time.sleep(2)

def main():
    print("Starting rcbellum main()")

    tesla = Tesla()

    # should pass in something, maybe a Tesla object ???
    rccortex.main(tesla)



if __name__ == '__main__':
    x = input("Quit? ")
    if x in ["q", "quit", "Quit", "yes"]:
        quit()
    else:
        main()
