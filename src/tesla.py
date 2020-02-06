'''
This will contain the actual neural network,
This will teach itself how to drive ...
    what does this mean?
    This means a couple things
1. we WONT HARD CODE any rules into the car, it will learn the rules with our help (supervised learning)
2.
'''

import cv2      # for NN
import serial   # just in case
import time     # t0 test out communication b/w pi and uno

# LEARN HOW TO MAKE A NEURAL NETWORK AND TRAIN IT

dir_dict = \
  { "FORWARD"   : ( 0, 1, 1),  # straight forward
    "BACKWARD"  : (-1, 0, 2), # straight back
    "LEFT"      : (-1, 1, 3),  # forward left
    "RIGHT"     : ( 1, 1, 4)}  # forward right

class Tesla(object):
    """docstring for Tesla."""

    def __init__(self):
        super(Tesla, self).__init__()
        # self.arg = arg
        self.posx = 0
        self.posy = 0
        # kind of like x, y coordinates
        # x  = 0 STRAIGHT, -2 -1 left, 1 2 right
        # y  = 0 STOP -2 -1 reverse, 1 2 forwards,
        self.current_direction = (0, 0)

    # return as a string
    def __str__(self):
        return 'Dir='+str(self.current_direction)

    def get_pos(self):
        return (self.posx, self.posy)

    def get_direction(self):
er of seconds execution to be su        return self.current_direction

    def set_pos(self, new_x, new_y):
        self.posx = new_x
        self.posy = new_y

    def set_direction(self, newx, newy):
        self.current_direction = newx, newy
        # send instructions to arduino

    # CAN I MAKE A FUNCTION THAT CAN DETECT TURNS?
    # yes

    def __nn__(self):
        pass
        # for future alan, the one who knows how to make a nn and train it

# from rcbellum.py
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

def eight_figure(tesla, uno, command):
    pass

def circles(tesla, uno, command):
    print("Starting circles() ... ")
    
    go = 1 
    back = 2
    left = 3
    right = 4
    
    if uno is not None:
        while True:
            uno.write(go.encode())
            uno.write(right.encode())
            sleep(2)
            uno.write(back.encode())
            uno.write(left.encode())
            sleep(1)

def forwards_and_back(tesla, uno, command):
    print("Starting forwards_and_back() ... ")
    
    go = 1 
    back = 2
    
    if uno is not None:
        while True:
            uno.write(go.encode())
            sleep(2)
            uno.write(back.encode())
            sleep(2)
    
def test_run(tesla, uno, command):
    print("Starting Tesla.test_run() ... ")

    forwards_and_back(tesla, uno, command)
    #circles(tesla, uno, command)
    #eight_figure(tesla, uno, command)
