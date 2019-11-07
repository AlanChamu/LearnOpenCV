
# script to test if we can control the rc with our keyboard

# from ../src import rcbellum
import sys
sys.path.append('../src/')
import rcbellum
import tesla

import pygame   # to control the rc car with a keyboard
from pygame.locals import *


def main():
    print("Starting driverctest.py")
    t = tesla.Tesla()
    rcbellum.main(t)

if __name__ == '__main__':
    main()
