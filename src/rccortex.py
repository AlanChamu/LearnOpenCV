# yolov3 https://pjreddie.com/darknet/yolo/
# source: https://towardsdatascience.com/object-detection-with-less-than-10-lines-of-code-using-python-2d28eebc5b11

# XXX: this code is too slow!
# XXX: need to make faster
# figure out how to crop the image/video (1 k by 1k)

import numpy as np
import cv2
import matplotlib.pyplot as plt
import cvlib as cv
from cvlib.object_detection import draw_bbox

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

####################### HELPER FUNCTIONS ###################################
def canny(image): # makes image/video look cool (just black and white)
    #create a greyscale image
    gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY) #flag to turn gray
    #GaussianBlur to reduce noise
    blur = cv2.GaussianBlur(gray, (5, 5), 0) # to reduce noise
    # Step 3: detect lanes, figure out what Canny method does
    result = cv2.Canny(blur, 50, 150)
    return result

def analyze_view():

    # vid = cv2.VideoCapture("../videos/video2.mp4")
    vid = cv2.VideoCapture("../../videos/video1.mp4")

    while (vid.isOpened()):
        _, frame = vid.read()
        bbox, label, conf = cv.detect_common_objects(frame)
        output_img = draw_bbox(frame, bbox, label, conf)
        cannyimg = canny(frame)
        cropped_image = region_of_interest(cannyimg)

        cv2.imshow("Result", cropped_image)
        if cv2.waitKey(1) == ord('q'): # waits 1 millisecond between frames (if 0, then video will freeze)
            break

    vid.release()
    cv2.destroyAllWindows()

    # image = cv2.imread('stop.jpg')
    # image = cv2.imread('greentrafficlight.jpg')
    # image = cv2.imread('redtrafficlight.jpg')
'''
    bbox, label, conf = cv.detect_common_objects(image)
    output_img = draw_bbox(image, bbox, label, conf)

    print("Done Detecting ... ")
    print(bbox, label, conf)

    foo[label[0]](output_img)
'''
    # plt.imshow(output_img)
    # plt.show()

    # cv2.imshow("Result", output_img)
    # cv2.waitKey(0)
    # if cv2.waitKey(1) == ord('q'): # waits 1 millisecond between frames (if 0, then video will freeze)
    #     cv2.destroyAllWindows()

# if __name__ == '__main__':
#     analyze_view()


# https://pjreddie.com/darknet/yolo/
