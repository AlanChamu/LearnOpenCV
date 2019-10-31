# yolov3 https://pjreddie.com/darknet/yolo/
# source: https://towardsdatascience.com/object-detection-with-less-than-10-lines-of-code-using-python-2d28eebc5b11

# XXX: this code is too slow!
# XXX: need to make faster
# figure out how to crop the image/video (1 k by 1k)

import cv2
import numpy as np

# to detect lanes
import lanedetection
import matplotlib.pyplot as plt
import tesla

# for the nn
# import tesla

# for object detection
# import cvlib as cv
# from cvlib.object_detection import draw_bbox
# https://pjreddie.com/darknet/yolo/

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

def detect_lane_from_video(video):

    cap = cv2.VideoCapture("../../videos/"+video)

    while (cap.isOpened()):
        _, frame = cap.read()
        print("one")
        cannyimg = lanedetection.canny(frame)
        print("two")
        cropped_image = lanedetection.region_of_interest(cannyimg)
        print("three")
        lines = cv2.HoughLinesP(cropped_image, 2, np.pi/180,
            100, np.array([]), minLineLength=40, maxLineGap=5)
        print("four")
        averaged_lines = lanedetection.average_slope_intercept(frame, lines)
        print("five")
        # may have to return an angle value to allow car to turn directions
        middle_line = lanedetection.get_middle_line(averaged_lines)
        print("six")
        line_image = lanedetection.display_lines(frame, middle_line, (255, 0, 0))
        # line_image = lanedetection.display_lines(frame, averaged_lines, (255, 0, 0))
        print("seven")
        combo_img = cv2.addWeighted(frame, 0.8, line_image, 1, 1) # gamma value at end
        print("eight")

        cv2.imshow("Result", combo_img)
        # plt.imshow(combo_img)
        # plt.show()

        if cv2.waitKey(1) == ord('q'): # waits 1 millisecond between frames (if 0, then video will freeze)
            break

    cap.release()
    cv2.destroyAllWindows()

def main(tesla):
    print("Starting rcbellum.py ...")
    video = "video1.mp4"
    # video = "custom1.mp4"  # not good
    # video = "custom2.mp4"
    # video = "custom3.mp4"
    try:
        print(tesla)
        detect_lane_from_video(video)
    except Exception as exc:
        print("Noooo,", exc)
        # cap.release()
        cv2.destroyAllWindows()
    finally:
        print("Done")
