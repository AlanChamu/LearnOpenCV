# yolov3 https://pjreddie.com/darknet/yolo/
# source: https://towardsdatascience.com/object-detection-with-less-than-10-lines-of-code-using-python-2d28eebc5b11
#source: https://www.youtube.com/watch?v=eLTLtUVuuy4

import cv2
import numpy as np
import matplotlib.pyplot as plt

# for the nn
# import tesla

# for object detection
import cvlib as cv
from cvlib.object_detection import draw_bbox
# https://pjreddie.com/darknet/yolo/

def make_coordinates(image, line_parameters):
    # print("In make_coordinates,", image, line_parameters)
    try:
        slope, intercept = line_parameters
        y1 = image.shape[0]
        # print("y1,", y1)
        # why do we have to convert to int?
        y2 = int(y1*(3/5))
        x1 = int((y1 - intercept)/slope)
        x2 = int((y2 - intercept)/slope)
        # print("array,", [x1, y1, x2, y2])
        return np.array([x1, y1, x2, y2])
    except Exception as exc:
        print("Oops,",exc)
        # return None
        return np.array([0, 0, 0, 0])

def average_slope_intercept(image, lines):
    print("In average_slope_intercept ... ")
    left_fit = []
    right_fit = []

    if lines is None:
        return None

    for line in lines:
        x1, y1, x2, y2 = line.reshape(4)

        parameters = np.polyfit((x1, x2), (y1, y2), 1)

        slope = parameters[0]
        intercept = parameters[1]

        if slope <= 0:
            left_fit.append((slope, intercept))
        else:
            right_fit.append((slope, intercept))

    print("here?")
    left_fit_average = np.average(left_fit, axis=0)
    print("or here?")
    right_fit_average = np.average(right_fit, axis=0)

    left_line = make_coordinates(image, left_fit_average)
    # print("break 1")
    right_line = make_coordinates(image, right_fit_average)
    # print("break 2")

    # HAVE TO MAKE COMPATIBLE IF make_coordinates RETURNS A None
    # BETTER YET, WHY IS THERE AN ISSUE IN make_coordinates?
    print("out of average_slope_intercept ... ")
    return np.array([left_line, right_line])

def canny(image):
    #create a greyscale image
    gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY) #flag to turn gray
    #GaussianBlur to REDUCE NOISE -> FOR MORE ACCURATE EDGE DETECTION
    blur = cv2.GaussianBlur(gray, (5, 5), 0) # to reduce noise
    # Step 3: detect lanes, figure out what Canny method does
    # canny applies a 5by5 GaussianBlur either way
    result = cv2.Canny(blur, 50, 150) # low to high threshold
    return result

# focuses HoughLinesP into a certain area in the mp4 file,
# results in a smoother more accurrate line
def region_of_interest(image):
    height = image.shape[0]

    # for test mp4 file from tutorial
    polygons = np.array([
    [(0, height), (1000, height), (600, 100)]
    ])

    mask = np.zeros_like(image)
    cv2.fillPoly(mask, polygons, 255)
    masked_image = cv2.bitwise_and(image, mask)
    return masked_image

#now to make just the polygon display in front of the lanes

# now display lines in front of lanes
def display_lines(image, lines, color): # color is a three int tuple
    line_image = np.zeros_like(image)
    if lines is not None:
        for x1, y1, x2, y2 in lines:
            # last equals line thickness, 2nd to last is color BGR
            cv2.line(line_image, (x1, y1), (x2, y2), color, 10)
            # specify where you want it to be drawn, and color
    return line_image

def get_middle_line(lines):
    print("in get_middle_line,", lines)

    top = []
    bottom = []

    if lines is None:
        return None

    print(lines[0])
    print(lines[1])

    left = lines[0]
    right = lines[1]

    top_diff = (right[2] - left[2])/2
    bottom_diff = (right[0] - left[0])/2

    temp2 = int(top_diff) + left[2]
    temp1 = temp2
    # temp1 = int(bottom_diff) + left[0]  # will use this to get angle

    print("top_diff=", top_diff)
    print("bottom_diff=", bottom_diff)

    top = np.array([temp1, 720, temp2, 432])
    bottom = np.array([temp1, 720, temp2, 432])

    print("out of get_middle_line")
    return np.array([top,bottom])
    #  have to return something like [[(x1, y1),(x2, y2)],
    #                                   [(x1, y1), (x2, y2)]]

def detect_objects(cap, video):

    frames = cv.get_frames("../../videos/"+video)

    for frame in frames:
        bbox, label, conf = cv.detect_common_objects(frame,
            confidence=0.25, model='yolov3-tiny')

        out = draw_bbox(frame, bbox, label, conf)

        cv2.imshow("Object Detection", out)

        if cv2.waitKey(1) == ord('q'): # waits 1 millisecond between frames (if 0, then video will freeze)
            break

    cv2.destroyAllWindows()


    # while (cap.isOpened()):
    #     _, frame = cap.read()
    #
    #     bbox, label, conf = cv.detect_common_objects(frame)
    #
    #     output_image = draw_bbox(frame, bbox, label, conf)
    #
    #     cv2.imshow("test", output_image)
    #
    #     if cv2.waitKey(1) == ord('q'): # waits 1 millisecond between frames (if 0, then video will freeze)
    #         break
    #
    # cap.release()
    # cv2.destroyAllWindows()
