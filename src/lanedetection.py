#source: https://www.youtube.com/watch?v=eLTLtUVuuy4

# XXX: TAKE BETTER test footage VIDEO, WITH TURNS

import numpy as np
import cv2
# to get numerical values for region_of_interest()
import matplotlib.pyplot as plt

def make_coordinates(image, line_parameters):
    slope, intercept = line_parameters
    print("In make_coordinates,", image, line_parameters)
    y1 = image.shape[0]
    print("y1,", y1)
    # why do we have to convert to int?
    y2 = int(y1*(3/5))
    x1 = int((y1 - intercept)/slope)
    x2 = int((y2 - intercept)/slope)
    print("array,", [x1, y1, x2, y2])
    return np.array([x1, y1, x2, y2])

def average_slope_intercept(image, lines):
    print("In average_slope_intercept ... ")
    left_fit = []
    right_fit = []

    for line in lines:
        x1, y1, x2, y2 = line.reshape(4)
        parameters = np.polyfit((x1, x2), (y1, y2), 1)
        slope = parameters[0]
        intercept = parameters[1]

        if slope < 0:
            left_fit.append((slope, intercept))
        else:
            right_fit.append((slope, intercept))

    # print("here?")
    left_fit_average = np.average(left_fit, axis=0)
    right_fit_average = np.average(right_fit, axis=0)
    # print("here?")
    left_line = make_coordinates(image, left_fit_average)
    right_line = make_coordinates(image, right_fit_average)
    print("here?")
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

    polygons = np.array([
    [(500, height), (1250, height), (800, 0)]
    ])

    # for test mp4 file from tutorial
    # polygons = np.array([
    # [(200, height), (1100, height), (550, 200)]
    # ])

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
    print("in get_middle_line")

    top = []
    bottom = []

    if lines is not None:
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

    return np.array([top,bottom])

    #  have to return something like [[(x1, y1),(x2, y2)],
    #                                   [(x1, y1), (x2, y2)]]

def detect_lane_from_video(video):

    cap = cv2.VideoCapture("../../videos/"+video)

    while (cap.isOpened()):
        _, frame = cap.read()
        print("one")
        cannyimg = canny(frame)
        print("two")
        cropped_image = region_of_interest(cannyimg)
        print("three")
        lines = cv2.HoughLinesP(cropped_image, 2, np.pi/180,
            100, np.array([]), minLineLength=40, maxLineGap=5)
        print("four")
        averaged_lines = average_slope_intercept(frame, lines)
        print("five")
        # may have to return an angle value to allow car to turn directions
        middle_line = get_middle_line(averaged_lines)
        print("six")
        line_image = display_lines(frame, middle_line, (255, 0, 0))
        # line_image = display_lines(frame, averaged_lines, (255, 0, 0))
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

#STEPS
#step 1 convert image to graycscale
#step 2 finding lanes line - hough transform
#step 10 find lane lines in video
