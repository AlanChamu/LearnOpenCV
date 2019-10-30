#source: https://www.youtube.com/watch?v=eLTLtUVuuy4

import numpy as np
import cv2
# to get numerical values for region_of_interest()
# import matplotlib.pyplot as plt

# XXX: make test footage .mp4 of mock street lane
#  straight, then a turn, to get information we can work with

# xxx: will try to add a third line in the middle

def make_coordinates(image, line_parameters):
    slope, intercept = line_parameters
    y1 = image.shape[0]
    # why do we have to convert to int?
    y2 = int(y1*(3/5))
    x1 = int((y1 - intercept)/slope)
    x2 = int((y2 - intercept)/slope)

    return np.array([x1, y1, x2, y2])

def average_slope_intercept(image, lines):
    # print("In average_slope_intercept ... ")
    left_fit = []
    right_fit = []
    # what does line.reshape(4) do?
    for line in lines:
    # for x1, y1, x2, y2 in lines:
        x1, y1, x2, y2 = line.reshape(4)
        parameters = np.polyfit((x1, x2), (y1, y2), 1)
        slope = parameters[0]
        intercept = parameters[1]
        if slope < 0:
            left_fit.append((slope, intercept))
        else:
            right_fit.append((slope, intercept))

    # can we use this?
    left_fit_average = np.average(left_fit, axis=0)
    right_fit_average = np.average(right_fit, axis=0)

    left_line = make_coordinates(image, left_fit_average)
    right_line = make_coordinates(image, right_fit_average)

    # print(left_line, right_line)
    return np.array([left_line, right_line])

# 19:00
# more about Canny algorithm here: https://docs.opencv.org/trunk/da/d22/tutorial_py_canny.html
def canny(image):
    #create a greyscale image
    gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY) #flag to turn gray
    #GaussianBlur to REDUCE NOISE -> FOR MORE ACCURATE EDGE DETECTION
    blur = cv2.GaussianBlur(gray, (5, 5), 0) # to reduce noise
    # Step 3: detect lanes, figure out what Canny method does
    # canny applies a 5by5 GaussianBlur either way
    result = cv2.Canny(blur, 50, 150) # low to high threshold
    return result

#made 26:00
def region_of_interest(image):
    # print("In region of interest")
    height = image.shape[0]
    # print(height)
    # crops the image
    polygons = np.array([
    [(200, height), (1100, height), (550, 200)]
    ])
    mask = np.zeros_like(image)
    cv2.fillPoly(mask, polygons, 255)
    masked_image = cv2.bitwise_and(image, mask)
    return masked_image

#now to make just the polygon display in front of the lanes

# now display lines in front of lanes
def display_lines(image, lines):
    line_image = np.zeros_like(image)
    if lines is not None:
        for x1, y1, x2, y2 in lines:
            # x1, y1, x2, y2 = line.reshape(4)
            # last equals line thickness, 2nd to last is color BGR
            cv2.line(line_image, (x1, y1), (x2, y2), (255, 0, 0), 10)
            # specify where you want it to be drawn, and color
    return line_image

def detect_lane_from_image(image):
        image = cv2.imread(image)
        lane_image = np.copy(image)
        cannyimg = canny(lane_image)
        cropped_image = region_of_interest(cannyimg)

        # why is there a 100 there?
        # This does the whole shabang, gets the points of interest
        lines = cv2.HoughLinesP(cropped_image, 2, np.pi/180, 100,
            np.array([]), minLineLength=40, maxLineGap=5)

        
        averaged_lines = average_slope_intercept(lane_image, lines)
        line_image = display_lines(lane_image, averaged_lines)
        #combines images lines + lane
        # taking the weighted sum to two arrays
        combo_img = cv2.addWeighted(lane_image, 0.8, line_image, 1, 1) # gamma value at end

        cv2.imshow("Result", combo_img)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

#step 2 apply a gaussian blur to reduce noise
def detect_lane_from_video(video):

    cap = cv2.VideoCapture("../../videos/"+video)

    while (cap.isOpened()):
        _, frame = cap.read()

        # computes gradient in all directions of the blurred frame
        # gets the "edges" in the image
        cannyimg = canny(frame)

        # crops image to only get the region that we want
        cropped_image = region_of_interest(cannyimg)


        # finds the lane line points (calculates the numbers in the image)
        #why is there a 100 there?
        lines = cv2.HoughLinesP(cropped_image, 2, np.pi/180,
            100, np.array([]), minLineLength=40, maxLineGap=5)

        #
        averaged_lines = average_slope_intercept(frame, lines)

        line_image = display_lines(frame, averaged_lines)
        #combines images lines + lane
        # taking the weighted sum to two arrays

        combo_img = cv2.addWeighted(frame, 0.8, line_image, 1, 1) # gamma value at end
        cv2.imshow("Result", combo_img)
        # dont relaly need to combine images, not unless you want to make it look cool

        # cv2.imshow("Result", line_image)

        if cv2.waitKey(1) == ord('q'): # waits 1 millisecond between frames (if 0, then video will freeze)
            break

    cap.release()
    cv2.destroyAllWindows()

#STEPS
#step 1 convert image to graycscale
#step 2 finding lanes line - hough transform
#step 10 find lane lines in video
