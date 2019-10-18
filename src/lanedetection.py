import numpy as np
import cv2
import matplotlib.pyplot as plt


# finished section for finding + displayig lines on img of lanes

#source: https://www.youtube.com/watch?v=eLTLtUVuuy4

#step 1 convert image to graycscale
def detect_lane():
    print("Starting")
    #image = cv2.imread('test2.mp4')
    image = cv2.imread('drivingpov.jpg')
    lane_image = np.copy(image)
    #create a greyscale video
    gray = cv2.cvtColor(lane_image, cv2.COLOR_RGB2GRAY) #flag to turn gray
    cv2.imshow("result", gray)
    cv2.waitKey(0)

#step 3 Canny method edge detection algorithm

# more about Canny algorithm here:
# https://docs.opencv.org/trunk/da/d22/tutorial_py_canny.html
def canny(image):
    #create a greyscale image
    gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY) #flag to turn gray
    #GaussianBlur to reduce noise
    blur = cv2.GaussianBlur(gray, (5, 5), 0) # to reduce noise
    # Step 3: detect lanes, figure out what Canny method does
    result = cv2.Canny(blur, 50, 150)
    return result

#made 26:00
def region_of_interest(image):
    height = image.shape[0]
    polygons = np.array([
    [(200, height), (1100, height), (550, 200)]
    ])
    mask = np.zeros_like(image)
    cv2.fillPoly(mask, polygons, 255)
    masked_image = cv2.bitwise_and(image, mask)
    return masked_image

#now to make just the polygon display in front of the lanes

#now display lines in front of lanes
def display_lines(image, lines):
    line_image = np.zeros_like(image)
    if lines is not None:
        for line in lines:
            x1, y1, x2, y2 = line.reshape(4)
            cv2.line(line_image, (x1, y1), (x2, y2), (255, 0, 0), 10)
            # specify where you want it to be drawn, and color
    return line_image

#step 2 apply a gaussian blur to reduce noise
def detect_lane2():
    image = cv2.imread('test2.jpg')

    lane_image = np.copy(image)
    # cv2.imshow("result", gray)
    cannyimg = canny(lane_image)
    cropped_image = region_of_interest(cannyimg)

    #why is there a 100 there?
    lines = cv2.HoughLinesP(cropped_image, 2, np.pi/180, 100, np.array([]), minLineLength=40, maxLineGap=5)
    line_image = display_lines(lane_image, lines)
    #combines images lines + lane
    # taking the weighted sum to two arrays
    combo_img = cv2.addWeighted(lane_image, 0.8, line_image, 1, 1) # gamma value at end
    plt.imshow(combo_img)
    plt.show()
    # cv2.waitKey(0)
    cv2.destroyAllWindows()

if __name__ == '__main__':
    while True:
        x = input("Quit? ")
        if x in ["q", "quit", "Quit", "yes"]:
            quit()
        else:
            # detect_lane()
            detect_lane2()

#STEPS
#1.
#7. finding lanes line - hough transform
