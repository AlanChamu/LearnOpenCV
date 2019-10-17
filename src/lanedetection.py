import numpy as np
import cv2
import matplotlib


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

#step 2 apply a gaussian blur to reduce noise
def detect_lane2():
    image = cv2.imread('test2.jpg')
    lane_image = np.copy(image)
    #create a greyscale video
    gray = cv2.cvtColor(lane_image, cv2.COLOR_RGB2GRAY) #flag to turn gray
    #GaussianBlur to reduce noise
    blur = cv2.GaussianBlur(gray, (5, 5), 0) # to reduce noise
    # Step 3: detect lanes
    canny = cv2.Canny(blur, 50, 150)
    # cv2.imshow("result", gray)
    cv2.imshow("result", canny)
    cv2.waitKey(0)
    cv2.destroyAllWindows()



if __name__ == '__main__':
    while True:
        x = input("Quit? ")
        if x in ["q", "quit", "Quit", "yes"]:
            quit()
        else:
            # detect_lane()
            detect_lane2()
