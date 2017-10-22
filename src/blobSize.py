# https://stackoverflow.com/questions/8076889/how-to-use-opencv-simpleblobdetector
# Standard imports
import cv2
import numpy as np

def readImage(path):
    return cv2.imread(path)

def getBlobDetector():
    # Setup SimpleBlobDetector parameters.
    params = cv2.SimpleBlobDetector_Params()
    # Change thresholds
    params.minThreshold = 50
    params.maxThreshold = 300
    # Filter by Area.
    params.filterByArea = True
    params.minArea = 10
    # Filter by Circularity
    params.filterByCircularity = True
    params.minCircularity = 0.4
    # Filter by Convexity
    params.filterByConvexity = True
    params.minConvexity = 0.8
    # Filter by Inertia
    params.filterByInertia = True
    params.minInertiaRatio = 0.8
    # Create a detector with the parameters
    return cv2.SimpleBlobDetector_create(params)

def getBlobs(im):
    detector = getBlobDetector()
    return detector.detect(im)

def drawBlobs(blobs):
    # Draw detected blobs as red circles.
    # cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS ensures
    # the size of the circle corresponds to the size of blob
    im_with_keypoints = cv2.drawKeypoints(im, blobs, np.array([]), (0,0,255), cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)

    # Show blobs
    cv2.imshow("Keypoints", im_with_keypoints)
    cv2.waitKey(0)

def maskGreen(im):
    # define range of blue color in HSV
    # TODO these are wrong.... very wrong
    lower = np.array([50,50,100])
    upper = np.array([255,255,255])
    # Threshold the HSV image to get only green colors
    return cv2.inRange(im, lower, upper)

if __name__ == "__main__":
    im = readImage("../test/test-1.png")
    #im = maskGreen(im)
    blobs = getBlobs(im)
    drawBlobs(blobs)

# TODO get the blob detection to work... at all