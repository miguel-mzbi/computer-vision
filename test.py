import numpy as np
import cv2
from matplotlib import pyplot as plt

def getMask(hsvImage):

    lower = (0,10,50)
    upper = (180,40,100)
    mask = cv2.inRange(hsvImage, lower, upper)

    return mask

def applyMask(mask, hsvImage):
    return cv2.bitwise_and(hsvImage, hsvImage, mask=mask)

def processFrame(cap):
    _, frame = cap.read()

    frame = cv2.convertScaleAbs(frame, alpha=1, beta=50)
    
    hsvImage = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    grayImage = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    
    rgbImage = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    kernel = np.ones((5,5),np.uint8)
    rgbImage = cv2.morphologyEx(rgbImage, cv2.MORPH_OPEN, kernel, iterations=3)

    mask = getMask(hsvImage)

    segmentedImageHSV = applyMask(mask, hsvImage)
    segmentedImage = cv2.cvtColor(segmentedImageHSV, cv2.COLOR_HSV2RGB)

    # opening = cv2.morphologyEx(segmentedImage, cv2.MORPH_OPEN, kernel, iterations=2)
    # closing = cv2.morphologyEx(segmentedImage, cv2.MORPH_CLOSE, kernel, iterations=2)
    # gradient = cv2.morphologyEx(segmentedImage, cv2.MORPH_GRADIENT, kernel, iterations=1)


    plt.subplot(2,2,1)
    plt.title('Original')
    plt.imshow(rgbImage)
    plt.xticks([]),plt.yticks([])

    plt.subplot(2,2,2)
    plt.title('Mask')
    plt.imshow(mask)
    plt.xticks([]),plt.yticks([])

    plt.subplot(2,2,3)
    plt.title('Segmented')
    plt.imshow(segmentedImage)
    plt.xticks([]),plt.yticks([])

    # plt.subplot(2,2,1)
    # plt.title('Opening (Erosion->Dilation)')
    # plt.imshow(opening)
    # plt.xticks([]),plt.yticks([])
    # plt.subplot(2,2,2)
    # plt.title('Closing (Dilation->Erosion)')
    # plt.imshow(closing)
    # plt.xticks([]),plt.yticks([])
    # plt.subplot(2,2,4)
    # plt.title('Gradient (Dilation-Erosion)')
    # plt.imshow(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
    # plt.xticks([]),plt.yticks([])

if __name__ == "__main__":
    cap = cv2.VideoCapture(0)
    plt.ion()

    while(cap.isOpened()):
        processFrame(cap)
        
        press = plt.waitforbuttonpress(0.01)
        if press is None or press == False:
            pass
        else:
            break

    cap.release()
    plt.ioff()
    plt.show()