import numpy as np
import cv2
from matplotlib import pyplot as plt


def processFrame(cap):
    ret, frame = cap.read()
    img = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    _ ,thBinaryOtsu = cv2.threshold(img, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    thBinaryOtsu = np.where(thBinaryOtsu==0, 1, thBinaryOtsu)
    thBinaryOtsu = np.where(thBinaryOtsu==255, 0, thBinaryOtsu)
    
    edges = cv2.Canny(frame, 30, 80)

    plt.subplot(1,3,1)
    plt.imshow(thBinaryOtsu,'gray')
    plt.xticks([]),plt.yticks([])

    plt.subplot(1,3,2)
    plt.imshow(edges,'gray')
    plt.xticks([]),plt.yticks([])

    res = edges * thBinaryOtsu

    plt.subplot(1,3,3)
    plt.imshow(res,'gray')
    plt.xticks([]),plt.yticks([])

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