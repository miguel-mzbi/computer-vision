import numpy as np
import cv2
from matplotlib import pyplot as plt

def processFrame(cap):
    ret, frame = cap.read()
    imgBW = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    imgColor = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    plt.subplot(221)
    plt.title('Input Image')
    plt.imshow(imgColor)
    plt.xticks([]), plt.yticks([])

    plt.subplot(222)
    plt.title('Color Histogram')
    for i, col in enumerate(['r', 'g', 'b']):
        colorHist = cv2.calcHist([imgColor], [i], None, [256], [0, 256])
        plt.plot(colorHist, color = col)
        plt.xlim([0, 256])
    plt.xticks([]), plt.yticks([])

    plt.subplot(223)
    plt.title('Original Image B&W')
    plt.imshow(imgBW, cmap='gray')
    plt.xticks([]), plt.yticks([])

    plt.subplot(224)
    plt.title('B&W Histogram')
    histogram = cv2.calcHist([imgBW], [0], None, [256], [0, 256])
    plt.plot(histogram, color='k')
    plt.xlim([0, 256])
    plt.xticks([]), plt.yticks([])

cap = cv2.VideoCapture(0)
plt.ion()

while(cap.isOpened()):
    processFrame(cap)
    plt.pause(1)
    
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
plt.ioff()
plt.show()