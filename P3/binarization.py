import numpy as np
import cv2
from matplotlib import pyplot as plt

def processFrame(cap):
    ret, frame = cap.read()
    imgBW = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    _, thBinary = cv2.threshold(imgBW, 127, 255, cv2.THRESH_BINARY)
    thGaussian = cv2.adaptiveThreshold(imgBW, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 15, 4)
    _ ,thBinaryOtsu = cv2.threshold(imgBW, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    
    images = [imgBW, thBinary, thGaussian, thBinaryOtsu]
    titles = ['Original Image B&W', 'Global Thresholding (v = 127)', 'Adaptive Gaussian Thresholding', 'OTSU Thresholding']
    
    for i in range(4):
        plt.subplot(2,2,i+1)
        plt.imshow(images[i],'gray')
        plt.title(titles[i])
        plt.xticks([]),plt.yticks([])

cap = cv2.VideoCapture(0)
plt.ion()

while(cap.isOpened()):
    processFrame(cap)
    plt.pause(0.01)
    
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
plt.ioff()
plt.show()