import numpy as np
import cv2
from matplotlib import pyplot as plt
from matplotlib.widgets import Slider, Button, RadioButtons

def adjustBrightness(img, value=30):
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    h, s, v = cv2.split(hsv)
    cv2.add(v, value, v)
    final_hsv = cv2.merge((h, s, v))
    img = cv2.cvtColor(final_hsv, cv2.COLOR_HSV2BGR)
    return img

def processFrame(cap):
    brightnessAdjust = 100
    ret, frame = cap.read()

    # adjusted = adjustBrightness(frame, value=-100)

    adjusted = cv2.convertScaleAbs(frame, alpha=2, beta=50)
    
    adjustedRGB = cv2.cvtColor(adjusted, cv2.COLOR_BGR2RGB)
    originalRGB = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    plt.subplot(2,1,1)
    plt.imshow(originalRGB)
    plt.xticks([]),plt.yticks([])
    plt.subplot(2,1,2)
    plt.imshow(adjustedRGB)
    plt.xticks([]),plt.yticks([])

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