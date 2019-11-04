import numpy as np
import cv2
from matplotlib import pyplot as plt


def processFrame(cap):
    ret, frame = cap.read()
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    imgGaussian = cv2.GaussianBlur(gray, (3, 3), 0)

    sobelX64 = cv2.Sobel(imgGaussian, cv2.CV_64F, 1, 0, ksize=3)
    sobelY64 = cv2.Sobel(imgGaussian, cv2.CV_64F, 0, 1, ksize=3)
    sobelX = np.uint8(np.absolute(sobelX64))
    sobelY = np.uint8(np.absolute(sobelY64))
    sobel = sobelX + sobelY

    pretwittKernelX = np.array([[1, 1, 1], [0, 0, 0], [-1, -1, -1]])
    pretwittKernelY = np.array([[-1, 0, 1], [-1, 0, 1], [-1, 0, 1]])
    pretwittX = cv2.filter2D(imgGaussian, -1, pretwittKernelX)
    pretwittY = cv2.filter2D(imgGaussian, -1, pretwittKernelY)
    pretwitt = pretwittX + pretwittY

    canny = cv2.Canny(frame, 100, 150)

    plt.subplot(2,2,1)
    plt.title('Original')
    plt.imshow(rgb)
    plt.xticks([]),plt.yticks([])

    plt.subplot(2,2,2)
    plt.title('Sobel')
    plt.imshow(sobel,'gray')
    plt.xticks([]),plt.yticks([])

    plt.subplot(2,2,3)
    plt.title('Pretwitt')
    plt.imshow(pretwitt,'gray')
    plt.xticks([]),plt.yticks([])

    plt.subplot(2,2,4)
    plt.title('Canny')
    plt.imshow(canny,'gray')
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