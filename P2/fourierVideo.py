import numpy as np
import cv2
from matplotlib import pyplot as plt

def processFrame(cap):
    ret, frame = cap.read()
    img = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Obtain Fourier transform
    discreteFourier = cv2.dft(np.float32(img),flags = cv2.DFT_COMPLEX_OUTPUT)
    # Shift zero to center
    dfShift = np.fft.fftshift(discreteFourier)
    # Obtain magnitude spectrum
    magnitudeSpectrum = 20*np.log(cv2.magnitude(dfShift[:,:,0],dfShift[:,:,1]))

    # Show magnitude spectrum
    plt.figure(1)
    plt.imshow(img, cmap = 'gray')
    plt.title('Input Image'), plt.xticks([]), plt.yticks([])

    # Obtain img shape (rows and columns)
    rows, cols = img.shape
    # Create a mask (LPF), center square is 1 (Low frequencies in the center), remaining all zeros
    lpf = np.zeros((rows, cols, 2), np.uint8)
    lpf[(rows//2-30):(rows//2+30), (cols//2-30):(cols//2+30)] = 1
    # Create a mask (HPF), center square is 0 (Low frequencies in the center), remaining all ones
    hpf = np.ones((rows, cols, 2), np.uint8)
    hpf[(rows//2-30):(rows//2+30), (cols//2-30):(cols//2+30)] = 0

    # Apply mask and inverse discrete fourier (LPF)
    iShift = dfShift*lpf
    iFourier = np.fft.ifftshift(iShift)
    lowImg = cv2.idft(iFourier)
    lowImg = cv2.magnitude(lowImg[:,:,0],lowImg[:,:,1])
    # Show LPF output
    plt.figure(2)
    plt.imshow(lowImg, cmap = 'gray')
    plt.title('LPF'), plt.xticks([]), plt.yticks([])

    # Apply mask and inverse discrete fourier (HPF)
    iShift = dfShift*hpf
    iFourier = np.fft.ifftshift(iShift)
    highImg = cv2.idft(iFourier)
    highImg = cv2.magnitude(highImg[:,:,0],highImg[:,:,1])
    # Show HPF output
    plt.figure(3)
    plt.imshow(highImg, cmap = 'gray')
    plt.title('HPF'), plt.xticks([]), plt.yticks([])

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