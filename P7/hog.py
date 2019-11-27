import cv2

if __name__ == "__main__":
    hog = cv2.HOGDescriptor()
    im = cv2.imread('./P7/emojis.jpeg')
    h = hog.compute(im)
    print(h.shape)