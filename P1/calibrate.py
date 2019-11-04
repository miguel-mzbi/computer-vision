import cv2 as cv
import numpy as np
import glob
import math
criteria = (cv.TERM_CRITERIA_EPS + cv.TERM_CRITERIA_MAX_ITER, 30, 0.001)

objpoints = [] # 3d point in real world space
imgpoints = [] # 2d points in image plane.

images = glob.glob('images/*.jpg')
objp = np.zeros((7*9,3), np.float32)
objp[:,:2] = np.mgrid[0:9, 0:7].T.reshape(-1, 2)

for fname in images:
    img = cv.imread(fname)
    gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)

    ret, corners = cv.findChessboardCorners(gray, (9, 7), None)
    if ret == True:
        objpoints.append(objp)
        imgpoints.append(corners)

        refinedCorners = cv.cornerSubPix(gray, corners, (11, 11), (-1, -1), criteria)
        cv.drawChessboardCorners(img, (9, 7), refinedCorners, ret)
        cv.namedWindow('img', cv.WINDOW_NORMAL)
        cv.resizeWindow('img', 1920, 1080)
        cv.imshow('img', img)
        cv.waitKey(-1)

cv.destroyAllWindows()

ret, mtx, dist, rvecs, tvecs = cv.calibrateCamera(objpoints, imgpoints, gray.shape[::-1], None, None)
print(mtx)

fx = mtx[0][0]
fy = mtx[1][1]
print(fx, fy)

x1 = 730
y1 = 990

x2 = 737
y2 = 1094

p = 0.00000155
fx = fx*p*1000
fy = fy*p*1000
print(fx, fy)

dy1 = y1/fy
dy2 = y2/fy
dy = dy2-dy1
print(dy)

dx1 = x1/fx
dx2 = x2/fx
dx = dx2-dx1
print(dx)

h = math.sqrt(dy*dy + dx*dx)
print(h)
