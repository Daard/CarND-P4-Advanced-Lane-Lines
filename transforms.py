import pickle
import os.path
import numpy as np
import cv2
import glob
import matplotlib.pyplot as plt


def read_points(obj_file='obj_points.p', img_file='img_points.p', show=False, rewrite=False):

    if os.path.isfile(obj_file):
        obj_points = pickle.load(open(obj_file, 'rb'))

    if os.path.isfile(img_file):
        img_points = pickle.load(open(img_file, 'rb'))

    if rewrite:
        obj_points = []
        img_points = []
        images = glob.glob('./camera_cal/calibration*.jpg')
        obj_p = np.zeros((6 * 9, 3), np.float32)
        obj_p[:,:2] = np.mgrid[0:9, 0:6].T.reshape(-1, 2)

        for f_name in images:
            img = cv2.imread(f_name)
            # grayscale image
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            # find the chessboard corners
            ret, corners = cv2.findChessboardCorners(gray, (9, 6), None)
            if ret == True:
                img_points.append(corners)
                obj_points.append(obj_p)
                # Draw and display the corners
                if show:
                    cv2.drawChessboardCorners(img, (9, 6), corners, ret)
                    plt.imshow(img)
        pickle.dump(obj_points, open(obj_file, "wb"))
        pickle.dump(img_points, open(img_file, "wb"))

    return obj_points, img_points


def cal_undistort(img, objpoints, imgpoints):
    # Use cv2.calibrateCamera() and cv2.undistort()
    gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
    ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(objpoints, imgpoints, gray.shape[::-1], None, None)
    undist = cv2.undistort(img, mtx, dist, None, mtx)
    return undist


def perspective_tr(img, src, dst, img_size):
    t_m = cv2.getPerspectiveTransform(src, dst)
    tr_img = cv2.warpPerspective(img, t_m, img_size, flags=cv2.INTER_LINEAR)
    inv_m = cv2.getPerspectiveTransform(dst, src)
    return tr_img, t_m, inv_m

def gradient(img, s_thresh=(170, 255), sx_thresh=(20, 100)):
    img = np.copy(img)
    # Convert to HLS color space and separate the V channel
    hls = cv2.cvtColor(img, cv2.COLOR_RGB2HLS).astype(np.float)
    l_channel = hls[:,:,1]
    s_channel = hls[:,:,2]
    # Sobel x
    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    sobelx = cv2.Sobel(gray, cv2.CV_64F, 1, 0) # Take the derivative in x
    abs_sobelx = np.absolute(sobelx) # Absolute x derivative to accentuate lines away from horizontal
    scaled_sobel = np.uint8(255*abs_sobelx/np.max(abs_sobelx))
    
    # Threshold x gradient
    sxbinary = np.zeros_like(scaled_sobel)
    sxbinary[(scaled_sobel >= sx_thresh[0]) & (scaled_sobel <= sx_thresh[1])] = 1
    
    # Threshold color channel
    s_binary = np.zeros_like(s_channel)
    s_binary[(s_channel >= s_thresh[0]) & (s_channel <= s_thresh[1])] = 1
    # Stack each channel
    # Note color_binary[:, :, 0] is all 0s, effectively an all black image. It might
    # be beneficial to replace this channel with something else.
    color_binary = np.dstack(( np.zeros_like(sxbinary), sxbinary, s_binary)) * 255
    combined_binary = np.zeros_like(sxbinary)
    combined_binary[(s_binary == 1) | (sxbinary == 1)] = 1
    return combined_binary

