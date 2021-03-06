import cv2
import matplotlib.pyplot as plt
import numpy as np
import pickle
import os.path
import glob
import math as m


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

def score_pixels(img) -> np.ndarray:
    """
    Takes a road image and returns an image where pixel intensity maps to likelihood of it being part of the lane.

    Each pixel gets its own score, stored as pixel intensity. An intensity of zero means it is not from the lane,
    and a higher score means higher confidence of being from the lane.

    :param img: an image of a road, typically from an overhead perspective.
    :return: The score image.
    """
    # Settings to run thresholding operations on
    settings = [{'name': 'lab_b', 'cspace': 'LAB', 'channel': 2, 'clipLimit': 2.0, 'threshold': 150},
                {'name': 'value', 'cspace': 'HSV', 'channel': 2, 'clipLimit': 6.0, 'threshold': 220},
                {'name': 'lightness', 'cspace': 'HLS', 'channel': 1, 'clipLimit': 2.0, 'threshold': 210}]

    # Perform binary thresholding according to each setting and combine them into one image.
    scores = np.zeros(img.shape[0:2]).astype('uint8')
    for params in settings:
        # Change color space
        color_t = getattr(cv2, 'COLOR_RGB2{}'.format(params['cspace']))
        gray = cv2.cvtColor(img, color_t)[:, :, params['channel']]

        # Normalize regions of the image using CLAHE
        clahe = cv2.createCLAHE(params['clipLimit'], tileGridSize=(8, 8))
        norm_img = clahe.apply(gray)

        # Threshold to binary
        ret, binary = cv2.threshold(norm_img, params['threshold'], 1, cv2.THRESH_BINARY)

        scores += binary

        # Sobel x
    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    sobelx = cv2.Sobel(gray, cv2.CV_64F, 1, 0)  # Take the derivative in x
    abs_sobelx = np.absolute(sobelx)  # Absolute x derivative to accentuate lines away from horizontal
    scaled_sobel = np.uint8(255 * abs_sobelx / np.max(abs_sobelx))

    # Threshold x gradient
    sxbinary = np.zeros_like(scaled_sobel)
    sxbinary[(scaled_sobel >= 20) & (scaled_sobel <= 100)] = 1

    scores += sxbinary

    combined_binary = np.zeros_like(scores)
    combined_binary[scores > 1] = 1

    #         return cv2.normalize(scores, None, 0, 255, cv2.NORM_MINMAX)
    return combined_binary


def lane_lines(binary_warped, visualise=False):
    # Assuming you have created a warped binary image called "binary_warped"
    # Take a histogram of the bottom half of the image
    histogram = np.sum(binary_warped[binary_warped.shape[0] // 2:, :], axis=0)
    # Create an output image to draw on and  visualize the result
    if visualise:
        out_img = np.dstack((binary_warped, binary_warped, binary_warped)) * 255
    # Find the peak of the left and right halves of the histogram
    # These will be the starting point for the left and right lines
    midpoint = np.int(histogram.shape[0] / 2)
    leftx_base = np.argmax(histogram[:midpoint])
    rightx_base = np.argmax(histogram[midpoint:]) + midpoint

    # Choose the number of sliding windows
    nwindows = 9
    # Set height of windows
    window_height = np.int(binary_warped.shape[0] / nwindows)
    # Identify the x and y positions of all nonzero pixels in the image
    nonzero = binary_warped.nonzero()
    nonzeroy = np.array(nonzero[0])
    nonzerox = np.array(nonzero[1])
    # Current positions to be updated for each window
    leftx_current = leftx_base
    rightx_current = rightx_base
    # Set the width of the windows +/- margin
    margin = 100
    # Set minimum number of pixels found to recenter window
    minpix = 50
    # Create empty lists to receive left and right lane pixel indices
    left_lane_inds = []
    right_lane_inds = []

    # Step through the windows one by one
    for window in range(nwindows):
        # Identify window boundaries in x and y (and right and left)
        win_y_low = binary_warped.shape[0] - (window + 1) * window_height
        win_y_high = binary_warped.shape[0] - window * window_height
        win_xleft_low = leftx_current - margin
        win_xleft_high = leftx_current + margin
        win_xright_low = rightx_current - margin
        win_xright_high = rightx_current + margin
        # Draw the windows on the visualization image
        if visualise:
            cv2.rectangle(out_img, (win_xleft_low, win_y_low), (win_xleft_high, win_y_high),
                          (0, 255, 0), 2)
            cv2.rectangle(out_img, (win_xright_low, win_y_low), (win_xright_high, win_y_high),
                          (0, 255, 0), 2)
        # Identify the nonzero pixels in x and y within the window
        good_left_inds = ((nonzeroy >= win_y_low) & (nonzeroy < win_y_high) &
                          (nonzerox >= win_xleft_low) & (nonzerox < win_xleft_high)).nonzero()[0]
        good_right_inds = ((nonzeroy >= win_y_low) & (nonzeroy < win_y_high) &
                           (nonzerox >= win_xright_low) & (nonzerox < win_xright_high)).nonzero()[0]
        # Append these indices to the lists
        left_lane_inds.append(good_left_inds)
        right_lane_inds.append(good_right_inds)
        # If you found > minpix pixels, recenter next window on their mean position
        if len(good_left_inds) > minpix:
            leftx_current = np.int(np.mean(nonzerox[good_left_inds]))
        if len(good_right_inds) > minpix:
            rightx_current = np.int(np.mean(nonzerox[good_right_inds]))

    # Concatenate the arrays of indices
    left_lane_inds = np.concatenate(left_lane_inds)
    right_lane_inds = np.concatenate(right_lane_inds)

    # Extract left and right line pixel positions
    leftx = nonzerox[left_lane_inds]
    lefty = nonzeroy[left_lane_inds]
    rightx = nonzerox[right_lane_inds]
    righty = nonzeroy[right_lane_inds]

    # Fit a second order polynomial to each
    left_fit = np.polyfit(lefty, leftx, 2)
    right_fit = np.polyfit(righty, rightx, 2)

    l, r, d, la, ra = get_curvature(leftx, lefty, rightx, righty, binary_warped)

    if visualise:
        out_img[nonzeroy[left_lane_inds], nonzerox[left_lane_inds]] = [255, 0, 0]
        out_img[nonzeroy[right_lane_inds], nonzerox[right_lane_inds]] = [0, 0, 255]
        return left_fit, right_fit, l, r, d, la, ra, out_img
    else:
        return left_fit, right_fit, l, r, d, la, ra


def lane_lines2(binary_warped, left_fit, right_fit):
    # Assume you now have a new warped binary image
    # from the next frame of video (also called "binary_warped")
    # It's now much easier to find line pixels!
    nonzero = binary_warped.nonzero()
    nonzeroy = np.array(nonzero[0])
    nonzerox = np.array(nonzero[1])
    margin = 100
    left_lane_inds = ((nonzerox > (left_fit[0] * (nonzeroy ** 2) + left_fit[1] * nonzeroy +
                                   left_fit[2] - margin)) & (nonzerox < (left_fit[0] * (nonzeroy ** 2) +
                                                                         left_fit[1] * nonzeroy + left_fit[
                                                                             2] + margin)))

    right_lane_inds = ((nonzerox > (right_fit[0] * (nonzeroy ** 2) + right_fit[1] * nonzeroy +
                                    right_fit[2] - margin)) & (nonzerox < (right_fit[0] * (nonzeroy ** 2) +
                                                                           right_fit[1] * nonzeroy + right_fit[
                                                                               2] + margin)))

    # Again, extract left and right line pixel positions
    leftx = nonzerox[left_lane_inds]
    lefty = nonzeroy[left_lane_inds]
    rightx = nonzerox[right_lane_inds]
    righty = nonzeroy[right_lane_inds]
    # Fit a second order polynomial to each
    left_fit = np.polyfit(lefty, leftx, 2)
    right_fit = np.polyfit(righty, rightx, 2)

    l, r, d, la, ra = get_curvature(leftx, lefty, rightx, righty, binary_warped)

    return left_fit, right_fit, l, r, d, la, ra


def fit_lanes(warped, left_fit, right_fit):
    # Generate x and y values for plotting
    ploty = np.linspace(0, warped.shape[0] - 1, warped.shape[0])
    left_fitx = left_fit[0] * ploty ** 2 + left_fit[1] * ploty + left_fit[2]
    right_fitx = right_fit[0] * ploty ** 2 + right_fit[1] * ploty + right_fit[2]
    return left_fitx, right_fitx, ploty


def get_curvature(leftx, lefty, rightx, righty, binary_warped):
    image_size = binary_warped.shape
    ploty = np.linspace(0, binary_warped.shape[0] - 1, binary_warped.shape[0])

    y_eval = np.max(ploty)
    # Define conversions in x and y from pixels space to meters
    ym_per_pix = 30 / 720  # meters per pixel in y dimension
    xm_per_pix = 3.7 / 700  # meters per pixel in x dimension

    # Fit new polynomials to x,y in world space
    left_fit_cr = np.polyfit(lefty * ym_per_pix, leftx * xm_per_pix, 2)
    right_fit_cr = np.polyfit(righty * ym_per_pix, rightx * xm_per_pix, 2)

    # Calculate the new radius of curvature
    left_curverad = ((1 + (2 * left_fit_cr[0] * y_eval * ym_per_pix + left_fit_cr[1]) ** 2) ** 1.5) / np.absolute(
        2 * left_fit_cr[0])
    right_curverad = ((1 + (2 * right_fit_cr[0] * y_eval * ym_per_pix + right_fit_cr[1]) ** 2) ** 1.5) / np.absolute(
        2 * right_fit_cr[0])

    # Now our radius of curvature is in meters
    # Example values: 632.1 m    626.2 m

    # Calculate Lane Deviation from center of lane:
    # First we calculate the intercept points at the bottom of our image, then use those to
    # calculate the lane deviation of the vehicle (assuming camera is in center of vehicle)
    scene_height = image_size[0] * ym_per_pix
    scene_width = image_size[1] * xm_per_pix

    left_intercept = left_fit_cr[0] * scene_height ** 2 + left_fit_cr[1] * scene_height + left_fit_cr[2]
    right_intercept = right_fit_cr[0] * scene_height ** 2 + right_fit_cr[1] * scene_height + right_fit_cr[2]
    calculated_center = (left_intercept + right_intercept) / 2.0

    # Calculate angle of
    def angle(value):
        return m.atan(value) * 180 / m.pi

    left_agnle = angle(2 * left_fit_cr[0] * scene_height + left_fit_cr[1])
    right_angle = angle(2 * right_fit_cr[0] * scene_height + right_fit_cr[1])

    lane_deviation = (calculated_center - scene_width / 2.0)

    return left_curverad, right_curverad, lane_deviation, left_agnle, right_angle


def draw(warped, undist, Minv, left_fit, right_fit, l, r, d, la, ra):
    # Draw lanes on scene
    left_fitx, right_fitx, ploty = fit_lanes(warped, left_fit, right_fit)

    # Create an image to draw the lines on
    warp_zero = np.zeros_like(warped).astype(np.uint8)
    color_warp = np.dstack((warp_zero, warp_zero, warp_zero))

    # Recast the x and y points into usable format for cv2.fillPoly()
    pts_left = np.array([np.transpose(np.vstack([left_fitx, ploty]))])
    pts_right = np.array([np.flipud(np.transpose(np.vstack([right_fitx, ploty])))])
    pts = np.hstack((pts_left, pts_right))

    # Draw the lane onto the warped blank image
    cv2.fillPoly(color_warp, np.int_([pts]), (0, 255, 0))

    # Warp the blank back to original image space using inverse perspective matrix (Minv)
    newwarp = cv2.warpPerspective(color_warp, Minv, (undist.shape[1], undist.shape[0]))
    # Combine the result with the original image
    result = cv2.addWeighted(undist, 1, newwarp, 0.3, 0)

    # TODO: add left_r, right_r, deviation
    curvature_text = "Curvature: Left = " + str(np.round(l, 2)) + ", Right = " + str(np.round(r, 2))
    font = cv2.FONT_HERSHEY_COMPLEX
    cv2.putText(result, curvature_text, (30, 60), font, 1, (0, 255, 0), 2)
    deviation_text = "Lane deviation from center = {:.2f} m".format(d)
    cv2.putText(result, deviation_text, (30, 90), font, 1, (0, 255, 0), 2)
    angle_text = "Angle: Left = " + str(np.round(la, 2)) + ", Right = " + str(np.round(ra, 2))
    cv2.putText(result, angle_text, (30, 120), font, 1, (0, 255, 0), 2)
    return result






