import cv2
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import numpy as np
import imageio
from moviepy.editor import VideoFileClip
from IPython.display import HTML
import pickle
import os.path
import glob
import math as m


def fit_lanes(): None


def read_points(): None


def cal_undistort(): None


def perspective_tr(): None


def score_pixels(): None


# check that parameters lie near mean
class Line():
    def __init__(self):
        # was the line detected in the last iteration?
        self.detected = False

        # Left or Right line
        self.left = True

        # polynomial coefficients for the most recent fit
        self.current_fit = None

        # fit histovry over last n img
        self.history_fit = []

        # Last angle
        self.angle = None

        # Last curvature rad
        self.curve_rad = None

        # Best fit
        self.best_fit = None



    def lane_line_from_fit(self, binary_warped, fit):
        # Assume you now have a new warped binary image
        # from the next frame of video (also called "binary_warped")
        # It's now much easier to find line pixels!
        nonzero = binary_warped.nonzero()
        nonzeroy = np.array(nonzero[0])
        nonzerox = np.array(nonzero[1])
        margin = 100
        lane_inds = ((nonzerox > (fit[0] * (nonzeroy ** 2) + fit[1] * nonzeroy +
                                  fit[2] - margin)) & (nonzerox < (fit[0] * (nonzeroy ** 2) +
                                                                   fit[1] * nonzeroy + fit[
                                                                       2] + margin)))
        # Again, extract left and right line pixel positions
        leftx = nonzerox[lane_inds]
        lefty = nonzeroy[lane_inds]
        # Fit a second order polynomial to each
        return np.polyfit(lefty, leftx, 2)

    def lane_line(self, binary_warped):
        # Assuming you have created a warped binary image called "binary_warped"
        # Take a histogram of the bottom half of the image
        histogram = np.sum(binary_warped[binary_warped.shape[0] // 2:, :], axis=0)
        # Find the peak of the left and right halves of the histogram
        # These will be the starting point for the left and right lines
        midpoint = np.int(histogram.shape[0] / 2)
        if self.left:
            x_base = np.argmax(histogram[:midpoint])
        else:
            x_base = np.argmax(histogram[midpoint:]) + midpoint

        # Choose the number of sliding windows
        nwindows = 9
        # Set height of windows
        window_height = np.int(binary_warped.shape[0] / nwindows)
        # Identify the x and y positions of all nonzero pixels in the image
        nonzero = binary_warped.nonzero()
        nonzeroy = np.array(nonzero[0])
        nonzerox = np.array(nonzero[1])
        # Current positions to be updated for each window
        x_current = x_base
        # Set the width of the windows +/- margin
        margin = 100
        # Set minimum number of pixels found to recenter window
        minpix = 50
        # Create empty lists to receive left and right lane pixel indices
        lane_inds = []

        # Step through the windows one by one
        for window in range(nwindows):
            # Identify window boundaries in x and y (and right and left)
            win_y_low = binary_warped.shape[0] - (window + 1) * window_height
            win_y_high = binary_warped.shape[0] - window * window_height
            win_x_low = x_current - margin
            win_x_high = x_current + margin
            # Identify the nonzero pixels in x and y within the window
            good_inds = ((nonzeroy >= win_y_low) & (nonzeroy < win_y_high) &
                         (nonzerox >= win_x_low) & (nonzerox < win_x_high)).nonzero()[0]
            # Append these indices to the lists
            lane_inds.append(good_inds)

            # If you found > minpix pixels, recenter next window on their mean position
            if len(good_inds) > minpix:
                x_current = np.int(np.mean(nonzerox[good_inds]))

        # Concatenate the arrays of indices
        lane_inds = np.concatenate(lane_inds)

        # Extract left and right line pixel positions
        x = nonzerox[lane_inds]
        y = nonzeroy[lane_inds]

        # Fit a second order polynomial to each
        return np.polyfit(y, x, 2)

    def fit_lane(self, warped, fit):
        # Generate x and y values for plotting
        ploty = np.linspace(0, warped.shape[0] - 1, warped.shape[0])
        fitx = fit[0] * ploty ** 2 + fit[1] * ploty + fit[2]
        return fitx, ploty

    def geometry(self, leftx, lefty, binary_warped):
        image_size = binary_warped.shape
        ploty = np.linspace(0, binary_warped.shape[0] - 1, binary_warped.shape[0])

        y_eval = np.max(ploty)
        # Define conversions in x and y from pixels space to meters
        ym_per_pix = 30 / 720  # meters per pixel in y dimension
        xm_per_pix = 3.7 / 700  # meters per pixel in x dimension

        # Fit new polynomials to x,y in world space
        left_fit_cr = np.polyfit(lefty * ym_per_pix, leftx * xm_per_pix, 2)

        # Calculate the new radius of curvature
        curve_rad = ((1 + (2 * left_fit_cr[0] * y_eval * ym_per_pix + left_fit_cr[1]) ** 2) ** 1.5) / np.absolute(
            2 * left_fit_cr[0])

        # Now our radius of curvature is in meters
        # Example values: 632.1 m    626.2 m

        # Calculate Lane Deviation from center of lane:
        # First we calculate the intercept points at the bottom of our image, then use those to
        # calculate the lane deviation of the vehicle (assuming camera is in center of vehicle)
        scene_height = image_size[0] * ym_per_pix
        scene_width = image_size[1] * xm_per_pix

        # Calculate angle of
        angle = m.atan(2 * left_fit_cr[0] * scene_height + left_fit_cr[1]) * 180 / m.pi

        return curve_rad, angle

    def calculate(self, binary_warped):
        if (not self.detected):
            fit = self.lane_line(binary_warped)
        else:
            # Get get line from previous fit
            fit = self.lane_line_from_fit(binary_warped, self.current_fit)
        if self.current_fit is None:
            self.current_fit = fit
        # Get line pixels for drawing
        fitx, ploty = self.fit_lane(binary_warped, fit)
        # get geometry
        curve_rad, angle = self.geometry(fitx, ploty, binary_warped)

        # Sanity check, lines must be parallel
        if self.angle is not None & m.fabs(self.angle - angle) < 0.5:
            self.current_fit = fit
        else:
            self.detected = False
        self.angle = angle
        self.curve_rad = curve_rad
        # If history queue has max length, pop first entered(last seen image)
        if (len(self.history_fit) == 5):
            self.history_fit.pop()
        self.history_fit.insert(0, self.current_fit)
        self.best_fit = np.mean(self.history_fit, axis=0)


def draw2(warped, undist, Minv, left, right):
    # Draw lanes on scene
    left_fitx, right_fitx, ploty = fit_lanes(warped, left.best_fit, right.best_fit)

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

    curvature_text = "Curvature: Left = " + str(np.round(left.curve_rad, 2)) + ", Right = " + str(np.round(right.curve_rad, 2))
    font = cv2.FONT_HERSHEY_COMPLEX
    cv2.putText(result, curvature_text, (30, 60), font, 1, (0, 255, 0), 2)
    #TODO: add deviation from center
    deviation_text = "Lane deviation from center = {:.2f} m".format(0)
    cv2.putText(result, deviation_text, (30, 90), font, 1, (0, 255, 0), 2)
    angle_text = "Angle: Left = " + str(np.round(left.angle, 2)) + ", Right = " + str(np.round(right.angle, 2))
    cv2.putText(result, angle_text, (30, 120), font, 1, (0, 255, 0), 2)
    return result

def pipeline2(img):
    global left_line
    global right_line
    obj_points, img_points = read_points()
    undist = cal_undistort(img, obj_points, img_points)
    corners = np.float32([[253, 697], [585, 456], [700, 456], [1061, 690]])
    new_top_left = np.array([corners[0, 0], 0])
    new_top_right = np.array([corners[3, 0], 0])
    offset = [50, 0]
    src = np.float32([corners[0], corners[1], corners[2], corners[3]])
    dst = np.float32([corners[0] + offset, new_top_left + offset, new_top_right - offset, corners[3] - offset])
    tr_img, t_m, inv_m = perspective_tr(undist, src, dst, (undist.shape[1], undist.shape[0]))
    binary_warped = score_pixels(tr_img)
    left_line.calculate(binary_warped)
    right_line.calculate(binary_warped)
    result = draw2(binary_warped, img, inv_m, left_line, right_line)
    return result