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

#check that parameters lie near mean
class Line():
    def __init__(self):
        # was the line detected in the last iteration?
        self.detected = False

        #Left or Right line
        self.left = True

        #polynomial coefficients for the most recent fit
        self.current_fit = []

        # fit histovry over last n img
        self.history_fit = []

        #Last angle
        self.angle = None

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
        ploty = np.linspace(0, binary_warped.shape[0]-1, binary_warped.shape[0] )

        y_eval = np.max(ploty)
        # Define conversions in x and y from pixels space to meters
        ym_per_pix = 30/720 # meters per pixel in y dimension
        xm_per_pix = 3.7/700 # meters per pixel in x dimension

        # Fit new polynomials to x,y in world space
        left_fit_cr = np.polyfit(lefty*ym_per_pix, leftx*xm_per_pix, 2)

        # Calculate the new radius of curvature
        curve_rad = ((1 + (2*left_fit_cr[0]*y_eval*ym_per_pix + left_fit_cr[1])**2)**1.5) / np.absolute(2*left_fit_cr[0])

        # Now our radius of curvature is in meters
        # Example values: 632.1 m    626.2 m

        # Calculate Lane Deviation from center of lane:
        # First we calculate the intercept points at the bottom of our image, then use those to
        # calculate the lane deviation of the vehicle (assuming camera is in center of vehicle)
        scene_height = image_size[0] * ym_per_pix
        scene_width = image_size[1] * xm_per_pix

        #Calculate angle of
        def angle(value):
            return m.atan(value)*180/m.pi

        agnle = angle(2 * left_fit_cr[0] * scene_height + left_fit_cr[1])

        return curve_rad, agnle

    def best_fit(self, binary_warped):
        if(self.detected):
            fit = self.lane_line(binary_warped)
        else:
            #Get get line from previous fit
            fit = self.lane_line_from_fit(binary_warped, self.current_fit)
        #Get line pixels for drawing
        fitx, ploty = self.fit_lane(binary_warped, fit)
        #get geometry
        rad, angle = self.geometry(fitx, ploty, binary_warped)
        #TODO: Check angle
        if (True):
            self.current_fit = fit
        else:
            self.detected = False
        # If history queue has max length, pop first entered(last seen image)
        if(len(self.history_fit) == 5):
            self.history_fit.pop()
        self.history_fit.insert(0, self.current_fit)
        return m.mean(self.history_fit, axis=0)

# def pipeline(img):
#     global first
#     global left_fit
#     global right_fit
#     global last_left
#     global last_right
#     global last_la
#     global right_la
#     global n
#     obj_points, img_points = read_points()
#     undist = cal_undistort(img, obj_points, img_points)
#     corners = np.float32([[253, 697], [585, 456], [700, 456], [1061, 690]])
#     new_top_left = np.array([corners[0, 0], 0])
#     new_top_right = np.array([corners[3, 0], 0])
#     offset = [50, 0]
#     src = np.float32([corners[0], corners[1], corners[2], corners[3]])
#     dst = np.float32([corners[0] + offset, new_top_left + offset, new_top_right - offset, corners[3] - offset])
#     tr_img, t_m, inv_m = perspective_tr(undist, src, dst, (undist.shape[1], undist.shape[0]))
#     binary_warped = score_pixels(tr_img)
#     if first:
#         left_fit, right_fit, l, r, d, la, ra = lane_lines(binary_warped, visualise=False)
#         first = False
#     else:
#         left_fit, right_fit, l, r, d, la, ra = lane_lines2(binary_warped, left_fit, right_fit)
#     #smooth lanes over last 5 imgs
#     left_fit = smooth(left_fit, last_left, n, la, last_la)
#     right_fit = smooth(right_fit, last_right, n, ra, last_ra)
#     result = draw(binary_warped, img, inv_m, left_fit, right_fit, l, r, d, la, ra)
#     return result


