import matplotlib.image as mpimg
import matplotlib.pyplot as plt
import numpy as np
import cv2
import glob
import time
import pickle
import os
from sklearn.svm import LinearSVC
from sklearn.preprocessing import StandardScaler
from skimage.feature import hog
from sklearn.model_selection import train_test_split
from scipy.ndimage.measurements import label
from moviepy.editor import VideoFileClip
from heatmap import *
from feature_extract import *
from car_search import *


def pipeline(frame):
    # find cars
    ystart = 350
    ystop = 656
    scale = 1.5
    #img = mpimg.imread('../CarND-Vehicle-Detection/test_images/test1.jpg')
    out_img, hot_windows = find_cars(frame, ystart, ystop, scale, svc, X_scaler, orient, pix_per_cell, cell_per_block,
                                     spatial_size, hist_bins)
    # apply heatmap with threshold
    box_list = hot_windows
    heat = np.zeros_like(frame[:, :, 0]).astype(np.float)
    # Add heat to each box in box list
    heat = add_heat(heat, box_list)
    # Apply threshold to help remove false positives
    heat = apply_threshold(heat, 3)
    # Visualize the heatmap when displaying
    heatmap = np.clip(heat, 0, 255)
    #image = mpimg.imread('../CarND-Vehicle-Detection/test_images/test1.jpg')
    # Find final boxes from heatmap using label function
    labels = label(heatmap)
    draw_img = draw_labeled_bboxes(np.copy(frame), labels)

    '''
    # draw_img = draw_img.astype(np.float32) /255
    fig = plt.figure()
    plt.subplot(121)
    plt.imshow(draw_img)
    plt.title('Car Positions')
    plt.imsave('../CarND-Vehicle-Detection/output_images/Car_Position.jpg', draw_img)
    plt.subplot(122)
    plt.imshow(heatmap, cmap='hot')
    plt.title('Heat Map')
    plt.imsave('../CarND-Vehicle-Detection/output_images/Heat_Map.png', heatmap)
    fig.tight_layout()
    plt.show()
    '''

    return draw_img

if __name__ == '__main__':
    # load model
    svc = pickle.load(open('../CarND-Vehicle-Detection/svm_trained1.pickle', 'rb'))
    X_scaler = pickle.load(open('../CarND-Vehicle-Detection/X_scaler1.pickle', 'rb'))
    # parameters
    color_space = 'YCrCb'  # Can be RGB, HSV, LUV, HLS, YUV, YCrCb
    orient = 9  # HOG orientations
    pix_per_cell = 8  # HOG pixels per cell
    cell_per_block = 2  # HOG cells per block
    hog_channel = 'ALL'  # Can be 0, 1, 2, or "ALL"
    spatial_size = (32, 32)  # Spatial binning dimensions
    hist_bins = 16  # Number of histogram bins
    spatial_feat = True  # Spatial features on or off
    hist_feat = True  # Histogram features on or off
    hog_feat = True  # HOG features on or off
    y_start_stop = [350, 656]  # Min and max in y to search in slide_window()
    video = '../CarND-Vehicle-Detection/project_video.mp4'
    clip = VideoFileClip(video)
    clip1 = clip.fl_image(pipeline)
    out_video = '../CarND-Vehicle-Detection/out_video_project2.mp4'
    clip1.write_videofile(out_video, audio=False)
