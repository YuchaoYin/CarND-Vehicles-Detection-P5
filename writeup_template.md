**Vehicle Detection Project**

The goals / steps of this project are the following:

* Perform a Histogram of Oriented Gradients (HOG) feature extraction on a labeled training set of images and train a classifier Linear SVM classifier
* Optionally, you can also apply a color transform and append binned color features, as well as histograms of color, to your HOG feature vector. 
* Note: for those first two steps don't forget to normalize your features and randomize a selection for training and testing.
* Implement a sliding-window technique and use your trained classifier to search for vehicles in images.
* Run your pipeline on a video stream (start with the test_video.mp4 and later implement on full project_video.mp4) and create a heat map of recurring detections frame by frame to reject outliers and follow detected vehicles.
* Estimate a bounding box for vehicles detected.

[//]: # (Image References)
[image1]: ./output_images/data_exploration.png
[image2]: ./output_images/color_hist.png
[image3]: ./output_images/hog_feature.png
[image4]: ./output_images/hot_windows.png
[image5]: ./output_images/Heat_Map.png
[video1]: ./project_video.mp4

## [Rubric](https://review.udacity.com/#!/rubrics/513/view) Points
### Here I will consider the rubric points individually and describe how I addressed each point in my implementation.  

### Data Exploration

For this project we used the dataset for vehicles and non-vehicles. 
The code can be found in [data_exploration.py](data_exploration.py)
The function returned a count of 8792  cars and 8968  non-cars of size:  (64, 64, 3)  and data type: float32
Here is an example of car and not-car images:

![alt text][image1]

### Features Extraction
The code can be found in [feature_extract.py](feature_extract.py)
#### 1. Spatial Binning of Color

Raw pixel values are quite useful to include in feature vector in searchin for cars. 
The image is resized to 32x32 and converted to a one dimensional feature vector.
```
def bin_spatial(img, size=(32, 32)):
    # Use cv2.resize().ravel() to create the feature vector
    features = cv2.resize(img, size).ravel()
    # Return the feature vector
return features
```

#### 2. Color Histogram Features
I also used histograms of pixel intensity (color histograms) as features.
Here is the code:
```
def color_hist(img, nbins=32, bins_range=(0, 256)):
    img = np.float32(img)
    img = img/ img.max() * 255
    # Compute the histogram of the color channels separately
    channel1_hist = np.histogram(img[:,:,0], bins=nbins, range=bins_range)
    channel2_hist = np.histogram(img[:,:,1], bins=nbins, range=bins_range)
    channel3_hist = np.histogram(img[:,:,2], bins=nbins, range=bins_range)
    # Concatenate the histograms into a single feature vector
    hist_features = np.concatenate((channel1_hist[0], channel2_hist[0], channel3_hist[0]))
    # Return the individual histograms, bin_centers and feature vector
return hist_features
```
The following figure shows the result using RGB color space. But actually in the end, I used YCrCb color space to train the model.

![alt text][image2]

#### 3. Histogram of Oriented Gradients (HOG)

For HOG features I applied the sub-sampling window search method to save the computational cost.
The code can be found in [hog_subsampling.py](hog_subsampling.py)
The hog() function takes in a single color channel or grayscaled image as input, as well as various parameters. These parameters include orientations, pixels_per_cell and cells_per_block. 
I extracted HOG features from all color channels. 
here is an example of one channel result:

![alt text][image3]

#### 4. Parameters Configuration
After various attemps, I came up to the following parameters:
```
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
    y_start_stop = [350, 656] # Min and max in y to search in slide_window()
```
### Model Traning

I used all the features mentioned above to train a linear SVM.
The code can be found in [train_main.py](train_main.py)
The trained model is stored in svm_trained1.pickle and X_scaler1.pickle.


### Sliding Window Search

I also tried to use a basic sliding window search in a first phase. The code can be found in [car_search.py](car_search.py). Unfortunately, this naive method could be very slow. In order to save the computational cost, I implemented a sub sampling window search as mentioned before. This method can only search the region of interest and the HOG features are computed only once. The function is called find_cars in [hog_subsampling.py](hog_subsampling.py)

The result is shown bellow:

![alt text][image4]

### Heat Map

Obviously, there are some false positive detections in the above image. 
In order to overcomet this, I implemented a heat-map with threshold. 
The code is in [heatmap.py](heatmap.py).

Final result:

![alt text][image5]


### Video Implementation

The code can be found in [main.py](main.py).
<p align="center">
<img src="./final.gif" alt="final gif" width="50%" height="50%"></a>
 <br>Final Gif Result. 
</p>


Here's a [link to my video result](./out_video_project2.mp4)

---



### Discussion

#### 1. Briefly discuss any problems / issues you faced in your implementation of this project.  Where will your pipeline likely fail?  What could you do to make it more robust?

The performance is strongly relied on the chosed parameters. So if the environment changes, I am afraid our approach might not work as expected. A neural network can be a good choise instead.


