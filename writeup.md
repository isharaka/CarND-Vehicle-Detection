##Writeup Template
###You can use this file as a template for your writeup if you want to submit it as a markdown file, but feel free to use some other method and submit a pdf if you prefer.

---

**Vehicle Detection Project**

The goals / steps of this project are the following:

* Perform a Histogram of Oriented Gradients (HOG) feature extraction on a labeled training set of images and train a classifier Linear SVM classifier
* Optionally, you can also apply a color transform and append binned color features, as well as histograms of color, to your HOG feature vector. 
* Note: for those first two steps don't forget to normalize your features and randomize a selection for training and testing.
* Implement a sliding-window technique and use your trained classifier to search for vehicles in images.
* Run your pipeline on a video stream (start with the test_video.mp4 and later implement on full project_video.mp4) and create a heat map of recurring detections frame by frame to reject outliers and follow detected vehicles.
* Estimate a bounding box for vehicles detected.

[//]: # (Image References)
[image1]: ./output_images/example_car_noncar_hog.jpg
[image2]: ./output_images/car_hog_colorspace.jpg
[image3]: ./output_images/noncar_hog_colorspace.jpg
[image4]: ./output_images/car_hog_orient.jpg
[image5]: ./output_images/car_hog_pix_per_cell.jpg
[image6]: ./output_images/car_hog_cell_per_block.jpg
[image7]: ./output_images/pipeline1.jpg
[image8]: ./output_images/pipeline2.jpg
[video1]: ./output.mp4

## [Rubric](https://review.udacity.com/#!/rubrics/513/view) Points
###Here I will consider the rubric points individually and describe how I addressed each point in my implementation.  

---
###Writeup / README

####1. Provide a Writeup / README that includes all the rubric points and how you addressed each one.  You can submit your writeup as markdown or pdf.  [Here](https://github.com/udacity/CarND-Vehicle-Detection/blob/master/writeup_template.md) is a template writeup for this project you can use as a guide and a starting point.  

You're reading it!

###Histogram of Oriented Gradients (HOG)

####1. Explain how (and identify where in your code) you extracted HOG features from the training images.

I used skimage hog function to extract HOG features.

See function get_hog_features in file `features.py` (lines 11-28)

	# Function to return HOG features and visualization
	def get_hog_features(img, orient, pix_per_cell, cell_per_block, 
	                        vis=False, feature_vec=True):

The following image in an example visualization of HOGs for a car image and a non car image

![alt text][image1]




####2. Explain how you settled on your final choice of HOG parameters.

See main function in `features.py`

First I tries different color spaces, for randomly picked car and non car images. RGB ans YCRCb seemed to pick up the car outlines better than HLS. I chose YCrCb.

HOG features ind different color spaces for a car image.
![alt text][image2]

HOG features ind different color spaces for a non car image.
![alt text][image3]

I also tried various values of orientations, pixels per cells and cells per block. It seems using 3 orientations is too coarse to detect the shapes (see below). It is difficult to see a difference between 9 and 15 orientations from the visualisations. I pcked 9 since this reduces the feature vector size.

The [original HOG paper by Dalal & Triggs](https://lear.inrialpes.fr/people/triggs/pubs/Dalal-cvpr05.pdf) also observes that there is no performance increase by increasing the orientations beyond 9.
![alt text][image4]


WIth pixels per cell, smaller (4) size seems to pick up the fine details in shape. Larger size (16) is definitely too coarse to detect the image. We need features to be fine enough to pick out the shapes but not too fine so that the classifier can generalize. Therefore I left this parameter as default at 8.
![alt text][image5]

It is not clear from visualisations which value is best for the number of cells per block. This was left at 2x2 (the default used in lessons).
![alt text][image6]

####3. Describe how (and identify where in your code) you trained a classifier using your selected HOG features (and color features if you used them).

The feature vector was created by stacking spatial features, color histogram and HOG features. See functions extract_features_from_image(lines 49-105) and extract_features_from_image_list (lines 107-128)in `features.py`

	# Function to extract features from a single image
	# Stack spatial, color histogram and hog features in that order
	def extract_features_from_image(img, color_space='RGB', spatial_size=(32, 32),
	                        hist_bins=32, orient=9, 
	                        pix_per_cell=8, cell_per_block=2, hog_channel=0,
	                        spatial_feat=True, hist_feat=True, hog_feat=True,
	                        vis=False): 
	                        
	# Function to extract features from a list of images
	# Calls extract_features_from_image for each image in the list
	def extract_features_from_image_list(imgs, color_space='RGB', spatial_size=(32, 32),
	                        hist_bins=32, orient=9, 
	                        pix_per_cell=8, cell_per_block=2, hog_channel=0,
	                        spatial_feat=True, hist_feat=True, hog_feat=True):

Training of the classifier is done in `train_classifier.py`.

Using the above functions the features were extracted for both car and noncar training data. Then the feature vectors were normalised using sklearn StandardScaler().fit() function to improve numerical accuracy in fitting a model.

I picked a linear support vector classifier since it worked quite well during the lessons. Training the classifier was done using sklearn svc.fit() function.

	# Train a Support Vector Machine Classifier
	svc = LinearSVC()
	
	svc.fit(X_train, y_train)

The test accuracy of the classifier was arpud 99%.

>Number of Vehicle Images found: 8792
>Number of Non-Vehicle Images found: 8968
>71.9089663028717 seconds to compute features
>Using: 9 orientations 8 pixels per cell and 2 cells per block
>Feature vector length: 8460
>5.27 Seconds to train SVC...
>Test Accuracy of SVC =  0.9904

The trained classifier is saved to a file to be used in the vehicle detection pipeline.

	# Save Classifier
	
	classifer_model = {}
	
	classifer_model['svc'] = svc
	classifer_model['color_space'] = color_space
	classifer_model['orient'] = orient
	classifer_model['pix_per_cell'] = pix_per_cell
	classifer_model['cell_per_block'] = cell_per_block
	classifer_model['hog_channel'] = hog_channel
	classifer_model['spatial_size'] = spatial_size
	classifer_model['hist_bins'] = hist_bins
	classifer_model['X_scaler'] = X_scaler
	
	pickle.dump(classifer_model, open('classifer_model.pkl', 'wb'))

###Sliding Window Search

####1. Describe how (and identify where in your code) you implemented a sliding window search.  How did you decide what scales to search and how much to overlap windows?

I searched using different windows of different sizes in different areas of the image. In the foreground of the image I used large window sizes (128 by128) and at the middle of the image (where cars are the farthest)
 I searched using small windows (32 by 32). In betwenn I also used windows of 96 by 96 and 64 by 64 with suitable overlap. See `main.py` (lines 32-37)
 
	 scales = [
		{'scale':0.5, 'ystart':352, 'ystop':381},
		{'scale':1, 'ystart':384, 'ystop':639},
		{'scale':1.5, 'ystart':384, 'ystop':671},
		{'scale':2, 'ystart':464, 'ystop':719},
	]
 
The window serach was implemented by claculating HOGs for the entire image and subsampling. This avoids recalculating HOGs for overlapping windows. Spatial and color features were calculated for each window individually. This is implemented in function find_cars_in_image (lines 44-122) in `find_cars.py`.

	# Given an image return a list of windows where cars are detected
	def find_cars_in_image(img, ystart, ystop, clf, X_scaler, scale, orient, pix_per_cell, cell_per_block, spatial_size, hist_bins, color_space='RGB'):
	

####2. Show some examples of test images to demonstrate how your pipeline is working.  What did you do to optimize the performance of your classifier?

I used the training data provided by udacity, which uses cropped images from the project video frames for non-car data. (i,e, negative data mining)

See next section for example images.

---

### Video Implementation

####1. Provide a link to your final video output.  Your pipeline should perform reasonably well on the entire project video (somewhat wobbly or unstable bounding boxes are ok as long as you are identifying the vehicles most of the time with minimal false positives.)
Here's a [link to my video result](./output.mp4)


####2. Describe how (and identify where in your code) you implemented some kind of filter for false positives and some method for combining overlapping bounding boxes.

In order to reconcile multiple detections of the same car I created a heatmap using the windows where cars were detected for each frame.
Once the heatmap is created I used scipy.ndimage.measurements.label() to extract a bounding box for each set of overlapping detetions.

In order to filter out false positives, I applied a threshold for the heatmap before using the label() function.

See `heatmap.py` and lines 71,72,87 and 89 of `main.py`.

The following images shows the stages of the pipeline for the 6 test images. 

column 1: windows where cars were present. (hot windows)
column 2: heatmap created using hot windows
column 3: bounding boxes for detected cars

![alt text][image7]
![alt text][image8]

---

###D-iscussion

####1. Briefly discuss any problems / issues you faced in your implementation of this project.  Where will your pipeline likely fail?  What could you do to make it more robust?

After running my pipeline on the project video I noticed that 
- the bounding boxes for cars move around the car from frame to frame.
- in some frames cars are not detected where they were detected in the previous frames.

In order to improve on these I took a weighted average of the hetamps before extracting blobs out of it. See lines 75-87 in `main.py`.

The pipeline might fail for a video taken in a different part of the world desipite the high test accuracy of the classifier since the classifier was trained using negative data mining. In order to improve this more diverse data needs to be used.
