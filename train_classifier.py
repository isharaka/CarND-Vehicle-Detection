import os
import glob

import matplotlib.image as mpimg
import matplotlib.pyplot as plt
import numpy as np
import cv2
import time
import pickle

#from sklearn.model_selection import train_test_split
from sklearn.model_selection import train_test_split
from sklearn.svm import LinearSVC
from sklearn.preprocessing import StandardScaler
from sklearn.externals import joblib

from features import extract_features_from_image_list

from utils import slide_window
from utils import draw_boxes
from utils import visualize
from utils import get_training_data

from find_cars import find_cars_in_windows

if __name__ == '__main__':

	# Hyper paramaters
	color_space = 'YCrCb'
	orient = 9
	pix_per_cell = 8
	cell_per_block = 2
	hog_channel = 'ALL' # 0,12,'ALL'
	spatial_size = (32,32)
	hist_bins = 32

	# Read training data
	train_cars, train_notcars = get_training_data()

	t = time.time()

	# Extract Features
	car_features = extract_features_from_image_list(train_cars, color_space=color_space, spatial_size=spatial_size,
	                        hist_bins=hist_bins, orient=orient, 
	                        pix_per_cell=pix_per_cell, cell_per_block=cell_per_block, hog_channel=hog_channel,
	                        spatial_feat=True, hist_feat=True, hog_feat=True)

	notcar_features = extract_features_from_image_list(train_notcars, color_space=color_space, spatial_size=spatial_size,
	                        hist_bins=hist_bins, orient=orient, 
	                        pix_per_cell=pix_per_cell, cell_per_block=cell_per_block, hog_channel=hog_channel,
	                        spatial_feat=True, hist_feat=True, hog_feat=True)

	print(time.time()-t, 'seconds to compute features')

	X = np.vstack((car_features, notcar_features)).astype(np.float64)  

	# Normalise features
	X_scaler = StandardScaler().fit(X)
	scaled_X = X_scaler.transform(X)

	# Define the labels vector
	y = np.hstack((np.ones(len(car_features)), np.zeros(len(notcar_features))))


	# Split up data into randomized training and test sets
	rand_state = np.random.randint(0, 100)
	X_train, X_test, y_train, y_test = train_test_split(
	    scaled_X, y, test_size=0.2, random_state=rand_state)

	print('Using:',orient,'orientations',pix_per_cell,
	    'pixels per cell and', cell_per_block,'cells per block')
	print('Feature vector length:', len(X_train[0]))

	# Train a Support Vector Machine Classifier
	svc = LinearSVC()

	t=time.time()

	svc.fit(X_train, y_train)

	t2 = time.time()

	print(round(t2-t, 2), 'Seconds to train SVC...')
	print('Test Accuracy of SVC = ', round(svc.score(X_test, y_test), 4))

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


	# Test Classifier

	classifer_model = pickle.load( open('classifer_model.pkl', 'rb' ) )

	svc = classifer_model['svc']
	color_space = classifer_model['color_space']
	orient = classifer_model['orient']
	pix_per_cell = classifer_model['pix_per_cell'] 
	cell_per_block = classifer_model['cell_per_block']
	hog_channel = classifer_model['hog_channel'] 
	spatial_size = classifer_model['spatial_size']
	hist_bins = classifer_model['hist_bins']
	X_scaler = classifer_model['X_scaler']



	t=time.time()

	example_images = glob.glob('test_images/*')

	images = []
	titles = []
	y_start_stop = [None, None]
	overlap = 0.5

	for img_src in example_images:
		t1 = time.time()
		img = mpimg.imread(img_src)
		draw_img = np.copy(img)
		img = img.astype(np.float32)/255
		print(np.min(img), np.max(img))

		windows = slide_window(img, x_start_stop=[None, None], y_start_stop=y_start_stop, 
                    xy_window=(128, 128), xy_overlap=(overlap, overlap))

		hot_windows = find_cars_in_windows(img, windows, svc, X_scaler, color_space=color_space, 
                    spatial_size=spatial_size, hist_bins=hist_bins, 
                    hist_range=(0, 256), orient=orient, 
                    pix_per_cell=pix_per_cell, cell_per_block=cell_per_block, 
                    hog_channel=hog_channel, spatial_feat=True, 
                    hist_feat=True, hog_feat=True)

		window_img = draw_boxes(draw_img, hot_windows, color=(0,0,255), thick=6)
		images.append(window_img)
		titles.append('')
		print(time.time()-t1, ' seconds to process one image searching ', len(windows), 'windows')

	fig = plt.figure()
	visualize(fig, 3, 2, images, titles)

