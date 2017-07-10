import os
import glob

import matplotlib.image as mpimg
import matplotlib.pyplot as plt
import numpy as np
import cv2
import time
import pickle

from sklearn.model_selection import train_test_split
from sklearn.svm import LinearSVC
from sklearn.preprocessing import StandardScaler
from sklearn.externals import joblib

from features import single_img_features
from features import extract_features

from utils import slide_window
from utils import draw_boxes
from utils import visualize

from find_cars import search_windows

if __name__ == '__main__':
	basedir = 'data/vehicles/vehicles/'

	image_types = os.listdir(basedir)
	cars = []

	for imtype in image_types:
		cars.extend(glob.glob(basedir+imtype+'/*'))

	print('Number of Vehicle Images found:',len(cars))

	with open("cars.txt", 'w') as f:
		for fn in cars:
			f.write(fn+'\n')

	basedir = 'data/non-vehicles/non-vehicles/'

	image_types = os.listdir(basedir)
	notcars = []

	for imtype in image_types:
		notcars.extend(glob.glob(basedir+imtype+'/*'))

	print('Number of Non-Vehicle Images found:',len(notcars))

	with open("notcars.txt", 'w') as f:
		for fn in notcars:
			f.write(fn+'\n')

	color_space = 'YCrCb'
	orient = 9
	pix_per_cell = 8
	cell_per_block = 2
	hog_channel = 'ALL' # 0,12,'ALL'
	spatial_size = (32,32)
	hist_bins = 32

	t = time.time()
	n_samples = 1000
	random_idxs = np.random.randint(0, len(cars), n_samples)
	train_cars = cars #np.array(cars)[random_idxs]
	train_notcars = notcars #np.array(notcars)[random_idxs]

	car_features = extract_features(train_cars, color_space=color_space, spatial_size=spatial_size,
	                        hist_bins=hist_bins, orient=orient, 
	                        pix_per_cell=pix_per_cell, cell_per_block=cell_per_block, hog_channel=hog_channel,
	                        spatial_feat=True, hist_feat=True, hog_feat=True)

	notcar_features = extract_features(train_notcars, color_space=color_space, spatial_size=spatial_size,
	                        hist_bins=hist_bins, orient=orient, 
	                        pix_per_cell=pix_per_cell, cell_per_block=cell_per_block, hog_channel=hog_channel,
	                        spatial_feat=True, hist_feat=True, hog_feat=True)

	print(time.time()-t, 'seconds to compute features')

	X = np.vstack((car_features, notcar_features)).astype(np.float64)                        
	# Fit a per-column scaler
	X_scaler = StandardScaler().fit(X)
	# Apply the scaler to X
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

	# Use a linear SVC 
	svc = LinearSVC()
	# Check the training time for the SVC
	t=time.time()
	svc.fit(X_train, y_train)
	t2 = time.time()
	print(round(t2-t, 2), 'Seconds to train SVC...')
	# Check the score of the SVC
	print('Test Accuracy of SVC = ', round(svc.score(X_test, y_test), 4))

	dist_pickle = {}

	dist_pickle['svc'] = svc
	dist_pickle['color_space'] = color_space
	dist_pickle['orient'] = orient
	dist_pickle['pix_per_cell'] = pix_per_cell
	dist_pickle['cell_per_block'] = cell_per_block
	dist_pickle['hog_channel'] = hog_channel
	dist_pickle['spatial_size'] = spatial_size
	dist_pickle['hist_bins'] = hist_bins
	dist_pickle['X_scaler'] = X_scaler

	pickle.dump(dist_pickle, open('dist_pickle.pkl', 'wb'))

	dist_pickle = pickle.load( open('dist_pickle.pkl', 'rb' ) )

	svc = dist_pickle['svc']
	color_space = dist_pickle['color_space']
	orient = dist_pickle['orient']
	pix_per_cell = dist_pickle['pix_per_cell'] 
	cell_per_block = dist_pickle['cell_per_block']
	hog_channel = dist_pickle['hog_channel'] 
	spatial_size = dist_pickle['spatial_size']
	hist_bins = dist_pickle['hist_bins']
	X_scaler = dist_pickle['X_scaler']



	#joblib.dump(svc, 'svc.pkl') 

	#svc = joblib.load('svc.pkl')


	# Check the prediction time for a single sample
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

		hot_windows = search_windows(img, windows, svc, X_scaler, color_space=color_space, 
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

