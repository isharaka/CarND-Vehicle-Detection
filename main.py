import os
import glob

import numpy as np
import time
import pickle

import matplotlib.image as mpimg
import matplotlib.pyplot as plt

from sklearn.externals import joblib


from moviepy.editor import VideoFileClip

from utils import slide_window
from utils import draw_boxes
from utils import visualize

from find_cars import search_windows
from find_cars import find_cars

from heatmap import add_heat
from heatmap import apply_threshold
from heatmap import get_labeled_bboxes

def pipeline(img):

	global dist_pickle

	global svc, color_space, orient, pix_per_cell, cell_per_block, hog_channel, spatial_size, hist_bins, X_scaler

	sub_sample = True

	draw_img = np.copy(img)
	draw_img2 = np.copy(img)
	img = img.astype(np.float32)/255
	print(np.min(img), np.max(img))

	if (sub_sample==True):
		hot_windows = find_cars(img, 0, 719, svc, X_scaler, scale=1.5, orient=orient, 
					pix_per_cell=pix_per_cell, cell_per_block=cell_per_block, spatial_size=spatial_size, 
					hist_bins=hist_bins, color_space=color_space)
	else:
		windows = slide_window(img, x_start_stop=[None, None], y_start_stop=y_start_stop, 
                    xy_window=(64, 64), xy_overlap=(overlap, overlap))

		hot_windows = search_windows(img, windows, svc, X_scaler, color_space=color_space, 
                    spatial_size=spatial_size, hist_bins=hist_bins, 
                    hist_range=(0, 256), orient=orient, 
                    pix_per_cell=pix_per_cell, cell_per_block=cell_per_block, 
                    hog_channel=hog_channel, spatial_feat=True, 
                    hist_feat=True, hog_feat=True)

	window_img = draw_boxes(draw_img, hot_windows, color=(0,0,255), thick=6)

	heatmap = np.zeros_like(img[:,:,0])
	heatmap = add_heat(heatmap,hot_windows)
	heatmap = apply_threshold(heatmap,2)
		

	window_img2 = draw_boxes(draw_img2, get_labeled_bboxes(heatmap), color=(0,0,255), thick=6)

	return window_img, heatmap, window_img2 

def process_image(image):
    window_img, heatmap, window_img2 = pipeline(image)
    return window_img2

if __name__ == '__main__':

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

	sub_sample = True


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


		window_img, heatmap, window_img2 = pipeline(img)

		images.append(window_img)
		titles.append('')
		images.append(heatmap)
		titles.append('')
		images.append(window_img2)
		titles.append('')
		print(time.time()-t1, ' seconds to process one image searching ', 0, 'windows')

	fig = plt.figure()
	visualize(fig, 6, 3, images, titles)

	# Process video clip
	#output_clip = 'output.mp4'
	#input_clip = VideoFileClip("test_video.mp4")
	#clip = input_clip.fl_image(process_image) #NOTE: this function expects color images!!
	#clip.write_videofile(output_clip, audio=False)