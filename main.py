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

from find_cars import find_cars_in_windows
from find_cars import find_cars_in_image

from heatmap import add_heat
from heatmap import apply_threshold
from heatmap import get_labeled_bboxes

smoothing_enabled = True
heatmap_buffer = []

HEATMAP_BUFFER_SIZE = 20

scales = [
	{'scale':0.5, 'ystart':352, 'ystop':381},
	{'scale':1, 'ystart':384, 'ystop':639},
	{'scale':1.5, 'ystart':384, 'ystop':671},
	{'scale':2, 'ystart':464, 'ystop':719},
]

def pipeline(img):
	global svc, color_space, orient, pix_per_cell, cell_per_block, hog_channel, spatial_size, hist_bins, X_scaler
	global smoothing_enabled

	sub_sample = True

	draw_img = np.copy(img)
	draw_img2 = np.copy(img)
	heatmap = np.zeros_like(img[:,:,0])

	img = img.astype(np.float32)/255

	if (sub_sample==True):
		hot_windows = []
		for i in range(len(scales)):
			hot_windows_for_scale = find_cars_in_image(img, scales[i]['ystart'], scales[i]['ystop'], svc, X_scaler, scale=scales[i]['scale'], orient=orient, 
					pix_per_cell=pix_per_cell, cell_per_block=cell_per_block, spatial_size=spatial_size, 
					hist_bins=hist_bins, color_space=color_space)
			hot_windows = hot_windows + hot_windows_for_scale
	else:
		windows = slide_window(img, x_start_stop=[None, None], y_start_stop=y_start_stop, 
                    xy_window=(64, 64), xy_overlap=(overlap, overlap))

		hot_windows = find_cars_in_windows(img, windows, svc, X_scaler, color_space=color_space, 
                    spatial_size=spatial_size, hist_bins=hist_bins, 
                    hist_range=(0, 256), orient=orient, 
                    pix_per_cell=pix_per_cell, cell_per_block=cell_per_block, 
                    hog_channel=hog_channel, spatial_feat=True, 
                    hist_feat=True, hog_feat=True)

	img_hot_windows = draw_boxes(draw_img, hot_windows, color=(0,0,255), thick=6)
	
	heatmap = add_heat(heatmap,hot_windows)
	heatmap = apply_threshold(heatmap,1)

	if (smoothing_enabled==True):
		if(len(heatmap_buffer) >= HEATMAP_BUFFER_SIZE):
			del heatmap_buffer[0]

		heatmap_buffer.append(heatmap)

		smooth_heatmap = np.zeros_like(img[:,:,0])
		total_weights = 0;
		for i in range(len(heatmap_buffer)):
			smooth_heatmap += ((i+1) * heatmap_buffer[i])
			total_weights += (i+1)
		smooth_heatmap = smooth_heatmap / total_weights

		car_windows = get_labeled_bboxes(smooth_heatmap)	
	else:
		car_windows = get_labeled_bboxes(heatmap)


	img_detected_cars = draw_boxes(draw_img2, car_windows, color=(0,0,255), thick=6)

	return img_hot_windows, heatmap, img_detected_cars, len(hot_windows), len(car_windows) 

def process_image(image):
    img_hot_windows, heatmap, img_detected_cars, no_hot_windows, no_cars = pipeline(image)
    return img_detected_cars

if __name__ == '__main__':

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

	smoothing_enabled = False

	for img_src in example_images:
		t1 = time.time()
		
		img_hot_windows, heatmap, img_detected_cars, no_hot_windows, no_cars = pipeline(mpimg.imread(img_src))

		images.append(img_hot_windows)
		titles.append("hot windows "+img_src)
		images.append(heatmap)
		titles.append("heat map "+img_src)
		images.append(img_detected_cars)
		titles.append("detections "+img_src)

		print(time.time()-t1, ' seconds to process one image. ', no_hot_windows, 'hot windows.', no_cars, ' cars detected.')

	fig = plt.figure(figsize=(24,16))
	visualize(fig, 3, 3, images[0:9], titles[0:9], 'output_images/pipeline1.jpg')
	fig = plt.figure(figsize=(24,16))
	visualize(fig, 3, 3, images[9:18], titles[9:18], 'output_images/pipeline2.jpg')

	smoothing_enabled = True

	# Process video clip
	output_clip = 'output.mp4'
	input_clip = VideoFileClip("project_video.mp4")
	clip = input_clip.fl_image(process_image) #NOTE: this function expects color images!!
	clip.write_videofile(output_clip, audio=False)