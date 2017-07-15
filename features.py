import numpy as np
import cv2
import matplotlib.image as mpimg
import matplotlib.pyplot as plt
from skimage.feature import hog

from utils import visualize
from utils import get_training_data

# Function to return HOG features and visualization
def get_hog_features(img, orient, pix_per_cell, cell_per_block, 
                        vis=False, feature_vec=True):
    # Call with two outputs if vis==True
    if vis == True:
        features, hog_image = hog(img, orientations=orient, 
                                  pixels_per_cell=(pix_per_cell, pix_per_cell),
                                  cells_per_block=(cell_per_block, cell_per_block), 
                                  transform_sqrt=True, 
                                  visualise=vis, feature_vector=feature_vec)
        return features, hog_image
    # Otherwise call with one output
    else:      
        features = hog(img, orientations=orient, 
                       pixels_per_cell=(pix_per_cell, pix_per_cell),
                       cells_per_block=(cell_per_block, cell_per_block), 
                       transform_sqrt=True, 
                       visualise=vis, feature_vector=feature_vec)
        return features

# Function to compute binned color features  
def get_spatial_binning_features(img, size=(32, 32)):
    # Use cv2.resize().ravel() to create the feature vector
    features = cv2.resize(img, size).ravel() 
    # Return the feature vector
    return features

# Function to compute color histogram features 
def get_color_hist_features(img, nbins=32):#, bins_range=(0, 256)):
    # Compute the histogram of the color channels separately
    channel1_hist = np.histogram(img[:,:,0], bins=nbins)#, range=bins_range)
    channel2_hist = np.histogram(img[:,:,1], bins=nbins)#, range=bins_range)
    channel3_hist = np.histogram(img[:,:,2], bins=nbins)#, range=bins_range)
    # Concatenate the histograms into a single feature vector
    hist_features = np.concatenate((channel1_hist[0], channel2_hist[0], channel3_hist[0]))
    # Return the concatanated histograms
    return hist_features


# Function to extract features from a single image
# Stack spatial, color histogram and hig features in that order
def extract_features_from_image(img, color_space='RGB', spatial_size=(32, 32),
                        hist_bins=32, orient=9, 
                        pix_per_cell=8, cell_per_block=2, hog_channel=0,
                        spatial_feat=True, hist_feat=True, hog_feat=True,
                        vis=False):    
    # Define an empty list to receive features
    img_features = []

    # Apply color conversion if other than 'RGB'
    if color_space != 'RGB':
        if color_space == 'HSV':
            feature_image = cv2.cvtColor(img, cv2.COLOR_RGB2HSV)
        elif color_space == 'LUV':
            feature_image = cv2.cvtColor(img, cv2.COLOR_RGB2LUV)
        elif color_space == 'HLS':
            feature_image = cv2.cvtColor(img, cv2.COLOR_RGB2HLS)
        elif color_space == 'YUV':
            feature_image = cv2.cvtColor(img, cv2.COLOR_RGB2YUV)
        elif color_space == 'YCrCb':
            feature_image = cv2.cvtColor(img, cv2.COLOR_RGB2YCrCb)
    else: feature_image = np.copy(img)   

    # Compute spatial features
    if spatial_feat == True:
        spatial_features = get_spatial_binning_features(feature_image, size=spatial_size)
        img_features.append(spatial_features)

    # Compute histogram features
    if hist_feat == True:
        hist_features = get_color_hist_features(feature_image, nbins=hist_bins)
        img_features.append(hist_features)

    # Compute HOG features
    if hog_feat == True:
        if hog_channel == 'ALL':
            hog_features = []
            for channel in range(feature_image.shape[2]):
                hog_features.extend(get_hog_features(feature_image[:,:,channel], 
                                    orient, pix_per_cell, cell_per_block, 
                                    vis=False, feature_vec=True))      
        else:
            if vis==True:
                hog_features, hog_image = get_hog_features(feature_image[:,:,hog_channel], orient, 
                            pix_per_cell, cell_per_block, vis=True, feature_vec=True)
            else:
                hog_features = get_hog_features(feature_image[:,:,hog_channel], orient, 
                            pix_per_cell, cell_per_block, vis=False, feature_vec=True)

        img_features.append(hog_features)

    # Return concatenated array of features
    if vis==True:
        return np.concatenate(img_features), hog_image
    else:
        return np.concatenate(img_features)

# Function to extract features from a list of images
# Calls extract_features_from_image for each image in the list
def extract_features_from_image_list(imgs, color_space='RGB', spatial_size=(32, 32),
                        hist_bins=32, orient=9, 
                        pix_per_cell=8, cell_per_block=2, hog_channel=0,
                        spatial_feat=True, hist_feat=True, hog_feat=True):

    features = []

    for file in imgs:
        image = mpimg.imread(file)

        file_features = extract_features_from_image(image, color_space=color_space, 
                            spatial_size=spatial_size, hist_bins=hist_bins, 
                            orient=orient, pix_per_cell=pix_per_cell, 
                            cell_per_block=cell_per_block, 
                            hog_channel=hog_channel, spatial_feat=spatial_feat, 
                            hist_feat=hist_feat, hog_feat=hog_feat)
        
        features.append(file_features)

    return features

if __name__ == '__main__':
    # Hyper paramaters
    
    color_space = 'RGB'
    hog_channel = 0
    orient = 9
    pix_per_cell = 8
    cell_per_block = 2
    spatial_size = (32,32)
    hist_bins = 32

    cars, noncars = get_training_data(nsamples=1)
    car_image = mpimg.imread(cars[0])
    noncar_image = mpimg.imread(noncars[0])

    hog_features, car_hog_image = get_hog_features(car_image[:,:,hog_channel], orient, 
                            pix_per_cell, cell_per_block, vis=True, feature_vec=True)

    hog_features, noncar_hog_image = get_hog_features(noncar_image[:,:,hog_channel], orient, 
                            pix_per_cell, cell_per_block, vis=True, feature_vec=True)


    images = [car_image, noncar_image, car_hog_image, noncar_hog_image]
    titles = ['car image', 'non car image', 'car hog', 'noncar hog']
    fig = plt.figure(figsize=(12,6))
    visualize(fig, 2, 2, images, titles, figname='output_images/example_car_noncar_hog.jpg')

    images = []
    titles = []

    color_space = 'RGB'
    feature_image = car_image

    hog_channel = 0
    hog_features, hog_image = get_hog_features(feature_image[:,:,hog_channel], orient, 
                            pix_per_cell, cell_per_block, vis=True, feature_vec=True)
    images.append(hog_image)
    titles.append('RGB Channel 0')

    hog_channel = 1 
    hog_features, hog_image = get_hog_features(feature_image[:,:,hog_channel], orient, 
                            pix_per_cell, cell_per_block, vis=True, feature_vec=True)
    images.append(hog_image)
    titles.append('RGB Channel 1')

    hog_channel = 2
    hog_features, hog_image = get_hog_features(feature_image[:,:,hog_channel], orient, 
                            pix_per_cell, cell_per_block, vis=True, feature_vec=True)
    images.append(hog_image)
    titles.append('RGB Channel 2')

    color_space = 'YCrCb'
    feature_image = cv2.cvtColor(car_image, cv2.COLOR_RGB2YCrCb)

    hog_channel = 0
    hog_features, hog_image = get_hog_features(feature_image[:,:,hog_channel], orient, 
                            pix_per_cell, cell_per_block, vis=True, feature_vec=True)
    images.append(hog_image)
    titles.append('YCrCb Channel 0')

    hog_channel = 1 
    hog_features, hog_image = get_hog_features(feature_image[:,:,hog_channel], orient, 
                            pix_per_cell, cell_per_block, vis=True, feature_vec=True)
    images.append(hog_image)
    titles.append('YCrCb Channel 1')

    hog_channel = 2
    hog_features, hog_image = get_hog_features(feature_image[:,:,hog_channel], orient, 
                            pix_per_cell, cell_per_block, vis=True, feature_vec=True)
    images.append(hog_image)
    titles.append('YCrCb Channel 2')

    color_space = 'HLS'
    feature_image = cv2.cvtColor(car_image, cv2.COLOR_RGB2HLS)

    hog_channel = 0
    hog_features, hog_image = get_hog_features(feature_image[:,:,hog_channel], orient, 
                            pix_per_cell, cell_per_block, vis=True, feature_vec=True)
    images.append(hog_image)
    titles.append('HLS Channel 0')

    hog_channel = 1 
    hog_features, hog_image = get_hog_features(feature_image[:,:,hog_channel], orient, 
                            pix_per_cell, cell_per_block, vis=True, feature_vec=True)
    images.append(hog_image)
    titles.append('HLS Channel 1')

    hog_channel = 2
    hog_features, hog_image = get_hog_features(feature_image[:,:,hog_channel], orient, 
                            pix_per_cell, cell_per_block, vis=True, feature_vec=True)
    images.append(hog_image)
    titles.append('HLS Channel 2')


    fig = plt.figure(figsize=(24,24))
    visualize(fig, 3, 3, images, titles, figname='output_images/car_hog_colorspace.jpg', cmap='gray')

    images = []
    titles = []

    color_space = 'RGB'
    feature_image = noncar_image

    hog_channel = 0
    hog_features, hog_image = get_hog_features(feature_image[:,:,hog_channel], orient, 
                            pix_per_cell, cell_per_block, vis=True, feature_vec=True)
    images.append(hog_image)
    titles.append('RGB Channel 0')

    hog_channel = 1 
    hog_features, hog_image = get_hog_features(feature_image[:,:,hog_channel], orient, 
                            pix_per_cell, cell_per_block, vis=True, feature_vec=True)
    images.append(hog_image)
    titles.append('RGB Channel 1')

    hog_channel = 2
    hog_features, hog_image = get_hog_features(feature_image[:,:,hog_channel], orient, 
                            pix_per_cell, cell_per_block, vis=True, feature_vec=True)
    images.append(hog_image)
    titles.append('RGB Channel 2')

    color_space = 'YCrCb'
    feature_image = cv2.cvtColor(noncar_image, cv2.COLOR_RGB2YCrCb)

    hog_channel = 0
    hog_features, hog_image = get_hog_features(feature_image[:,:,hog_channel], orient, 
                            pix_per_cell, cell_per_block, vis=True, feature_vec=True)
    images.append(hog_image)
    titles.append('YCrCb Channel 0')

    hog_channel = 1 
    hog_features, hog_image = get_hog_features(feature_image[:,:,hog_channel], orient, 
                            pix_per_cell, cell_per_block, vis=True, feature_vec=True)
    images.append(hog_image)
    titles.append('YCrCb Channel 1')

    hog_channel = 2
    hog_features, hog_image = get_hog_features(feature_image[:,:,hog_channel], orient, 
                            pix_per_cell, cell_per_block, vis=True, feature_vec=True)
    images.append(hog_image)
    titles.append('YCrCb Channel 2')

    color_space = 'HLS'
    feature_image = cv2.cvtColor(noncar_image, cv2.COLOR_RGB2HLS)

    hog_channel = 0
    hog_features, hog_image = get_hog_features(feature_image[:,:,hog_channel], orient, 
                            pix_per_cell, cell_per_block, vis=True, feature_vec=True)
    images.append(hog_image)
    titles.append('HLS Channel 0')

    hog_channel = 1 
    hog_features, hog_image = get_hog_features(feature_image[:,:,hog_channel], orient, 
                            pix_per_cell, cell_per_block, vis=True, feature_vec=True)
    images.append(hog_image)
    titles.append('HLS Channel 1')

    hog_channel = 2
    hog_features, hog_image = get_hog_features(feature_image[:,:,hog_channel], orient, 
                            pix_per_cell, cell_per_block, vis=True, feature_vec=True)
    images.append(hog_image)
    titles.append('HLS Channel 2')


    fig = plt.figure(figsize=(24,24))
    visualize(fig, 3, 3, images, titles, figname='output_images/noncar_hog_colorspace.jpg', cmap='gray')


    images = []
    titles = []

    color_space = 'YCrCb'
    feature_image = cv2.cvtColor(car_image, cv2.COLOR_RGB2YCrCb)

    orient = 3

    hog_channel = 0
    hog_features, hog_image = get_hog_features(feature_image[:,:,hog_channel], orient, 
                            pix_per_cell, cell_per_block, vis=True, feature_vec=True)
    images.append(hog_image)
    titles.append('YCrCb Channel 0 - 3 orientations')

    hog_channel = 1 
    hog_features, hog_image = get_hog_features(feature_image[:,:,hog_channel], orient, 
                            pix_per_cell, cell_per_block, vis=True, feature_vec=True)
    images.append(hog_image)
    titles.append('YCrCb Channel 1 - 3 orientations')

    hog_channel = 2
    hog_features, hog_image = get_hog_features(feature_image[:,:,hog_channel], orient, 
                            pix_per_cell, cell_per_block, vis=True, feature_vec=True)
    images.append(hog_image)
    titles.append('YCrCb Channel 2 - 3 orientations')

    orient = 9
    
    hog_channel = 0
    hog_features, hog_image = get_hog_features(feature_image[:,:,hog_channel], orient, 
                            pix_per_cell, cell_per_block, vis=True, feature_vec=True)
    images.append(hog_image)
    titles.append('YCrCb Channel 0 - 9 orientations')

    hog_channel = 1 
    hog_features, hog_image = get_hog_features(feature_image[:,:,hog_channel], orient, 
                            pix_per_cell, cell_per_block, vis=True, feature_vec=True)
    images.append(hog_image)
    titles.append('YCrCb Channel 1 - 9 orientations')

    hog_channel = 2
    hog_features, hog_image = get_hog_features(feature_image[:,:,hog_channel], orient, 
                            pix_per_cell, cell_per_block, vis=True, feature_vec=True)
    images.append(hog_image)
    titles.append('YCrCb Channel 2 - 9 orientations')

    orient = 15
    
    hog_channel = 0
    hog_features, hog_image = get_hog_features(feature_image[:,:,hog_channel], orient, 
                            pix_per_cell, cell_per_block, vis=True, feature_vec=True)
    images.append(hog_image)
    titles.append('YCrCb Channel 0 - 15 orientations')

    hog_channel = 1 
    hog_features, hog_image = get_hog_features(feature_image[:,:,hog_channel], orient, 
                            pix_per_cell, cell_per_block, vis=True, feature_vec=True)
    images.append(hog_image)
    titles.append('YCrCb Channel 1 - 15 orientations')

    hog_channel = 2
    hog_features, hog_image = get_hog_features(feature_image[:,:,hog_channel], orient, 
                            pix_per_cell, cell_per_block, vis=True, feature_vec=True)
    images.append(hog_image)
    titles.append('YCrCb Channel 2 - 15 orientations')

    fig = plt.figure(figsize=(24,24))
    visualize(fig, 3, 3, images, titles, figname='output_images/car_hog_orient.jpg', cmap='gray')

    images = []
    titles = []

    color_space = 'YCrCb'
    orient = 9
    feature_image = cv2.cvtColor(car_image, cv2.COLOR_RGB2YCrCb)

    pix_per_cell = 4

    hog_channel = 0
    hog_features, hog_image = get_hog_features(feature_image[:,:,hog_channel], orient, 
                            pix_per_cell, cell_per_block, vis=True, feature_vec=True)
    images.append(hog_image)
    titles.append('YCrCb Channel 0 - 4 pixels/cell')

    hog_channel = 1 
    hog_features, hog_image = get_hog_features(feature_image[:,:,hog_channel], orient, 
                            pix_per_cell, cell_per_block, vis=True, feature_vec=True)
    images.append(hog_image)
    titles.append('YCrCb Channel 1 - 4 pixels/cell')

    hog_channel = 2
    hog_features, hog_image = get_hog_features(feature_image[:,:,hog_channel], orient, 
                            pix_per_cell, cell_per_block, vis=True, feature_vec=True)
    images.append(hog_image)
    titles.append('YCrCb Channel 2 - 4 pixels/cell')

    pix_per_cell = 8
    
    hog_channel = 0
    hog_features, hog_image = get_hog_features(feature_image[:,:,hog_channel], orient, 
                            pix_per_cell, cell_per_block, vis=True, feature_vec=True)
    images.append(hog_image)
    titles.append('YCrCb Channel 0 - 8 pixels/cell')

    hog_channel = 1 
    hog_features, hog_image = get_hog_features(feature_image[:,:,hog_channel], orient, 
                            pix_per_cell, cell_per_block, vis=True, feature_vec=True)
    images.append(hog_image)
    titles.append('YCrCb Channel 1 - 8 pixels/cell')

    hog_channel = 2
    hog_features, hog_image = get_hog_features(feature_image[:,:,hog_channel], orient, 
                            pix_per_cell, cell_per_block, vis=True, feature_vec=True)
    images.append(hog_image)
    titles.append('YCrCb Channel 2 - 8 pixels/cell')

    pix_per_cell = 16
    
    hog_channel = 0
    hog_features, hog_image = get_hog_features(feature_image[:,:,hog_channel], orient, 
                            pix_per_cell, cell_per_block, vis=True, feature_vec=True)
    images.append(hog_image)
    titles.append('YCrCb Channel 0 - 16 pixels/cell')

    hog_channel = 1 
    hog_features, hog_image = get_hog_features(feature_image[:,:,hog_channel], orient, 
                            pix_per_cell, cell_per_block, vis=True, feature_vec=True)
    images.append(hog_image)
    titles.append('YCrCb Channel 1 - 16 pixels/cell')

    hog_channel = 2
    hog_features, hog_image = get_hog_features(feature_image[:,:,hog_channel], orient, 
                            pix_per_cell, cell_per_block, vis=True, feature_vec=True)
    images.append(hog_image)
    titles.append('YCrCb Channel 2 - 16 pixels/cell')

    fig = plt.figure(figsize=(24,24))
    visualize(fig, 3, 3, images, titles, figname='output_images/car_hog_pix_per_cell.jpg', cmap='gray')
    
    images = []
    titles = []

    color_space = 'YCrCb'
    orient = 9
    pix_per_cell = 8

    feature_image = cv2.cvtColor(car_image, cv2.COLOR_RGB2YCrCb)

    cell_per_block = 1

    hog_channel = 0
    hog_features, hog_image = get_hog_features(feature_image[:,:,hog_channel], orient, 
                            pix_per_cell, cell_per_block, vis=True, feature_vec=True)
    images.append(hog_image)
    titles.append('YCrCb Channel 0 - 1 cells/block')

    hog_channel = 1 
    hog_features, hog_image = get_hog_features(feature_image[:,:,hog_channel], orient, 
                            pix_per_cell, cell_per_block, vis=True, feature_vec=True)
    images.append(hog_image)
    titles.append('YCrCb Channel 1 - 1 cells/block')

    hog_channel = 2
    hog_features, hog_image = get_hog_features(feature_image[:,:,hog_channel], orient, 
                            pix_per_cell, cell_per_block, vis=True, feature_vec=True)
    images.append(hog_image)
    titles.append('YCrCb Channel 2 - 1 cells/block')

    cell_per_block = 2
    
    hog_channel = 0
    hog_features, hog_image = get_hog_features(feature_image[:,:,hog_channel], orient, 
                            pix_per_cell, cell_per_block, vis=True, feature_vec=True)
    images.append(hog_image)
    titles.append('YCrCb Channel 0 - 2 cells/block')

    hog_channel = 1 
    hog_features, hog_image = get_hog_features(feature_image[:,:,hog_channel], orient, 
                            pix_per_cell, cell_per_block, vis=True, feature_vec=True)
    images.append(hog_image)
    titles.append('YCrCb Channel 1 - 2 cells/block')

    hog_channel = 2
    hog_features, hog_image = get_hog_features(feature_image[:,:,hog_channel], orient, 
                            pix_per_cell, cell_per_block, vis=True, feature_vec=True)
    images.append(hog_image)
    titles.append('YCrCb Channel 2 - 2 cells/block')

    cell_per_block = 4
    
    hog_channel = 0
    hog_features, hog_image = get_hog_features(feature_image[:,:,hog_channel], orient, 
                            pix_per_cell, cell_per_block, vis=True, feature_vec=True)
    images.append(hog_image)
    titles.append('YCrCb Channel 0 - 4 cells/block')

    hog_channel = 1 
    hog_features, hog_image = get_hog_features(feature_image[:,:,hog_channel], orient, 
                            pix_per_cell, cell_per_block, vis=True, feature_vec=True)
    images.append(hog_image)
    titles.append('YCrCb Channel 1 - 4 cells/block')

    hog_channel = 2
    hog_features, hog_image = get_hog_features(feature_image[:,:,hog_channel], orient, 
                            pix_per_cell, cell_per_block, vis=True, feature_vec=True)
    images.append(hog_image)
    titles.append('YCrCb Channel 2 - 4 cells/block')

    fig = plt.figure(figsize=(24,24))
    visualize(fig, 3, 3, images, titles, figname='output_images/car_hog_cell_per_block.jpg', cmap='gray')

