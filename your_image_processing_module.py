import cv2
import os
import numpy as np
import pandas as pd
from skimage.feature import local_binary_pattern
from skimage.feature import graycomatrix
from sklearn.decomposition import PCA

def extract_features(image_path):
    # Read the input image
    image = cv2.imread(image_path)
    gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Texture Features - GLCM
    glcm = graycomatrix(gray_image, [5], [0], 256, symmetric=True, normed=True)
    contrast = np.mean((glcm[:, :, 0, 0] * (np.arange(256) - np.arange(256).mean())**2).sum())
    dissimilarity = np.mean(glcm[:, :, 0, 0])
    homogeneity = np.mean(glcm[:, :, 0, 0] / (1 + (np.arange(256) - np.arange(256).mean())**2))
    energy = np.mean(glcm[:, :, 0, 0]**2)
    correlation = np.mean((glcm[:, :, 0, 0] * (np.arange(256) - np.arange(256).mean())).sum() / (np.sqrt(np.var(np.arange(256))) * np.sqrt(np.var(np.arange(256)))))
    glcm_features = {'Contrast': [contrast], 'Dissimilarity': [dissimilarity], 'Homogeneity': [homogeneity], 'Energy': [energy], 'Correlation': [correlation]}

    # LBP Features
    radius = 3
    n_points = 8 * radius
    lbp_image = local_binary_pattern(gray_image, n_points, radius, method='uniform')
    
    # Calculate the mean and standard deviation of LBP features
    lbp_mean = np.mean(lbp_image)
    lbp_sd = np.std(lbp_image)
    
    lbp_features = {'LBP_Mean': [lbp_mean], 'LBP_SD': [lbp_sd]}
    
    # Apply PCA on LBP features
    lbp_pca = PCA(n_components=1)
    lbp_pca_result = lbp_pca.fit_transform(lbp_image.reshape(-1, 1))
    lbp_features['LBP_PCA'] = [lbp_pca_result[0, 0]]

    # Gabor Features
    gabor_kernels = cv2.getGaborKernel((5, 5), 3.0, np.pi/4, 10.0, 0.5, 0, ktype=cv2.CV_32F)
    gabor_filtered = cv2.filter2D(gray_image, cv2.CV_8UC3, gabor_kernels)
    gabor_energy = np.mean(gabor_filtered**2)
    gabor_features = {'Gabor Energy': [gabor_energy]}

    # Shape Features
    contours, _ = cv2.findContours(gray_image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if contours:  # Ensure contours were found
        largest_contour = max(contours, key=cv2.contourArea)
        contour_area = cv2.contourArea(largest_contour)
        perimeter = cv2.arcLength(largest_contour, closed=True)
        circularity = (4 * np.pi * contour_area) / perimeter**2 if perimeter > 0 else 0

        # Fit ellipse and calculate Aspect Ratio and Eccentricity
        if len(contours[0]) >= 5:
            ellipse = cv2.fitEllipse(largest_contour)
            major_axis, minor_axis = max(ellipse[1]), min(ellipse[1])
            aspect_ratio = major_axis / minor_axis if minor_axis != 0 else 0  # Check for division by zero
            eccentricity = np.sqrt(1 - (minor_axis / major_axis)**2)
        else:
            aspect_ratio = 0
            eccentricity = 0
            
        # Solidity and Convexity
        hull = cv2.convexHull(largest_contour)
        hull_area = cv2.contourArea(hull)
        solidity = contour_area / hull_area if hull_area > 0 else 0 
        convexity = contour_area / hull_area if hull_area > 0 else 0
        
        # Calculate Hu Moments
        moments = cv2.moments(largest_contour)
        hu_moments = cv2.HuMoments(moments).flatten()
        
        # Apply PCA on Hu Moments
        hu_pca = PCA(n_components=1)
        hu_pca_result = hu_pca.fit_transform(hu_moments.reshape(1, -1))
        hu_features = {'Hu_PCA': [hu_pca_result[0, 0]]}

        # Calculate edge features (e.g., count of edge pixels using Canny)
        edges = cv2.Canny(image, 100, 200)
        edge_count = np.sum(edges > 0)

        # Calculate contour features (curvature, convexity, and number of corners)
        curvature = 1.0 / cv2.arcLength(largest_contour, closed=True)
        num_corners = len(cv2.approxPolyDP(largest_contour, 0.02 * cv2.arcLength(largest_contour, closed=True), closed=True))

        # Extent
        bounding_box = cv2.boundingRect(contours[0])
        extent = contour_area / (bounding_box[2] * bounding_box[3]) if (bounding_box[2] * bounding_box[3]) > 0 else 0

        # Combine all features into a dictionary
        all_features = {
            
            'Contrast': contrast,
            'Dissimilarity': dissimilarity,
            'Homogeneity': homogeneity,
            'Energy': energy,
            'Correlation': correlation,
            'LBP_Mean': lbp_mean, 
            'LBP_SD': lbp_sd,
            'LBP_PCA': lbp_pca_result[0, 0],
            'Gabor Energy': gabor_energy,
            'Contour Area': contour_area,
            'Perimeter': perimeter,
            'Circularity': circularity,
            'Aspect Ratio': aspect_ratio,
            'Eccentricity': eccentricity,
            'Solidity': solidity,
            'Convexity': convexity,
            'Extent': extent,
            "Hu_PCA": hu_pca_result[0, 0],
            "Edge Count": edge_count,
            "Curvature": curvature,
            "Num Corners": num_corners
        }

        # Create a DataFrame from the dictionary
        # df_features = pd.DataFrame(all_features, index=[0])
        # print(df_features)
        features_array = np.array(list(all_features.values()))
        print(features_array)
        return features_array
        
    else:
        print(f"No contours found in {image_path}")
        return None

  