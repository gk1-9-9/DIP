import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt
import os

def histogram_equalization(img_path):
    # Read the image in grayscale
    img = cv.imread(img_path, cv.IMREAD_GRAYSCALE)
    
    # Check for invalid input
    if img is None:
        raise ValueError(f"Image at path {img_path} could not be loaded.")

    # Calculate histogram and CDF of the original image
    hist = cv.calcHist([img], [0], None, [256], [0, 256])
    cdf = hist.cumsum()
    cdf_normalized = cdf * float(hist.max() / cdf.max())

    # Plot original image, histogram, and CDF
    plt.figure(figsize=(12, 6))
    
    plt.subplot(231)
    plt.imshow(img, cmap='gray')
    plt.title('Original Image')

    plt.subplot(234)
    plt.plot(hist)
    plt.plot(cdf_normalized, color='b')
    plt.xlabel('Pixel Intensity')
    plt.ylabel('Pixel Count')
    plt.title('Original Histogram and CDF')

    # Perform histogram equalization
    equ_img = cv.equalizeHist(img)

    # Ensure the directory exists for saving the equalized image
    output_dir = 'equalized_images'
    os.makedirs(output_dir, exist_ok=True)
    
    # Save the equalized image
    equalized_img_path = os.path.join(output_dir, 'Lena_equalized.jpg')
    cv.imwrite(equalized_img_path, equ_img)

    # Calculate histogram and CDF of the equalized image
    equ_hist = cv.calcHist([equ_img], [0], None, [256], [0, 256])
    equ_cdf = equ_hist.cumsum()
    equ_cdf_normalized = equ_cdf * float(equ_hist.max() / equ_hist.max())

    # Plot equalized image, histogram, and CDF
    plt.subplot(232)
    plt.imshow(equ_img, cmap='gray')
    plt.title('Equalized Image')

    plt.subplot(235)
    plt.plot(equ_hist)
    plt.plot(equ_cdf_normalized, color='b')
    plt.xlabel('Pixel Intensity')
    plt.ylabel('Pixel Count')
    plt.title('Equalized Histogram and CDF')

    plt.tight_layout()
    plt.show()

if __name__ == '__main__':
    img_path = './images/Lena.png'
    histogram_equalization(img_path)
