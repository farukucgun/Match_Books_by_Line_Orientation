"""
The goal of this assignment is to match objects by comparing their line patterns. In particular, you will
try to match books according to the line orientation histograms computed from the images of their covers.
You are given a data set of books in which each book has two images.
One of them is the original image of the book cover, and the other one is a rotated version of the cover
with respect to an arbitrary angle between 0 and 180 degrees. The goal is to find a match between the
original and rotated books, and then, to find the angle of rotation. 

template images are in the folder "template_images"
rotated images are in the folder "rotated_images"


The approach can be summarised in terms of the following steps.
• The data:Half of the images belong to the original books, and the remaining belong to the rotated ones.
• Perform edge detection: run the Canny edge detector on each image after converting it to a greyscale
image. The output of this step is a binary edge image that marks pixels that represent a significant
change. You have to experiment with the parameters to obtain important edges that will be useful in
the following steps. After evaluating different parameter values for a subset of the data set, you must
fix them and use the same values for all images.
• Perform line fitting to find line segments: find straight line segments in the edge detector outputs using
Hough transform. After performing the Hough transform, you can find the bins that accumulated the
most points in the Hough array. You can play with the minimum number of peaks to find a reasonable
number of lines. Fig. 4 shows some examples.
• Compute line orientation histograms: the orientation values are typically in the range [−π,+π]. You
can divide this range into uniform bins, and compute a histogram of line orientations weighted by line
lengths. That is, a line should contribute to its corresponding bin by its length instead of just 1. You
have to experiment with the number of bins to find a good representation.
• Find a match and compute the angle of rotation for each rotated book: in this step, your task is to
find the original book of each rotated book using the orientation histograms. Rotated book histograms
can be considered as shifted versions of the corresponding original book histograms. The number of
bins in a rotated book histogram that has to be shifted to match the corresponding original book
histogram should be approximately proportional to the angle of rotation. Hence, the angle of rotation
can be estimated by finding how many bins the second histogram has to be shifted so that the two
orientation histograms are similar. The original book of each rotated book can be found by shifting the
rotated book histogram to one bin at each iteration and calculating the Euclidean distances between
the shifted histogram and the original book histograms. As a result, the original book that results in
the minimum Euclidean distance can be a reasonable match. Also, the amount of shift needed can
be used to find the approximate angle of rotation. Note that you should use circular shifts when you
compare the histograms.
• Discuss your results: you should discuss the results of each step and compare how the performance
is affected by different parameters (i.e., Canny parameters, Hough transform parameters, number of
bins).

Submission:
• A report (pdf file) that includes:
– The results for edge detection for different parameters. You can use edge detection code from
other sources but you must cite the source that you used.
– The results for line detection in which detected lines are overlayed on the original images. You
can use Hough transform code from other sources but you must cite the source that you used.
– Example line orientation histograms. You must provide results for different numbers of bins.
– Results for matching the rotated books to the original books as well as the estimated rotation
angles. You must provide a result for each book, i.e., 15 results. You can provide additional
results for different parameter settings.
• A well-documented script that runs the particular sequence of operations and reproduces the result
presented in your report for computing the line orientation histogram using the line detection results
and for matching the line orientation histograms for different books using circular shifts and Euclidean
distances.

"""

import cv2
import numpy as np
import matplotlib.pyplot as plt
import os


template_images = []
rotated_images = []
template_image_names = ["algorithms.png", "bitmemisoykuler.png", "kpss.png", "cinali.png", "cpp.png", "datamining.png", "harrypotter.png", "ters.png",
                   "heidi.png", "chess.png", "lordofrings.png", "patternrecognition.png", "sefiller.png", "shawshank.png", "stephenking.png",]

# rotated image names are the same with "R" added to the end of the names
rotated_image_names = template_image_names.copy()


def generate_rotated_image_array():
    for i in range(len(rotated_image_names)):
        rotated_image_names[i] = rotated_image_names[i].split(".")[0] + "R.png"


def two_d_convolution(image, filter):
    # Get dimensions of the image and the filter
    image_height, image_width = image.shape
    filter_height, filter_width = filter.shape

    # Calculate padding size for the image
    pad_height = filter_height // 2
    pad_width = filter_width // 2

    # Create an empty output image
    output_image = np.zeros((image_height, image_width), dtype=np.uint8)

    # Iterate over each pixel in the image
    for i in range(pad_height, image_height - pad_height):
        for j in range(pad_width, image_width - pad_width):
            # Fit the filter in the image at this pixel
            area_of_interest = image[i-pad_height:i+pad_height+1, j-pad_width:j+pad_width+1]

            # Perform the convolution operation
            output_image[i, j] = np.sum(area_of_interest * filter)

    return output_image


def sobel_filters(img):
    Kx = np.array([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], np.float32)
    Ky = np.array([[1, 2, 1], [0, 0, 0], [-1, -2, -1]], np.float32)
    
    Ix = two_d_convolution(img, Kx)
    Iy = two_d_convolution(img, Ky)
    
    G = np.hypot(Ix, Iy)
    G = G / G.max() * 255
    theta = np.arctan2(Iy, Ix)
    
    return (G, theta)


def canny_edge_detection(image, low_threshold=100, high_threshold=200):
    return cv2.Canny(image, low_threshold, high_threshold)


def line_detection(image, rho=1, theta=np.pi/180, threshold=50, min_line_length=50, max_line_gap=10):
    return cv2.HoughLinesP(image, rho, theta, threshold, np.array([]), minLineLength=min_line_length, maxLineGap=max_line_gap)


def compute_orientation_histogram(lines, num_bins=36):
    # Initialize the histogram with zeros
    histogram = np.zeros(num_bins)
    # Define the bin edges to cover the range from -π to π
    bin_edges = np.linspace(-np.pi, np.pi, num_bins + 1)

    for line in lines:
        x1, y1, x2, y2 = line[0]
        # Calculate orientation and length of the line segment
        orientation = np.arctan2(y2 - y1, x2 - x1)
        length = np.hypot(x2 - x1, y2 - y1)
        
        # Find the correct bin for the orientation
        bin_index = np.digitize(orientation, bin_edges) - 1  # Subtract 1 for correct bin index
        # Add the length to the bin
        histogram[bin_index] += length

    return histogram


def match_rotated_books():
    pass


def main():
    # Create the directories if they do not exist
    directories = ["edges", "edges/template_edges", "edges/rotated_edges", "lines", "lines/template_lines", "lines/rotated_lines",
                   "histograms", "histograms/template_histogram", "histograms/rotated_histogram"]
    
    for directory in directories:
        if not os.path.exists(directory):
            os.makedirs(directory)

    # Generate the rotated image names
    generate_rotated_image_array()

    # Load the images as grayscale
    for i in range(len(template_image_names)):
        template_images.append(cv2.imread("template_images/" + template_image_names[i], cv2.IMREAD_GRAYSCALE))
        rotated_images.append(cv2.imread("rotated_images/" + rotated_image_names[i], cv2.IMREAD_GRAYSCALE))

    # Apply Canny edge detection to the images to get the edges
    num_bins = 36
    for i in range(len(template_image_names)):
        template_edges = canny_edge_detection(template_images[i], 100, 200)
        rotated_edges = canny_edge_detection(rotated_images[i], 100, 200)

        plt.figure()
        plt.imshow(template_edges, cmap="gray")
        plt.axis("off")
        plt.savefig("edges/template_edges/" + template_image_names[i])
        plt.close()

        plt.figure()
        plt.imshow(rotated_edges, cmap="gray")
        plt.axis("off")
        plt.savefig("edges/rotated_edges/" + rotated_image_names[i])
        plt.close()
    
        # Apply Hough transform to the images to find the lines
        # image, rho=1, theta=np.pi/180, threshold=50, min_line_length=50, max_line_gap=10
        template_lines = line_detection(template_edges, 1, np.pi/180, 50, 50, 10)
        rotated_lines = line_detection(rotated_edges, 1, np.pi/180, 50, 50, 10)

        # Draw the lines on the images
        print("i: ", i)
        print(len(template_lines))
        print(len(rotated_lines))
        for line in template_lines:
            x1, y1, x2, y2 = line[0]
            cv2.line(template_images[i], (x1, y1), (x2, y2), (0, 255, 0), 1)

        for line in rotated_lines:
            x1, y1, x2, y2 = line[0]
            cv2.line(rotated_images[i], (x1, y1), (x2, y2), (0, 255, 0), 1)

        # Save the images with the lines drawn
        plt.figure()
        plt.imshow(template_images[i], cmap="gray")
        plt.axis("off")
        plt.savefig("lines/template_lines/" + template_image_names[i])
        plt.close()

        plt.figure()
        plt.imshow(rotated_images[i] , cmap="gray")
        plt.axis("off")
        plt.savefig("lines/rotated_lines/" + rotated_image_names[i])
        plt.close()

        # Compute the orientation histograms for the lines
        template_histogram = compute_orientation_histogram(template_lines, num_bins)
        rotated_histogram = compute_orientation_histogram(rotated_lines, num_bins)

        plt.figure()
        plt.bar(np.arange(len(template_histogram)), template_histogram)
        plt.savefig("histograms/template_histogram/" + template_image_names[i])
        plt.close()

        plt.figure()
        plt.bar(np.arange(len(rotated_histogram)), rotated_histogram)
        plt.savefig("histograms/rotated_histogram/" + rotated_image_names[i])
        plt.close()
                
        
if __name__ == "__main__":
    main()