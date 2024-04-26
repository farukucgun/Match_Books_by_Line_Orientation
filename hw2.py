import cv2
import numpy as np
import matplotlib.pyplot as plt
import os
import itertools


########### GLOBAL VARIABLES ###########

template_images = []
rotated_images = []
template_image_names = ["algorithms.png", "bitmemisoykuler.png", "kpss.png", "cinali.png", "cpp.png", "datamining.png", "harrypotter.png", "ters.png",
                   "heidi.png", "chess.png", "lordofrings.png", "patternrecognition.png", "sefiller.png", "shawshank.png", "stephenking.png",]

# rotated image names are the same with "R" added to the end of the names
rotated_image_names = template_image_names.copy()

########################################


def generate_rotated_image_array():
    for i in range(len(rotated_image_names)):
        rotated_image_names[i] = template_image_names[i].split(".")[0] + "R.png"


def canny_edge_detection(image, low_threshold=100, high_threshold=200):
    return cv2.Canny(image, low_threshold, high_threshold)


def line_detection(image, rho=1, theta=np.pi/180, threshold=50, min_line_length=50, max_line_gap=10):
    return cv2.HoughLinesP(image, rho, theta, threshold, np.array([]), minLineLength=min_line_length, maxLineGap=max_line_gap)


def compute_orientation_histogram(lines, num_bins=36):
    # Initialize the histogram with zeros
    histogram = np.zeros(num_bins)
    # Define the bin edges to cover the range from -π to π
    bin_edges = np.linspace(-np.pi, np.pi, num_bins + 1)

    if lines is None:
        return histogram
    
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


def circular_shift(hist, shift):
    return np.roll(hist, shift)


def find_best_match(rotated_hist, original_hists):
    min_distance = np.inf
    best_match_index = -1
    best_shift = 0
    
    # For each bin shift
    for shift in range(len(rotated_hist)):
        shifted_hist = circular_shift(rotated_hist, shift)
        
        # Compare with each original histogram
        for idx, original_hist in enumerate(original_hists):
            distance = np.linalg.norm(shifted_hist - original_hist)
            
            if distance < min_distance:
                min_distance = distance
                best_match_index = idx
                best_shift = shift
                
    return best_match_index, best_shift


def main(num_bins, canny_low_threshold, canny_high_threshold, hough_threshold, hough_min_line_length, hough_max_line_gap):
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
    num_bins = num_bins
    template_histograms = []
    rotated_histograms = []

    for i in range(len(template_image_names)):
        template_edges = canny_edge_detection(template_images[i], canny_low_threshold, canny_high_threshold)
        rotated_edges = canny_edge_detection(rotated_images[i], canny_low_threshold, canny_high_threshold)

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
        template_lines = line_detection(template_edges, 1, np.pi/180, hough_threshold, hough_min_line_length, hough_max_line_gap)
        rotated_lines = line_detection(rotated_edges, 1, np.pi/180, hough_threshold, hough_min_line_length, hough_max_line_gap)

        # Draw the lines on the images
        if template_lines is not None:
            for line in template_lines:
                x1, y1, x2, y2 = line[0]
                cv2.line(template_images[i], (x1, y1), (x2, y2), (0, 255, 0), 2)

        if rotated_lines is not None:
            for line in rotated_lines:
                x1, y1, x2, y2 = line[0]
                cv2.line(rotated_images[i], (x1, y1), (x2, y2), (0, 255, 0), 2)

        # Save the images with the lines drawn
        plt.figure()
        plt.imshow(template_images[i], cmap="gray")
        plt.axis("off")
        plt.savefig("lines/template_lines/" + template_image_names[i])
        plt.close()

        plt.figure()
        plt.imshow(rotated_images[i], cmap="gray")
        plt.axis("off")
        plt.savefig("lines/rotated_lines/" + rotated_image_names[i])
        plt.close()

        # Compute the orientation histograms for the lines
        template_histogram = compute_orientation_histogram(template_lines, num_bins)
        template_histograms.append(template_histogram)
        rotated_histogram = compute_orientation_histogram(rotated_lines, num_bins)
        rotated_histograms.append(rotated_histogram)

        plt.figure()
        plt.bar(np.arange(len(template_histogram)), template_histogram)
        plt.savefig("histograms/template_histogram/" + template_image_names[i])
        plt.close()

        plt.figure()
        plt.bar(np.arange(len(rotated_histogram)), rotated_histogram)
        plt.savefig("histograms/rotated_histogram/" + rotated_image_names[i])
        plt.close()

    # Match each rotated histogram to the best original histogram and estimate the rotation angle
    matches = []
    rotation_angles = []
    angle_per_bin = 360 / num_bins

    for rotated_hist in rotated_histograms:
        match_index, shift = find_best_match(rotated_hist, template_histograms)
        matches.append(match_index)
        rotation_angle = shift * angle_per_bin
        rotation_angles.append(rotation_angle)

    # Print the match results and estimated rotation angles
    accuracy = 0
    for i, (match_index, rotation_angle) in enumerate(zip(matches, rotation_angles)):
        if i == match_index:
            accuracy += 1
        print(f"Rotated Book {i} matches with Original Book {match_index} with an estimated rotation of {rotation_angle} degrees.")

    print(f"Accuracy: {accuracy / len(matches) * 100:.2f}%")

    # with open("results2.txt", "a") as f:
    #     f.write(f"Accuracy: {accuracy / len(matches) * 100:.2f}%\n\n")

        
if __name__ == "__main__":
    # Best Results:
    # Parameters: (36, (200, 300), 50, 50, 20)
    # Accuracy: 80.00%
    # num_bins, low_threshold, high_threshold, hough_threshold, hough_min_line_length, hough_max_line_gap
    main(36, 200, 300, 50, 50, 20)

    # Grid Search

    # num_bins_range = [24, 36, 72, 144]
    # canny_thresholds = [(100, 200), (150, 250), (200, 300)]
    # hough_threshold_range = [30, 50, 70, 90]
    # hough_min_line_length_range = [50, 70, 90]
    # hough_max_line_gap_range = [10, 20, 30]

    # param_grid = itertools.product(
    #     num_bins_range, 
    #     canny_thresholds, 
    #     hough_threshold_range,
    #     hough_min_line_length_range, 
    #     hough_max_line_gap_range
    # )

    # for params in param_grid:
    #     with open("results2.txt", "a") as f:
    #         f.write(f"Parameters: {params}\n")

    #     num_bins, (low_threshold, high_threshold), hough_threshold, hough_min_line_length, hough_max_line_gap = params
    #     main(num_bins, low_threshold, high_threshold, hough_threshold, hough_min_line_length, hough_max_line_gap)