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
    num_bins = 72
    template_histograms = []
    rotated_histograms = []
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
        template_lines = line_detection(template_edges, 1, np.pi/180, 70, 50, 10)
        rotated_lines = line_detection(rotated_edges, 1, np.pi/180, 70, 50, 10)

        # Draw the lines on the images
        # print("i: ", i)
        # print(len(template_lines))
        # print(len(rotated_lines))
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

        
if __name__ == "__main__":
    main()