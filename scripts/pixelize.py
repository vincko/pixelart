import cv2
import numpy as np
from sklearn.cluster import KMeans
import argparse
from collections import Counter
from termcolor import colored

def find_dominant_colors(image, k):
    # Reshape the image to be a list of pixels
    pixels = image.reshape(-1, 3)
    
    # Perform k-means clustering
    kmeans = KMeans(n_clusters=k)
    kmeans.fit(pixels)
    
    # Get the cluster centers (dominant colors)
    colors = kmeans.cluster_centers_
    labels = kmeans.labels_
    
    return colors, labels

def most_common_color(square, colors):
    # Calculate the distance of each pixel to each of the dominant colors
    pixels = square.reshape(-1, 3)
    distances = np.linalg.norm(pixels[:, None] - colors[None, :], axis=2)
    
    # Find the closest dominant color for each pixel
    closest_color = np.argmin(distances, axis=1)
    
    # Return the most common color
    return Counter(closest_color).most_common(1)[0][0]

def rgb_to_ansi(r, g, b):
    return f"\033[48;2;{r};{g};{b}m   \033[0m"

def is_bright_color(color):
    # Calculate the perceived brightness of the color
    r, g, b = color
    brightness = 0.299 * r + 0.587 * g + 0.114 * b
    return brightness > 127.5  # Midpoint of 0-255 range

def add_legend(image, colors, color_count, height, width):
    # Calculate the size of the legend
    legend_height = 50
    legend_width = width
    
    # Create a white strip for the legend
    legend = np.ones((legend_height, legend_width, 3), dtype=np.uint8) * 255
    
    # Define circle radius and font settings
    circle_radius = 10
    border_thickness = 2
    font_scale = 0.5
    font_thickness = 1
    offset = 30
    
    # Iterate through the colors and add them to the legend
    for idx, count in color_count.items():
        color = colors[idx].astype(int)
        position = (offset + idx * 80, legend_height // 2)
        
        # Draw the color circle
        cv2.circle(legend, position, circle_radius + border_thickness, (0,0,0), -1)
        cv2.circle(legend, position, circle_radius, color.tolist(), -1)
        
        # Add the count text next to the circle
        text_position = (position[0] + circle_radius + 10, position[1] + 5)
        text_color = (0, 0, 0)
        cv2.putText(legend, str(count), text_position, cv2.FONT_HERSHEY_SIMPLEX, 
                    font_scale, text_color, font_thickness)
    
    # Concatenate the legend at the bottom of the image
    combined_image_with_legend = np.vstack((image, legend))
    return combined_image_with_legend

def process_image(image_path, k):
    # Load the image
    image = cv2.imread(image_path)
    height, width, _ = image.shape
    
    # Calculate the size of each square
    square_height = height // 16
    square_width = width // 16
    
    # Find the dominant colors
    colors, _ = find_dominant_colors(image, k)
    colors = colors.astype(int)
    
    # Prepare to count the colors and create the output image
    color_count = Counter()
    output_image = np.zeros((16, 16, 3), dtype=np.uint8)
    bright_text_image = np.zeros((16*square_height, 16*square_width, 3), dtype=np.uint8)
    dark_text_image = np.zeros((16*square_height, 16*square_width, 3), dtype=np.uint8)

    # Divide the image into 16 rows and 16 columns
    for y in range(16):
        previous_color_idx = None
        sequence_count = 0
        for x in range(16):
            start_y = y * square_height
            start_x = x * square_width
            end_y = start_y + square_height if y < 15 else height
            end_x = start_x + square_width if x < 15 else width
            square = image[start_y:end_y, start_x:end_x]
            dominant_color_idx = most_common_color(square, colors)
            color_count[dominant_color_idx] += 1

            if dominant_color_idx == previous_color_idx:
                sequence_count += 1
            else:
                if previous_color_idx is not None:
                    text_color = (255, 255, 255)
                    text_image = dark_text_image if is_bright_color(colors[previous_color_idx]) else bright_text_image

                    # Write the sequence count in the first square of the sequence
                    cv2.putText(text_image, str(sequence_count), (first_x * square_width + 5, y * square_height + 15), 
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, text_color, 1)
                
                # Start a new sequence
                first_x = x
                sequence_count = 1
                previous_color_idx = dominant_color_idx
            
            # Set the output image square to the dominant color
            output_image[y, x] = colors[dominant_color_idx]

    # Handle the last sequence in the row
        if previous_color_idx is not None:
            # Choose text color based on brightness
            text_color = (255, 255, 255)
            text_image = dark_text_image if is_bright_color(colors[previous_color_idx]) else bright_text_image

            cv2.putText(text_image, str(sequence_count), (first_x * square_width + 5, y * square_height + 15), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, text_color, 1)

    for idx, count in color_count.items():
        b, g, r = colors[idx] # colors is BGR
        color_display = rgb_to_ansi(r, g, b)
        print(f"Color {color_display}: {count} squares")
    
    # Resize the output image to visualize better
    resized_output = cv2.resize(output_image, (16*square_width, 16*square_height), interpolation=cv2.INTER_NEAREST)
    combined_output = cv2.addWeighted(cv2.subtract(resized_output, dark_text_image), 1, bright_text_image, 1, 0)
    combined_output_with_legend = add_legend(combined_output, colors, color_count, 16*square_height, 16*square_width)

    # Add black grid lines to the resized image
    for i in range(1, 16):
        # Draw horizontal lines
        cv2.line(combined_output_with_legend, (0, i * square_height), (16*square_width, i * square_height), (0, 0, 0), 1)
        # Draw vertical lines
        cv2.line(combined_output_with_legend, (i * square_width, 0), (i * square_width, 16*square_height), (0, 0, 0), 1)
    
    # Save the output image
    cv2.imwrite("out.png", output_image)
    print(f"16x16 output at ./out.png")
    cv2.imwrite("out-scaled.png", combined_output_with_legend)
    print(f"Instruction output at ./out-scaled.png")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Process an image to find dominant colors and create a 16x16 image showing the most common color in each grid square.')
    parser.add_argument('image_path', type=str, help='Path to the input image.')
    parser.add_argument('k', type=int, help='Number of dominant colors to find.')
    
    args = parser.parse_args()
    
    process_image(args.image_path, args.k)