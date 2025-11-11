import cv2


def convert_to_grayscale(input_path, output_path):
    """
    Converts an image to grayscale.

    Args:
        input_path (str): Path to the input image.
        output_path (str): Path to save the grayscale image.
    """
    # Read the image
    image = cv2.imread(input_path)

    # Check if image was loaded successfully
    if image is None:
        print("Error: Could not open or find the image.")
        return

    # Convert the image to grayscale
    gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Save the grayscale image
   # cv2.imwrite(output_path, gray_image)
   # print(f"Grayscale image saved to: {output_path}")

    # (Optional) Display the images
    cv2.imshow("Original Image", image)
    cv2.imshow("Grayscale Image", gray_image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


input_image = "project_data/train/image_data/10001.jpg"  # Path to your input image
output_image = "grayscale.jpg"  # Path to save the grayscale image
#convert_to_grayscale(input_image, output_image)
convert_to_grayscale(input_image, output_image)
