# Doc Scanner

Doc Scanner is a Python script that transforms a photo of a document into a clean, top-down scanned version. It uses OpenCV, scikit-image, and matplotlib to detect the document edges, apply a perspective transform, and enhance the result for readability.


## Features
- Automatically detects the largest quadrilateral in the image (the document).
- Applies a perspective transform to obtain a "scanned" top-down view.
- Converts the result to black and white for a clean, paper-like effect.
- Saves the scanned image with a unique filename based on the original image name.

## Requirements

- Python 3.7 or newer
- OpenCV (`opencv-python`)
- scikit-image
- imutils
- matplotlib
- numpy

You can install the dependencies with:

```sh
pip install opencv-python scikit-image imutils matplotlib numpy
```

## Usage

Run the script from the command line, providing the path to your image:

```sh
python scanner.py -i path/to/your/image.jpg
```

The script will:

1. Show the original and edge-detected images.
2. Display the detected document contour.
3. Show the final scanned result.
4. Ask if you are satisfied with the result before saving.

If a file named `scanned_originalname.png` already exists, the script will save the new scan as `scanned_originalname_1.png`, `scanned_originalname_2.png`, etc.

## Tips for Best Results

- Ensure the document is well-lit and isolated from the background.
- Make sure all four corners are visible in the photo.
- Make sure that the document doesn't is not wrinkled.  
If the scan is not satisfying enough, try retaking the photo or adjusting the lighting.

