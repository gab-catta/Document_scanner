from skimage.filters import threshold_local
import numpy as np
import argparse
import cv2
import imutils
import matplotlib.pyplot as plt
import os


# four point ordering 
def order_points(points):
	# initialzie a list of coordinates of the corners, top-left, top-right, bottom-right, bottom-left
	ordered_points = np.zeros((4, 2),dtype="float32")
	# the top-left point will have the smallest x+y, the bottom-right point will have the largest x+y
	s = np.sum(points,axis=1)
	ordered_points[0] = points[np.argmin(s)]
	ordered_points[2] = points[np.argmax(s)]
	# the top-right point will have the smallest x-y, the bottom-left will have the largest x-y
	diff = np.diff(points, axis = 1)
	ordered_points[1] = points[np.argmin(diff)]
	ordered_points[3] = points[np.argmax(diff)]
	# return the ordered coordinates
	return ordered_points

# four point perspective transform
def four_point_transform(image, points):
	# obtain a consistent order of the points and unpack them
	# individually
	ordered_points = order_points(points)
	ordered_points = np.array(ordered_points, dtype="float32")  
	(tl, tr, br, bl) = ordered_points
	# image's width, which will be the maximum distance between bottom-right and bottom-left x-coordinates or the top-right and top-left x-coordinates
	widthA = np.sqrt(((br[0] - bl[0]) ** 2) + ((br[1] - bl[1]) ** 2))
	widthB = np.sqrt(((tr[0] - tl[0]) ** 2) + ((tr[1] - tl[1]) ** 2))
	maxWidth = max(int(widthA), int(widthB))
	# height of the new image, which will be the maximum distance between the top-right and bottom-right y-coordinates or the top-left and bottom-left y-coordinates
	heightA = np.sqrt(((tr[0] - br[0]) ** 2) + ((tr[1] - br[1]) ** 2))
	heightB = np.sqrt(((tl[0] - bl[0]) ** 2) + ((tl[1] - bl[1]) ** 2))
	maxHeight = max(int(heightA), int(heightB))
	# set of destination points to obtain a "birds eye view",(i.e. top-down view) of the image
	# top-left, top-right, bottom-right, and bottom-left order
	transf_ordered_points = np.array([
		[0, 0],
		[maxWidth - 1, 0],
		[maxWidth - 1, maxHeight - 1],
		[0, maxHeight - 1]],dtype="float32")
	# creating perspective transform matrix and then apply it to the image
	M = cv2.getPerspectiveTransform(ordered_points, transf_ordered_points)
	warped = cv2.warpPerspective(image, M, (maxWidth, maxHeight))

	return warped


def find_contour(edged,image):
	# define contours of the edged image
	cnts = cv2.findContours(edged.copy(), cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
	cnts = imutils.grab_contours(cnts)
	# keeping only the largest contours 
	cnts = sorted(cnts, key=cv2.contourArea, reverse=True)[:5]
	doc_cnt = None
	max_area = 0
	for c in cnts:
		# approximate the contour
		peri = cv2.arcLength(c, True)
		approx = cv2.approxPolyDP(c, 0.02 * peri, True)
		# if our approximated contour has four points, and is the largest so far
		if len(approx) == 4:
			area = cv2.contourArea(approx)
			if area > max_area:
				doc_cnt = approx
				max_area = area
	if doc_cnt is not None:
		# show the contour (outline) of the piece of paper
		cv2.drawContours(image, [doc_cnt], -1, (0, 255, 0), 2)  # Draw the contour in green
		plt.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
		plt.title("Outline")
		plt.axis('off')
		plt.show()
		return doc_cnt
	else:
		print("No document contour found!")
		return None
        

def get_unique_filename(base_name):
    if not os.path.exists(base_name):
        return base_name
    name, ext = os.path.splitext(base_name)
    i = 1
    while True:
        new_name = f"{name}_{i}{ext}"
        if not os.path.exists(new_name):
            return new_name
        i += 1

if __name__ == "__main__":
	# construct the argument parser and parse the arguments
	ap = argparse.ArgumentParser()
	ap.add_argument("-i", "--image", required = True,
		help = "Path to the image to be scanned")
	args = vars(ap.parse_args())

	# load the image and save the ratio of the old height compare to the new height, clone it, and resize it
	image = cv2.imread(args["image"])
	print("image shape = ",image.shape)
	ratio = image.shape[0] / 500.0
	orig = image.copy()
	image = imutils.resize(image, height = 500)
	print("STEP 1: Detecting Edges of the document......")
	# convert the image to grayscale, blur it, and find edges in the image
	gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
	gray = cv2.GaussianBlur(gray, (5, 5), 0)
	edged = cv2.Canny(gray, 75, 200)
	# show the original image and the edge detected image
	plt.figure(figsize=(10,5))
	plt.subplot(1,2,1)
	plt.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
	plt.title("Image")
	plt.axis('off')
	plt.subplot(1,2,2)
	plt.imshow(edged, cmap='gray')
	plt.title("Edged")
	plt.axis('off')
	plt.show()
	
	print("STEP 2: Finding contours of the document......")
	doc_cont = find_contour(edged,image)	
	if doc_cont is None:
		print("Please, ensure that the document is well-lit and isolated from the background. Exiting...")
		exit(1)

	print("STEP 3: Transforming the document into scan........")
	# apply the four point transform to obtain a top-down view of the original image
	warped = four_point_transform(orig, doc_cont.reshape(4, 2) * ratio)
	# warped image to grayscale, threshold it to give it that 'black and white' paper effect
	warped = cv2.cvtColor(warped, cv2.COLOR_BGR2GRAY)
	# Sharpen the image before thresholding
	blurred = cv2.GaussianBlur(warped, (0, 0), 3)
	sharpened = cv2.addWeighted(warped, 1.5, blurred, -0.5, 0)
	# Adaptive thresholding using skimage
	T = threshold_local(sharpened, block_size=9, offset=9, method="gaussian")
	# show the original and scanned images
	plt.figure(figsize=(10, 5))
	plt.subplot(1, 2, 1)
	plt.imshow(cv2.cvtColor(imutils.resize(orig, height=500), cv2.COLOR_BGR2RGB))
	plt.title("Original")
	plt.axis('off')
	plt.subplot(1, 2, 2)
	plt.imshow(warped, cmap='gray')
	plt.title("Scanned")
	plt.axis('off')
	plt.show()

	print("/-----------------------------------------/")
	print("If the scanned image doesn't look good, make sure that the 4 corners of the document are well visible.")
	print("/-----------------------------------------/")

	print("STEP 4: Saving the scanned image........")
	while True:
		check = input("Are you satisfied with the result? (y/n): ")
		if check.lower() == 'y':
			path_orig_image = args["image"]
			base = os.path.basename(path_orig_image)
			name, _ = os.path.splitext(base)
			scanned_name = f"scanned_{name}.png"
			filename = get_unique_filename(scanned_name)
			cv2.imwrite(filename, warped)
			print(f"Scanned image saved as '{filename}'.")
			break
		elif check.lower() == 'n':
			print("Please try again with a better image.")
			break
		else:
			print("Invalid input. Exiting...")
		
