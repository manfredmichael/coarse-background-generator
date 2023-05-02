from pathlib import Path
from rembg import remove, new_session

import cv2
import numpy as np

session = new_session()


def add_image_on_top(base_image, overlay_image):
    # Load the base image and overlay image

    # Get the dimensions of the base image
    base_height, base_width, _ = base_image.shape

    # Get the dimensions of the overlay image
    overlay_height, overlay_width, _ = overlay_image.shape

    # Calculate the center position of the base image
    center_x = int(base_width / 2)
    center_y = int(base_height / 2)

    # Calculate the position to place the overlay image in the center
    start_x = center_x - int(overlay_width / 2)
    start_y = center_y - int(overlay_height / 2)

    # Calculate the end position of the overlay image
    end_x = start_x + overlay_width
    end_y = start_y + overlay_height

    # Add the overlay image on top of the base image in the center
    base_image[start_y:end_y, start_x:end_x] = overlay_image

    # Show the image
    cv2.imshow("Result", base_image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

def resize_image(img, scale):
    width = int(img.shape[1] * scale)
    height = int(img.shape[0] * scale)
    resized_img = cv2.resize(img, (width, height))
    return resized_img

def resize_to_background(background_image, image):
    # Get the dimensions of the background image and image to be scaled
    background_height, background_width, _ = background_image.shape
    image_height, image_width, _ = image.shape

    # Calculate the scaling factors to scale the image to the background size
    scale_x = background_width / image_width
    scale_y = background_height / image_height
    scale = min(scale_x, scale_y)

    # Scale the image to the background size
    scaled_image = cv2.resize(image, None, fx=scale, fy=scale)
    # Split the 4-channel image into separate channels
    """
    b, g, r, a = cv2.split(scaled_image)
    # Merge the first 3 channels (BGR) back into a 3-channel image
    scaled_image= cv2.merge((b, g, r))


    # Calculate the position to place the scaled image in the center of the background image
    start_x = int((background_width - scaled_image.shape[1]) / 2)
    start_y = int((background_height - scaled_image.shape[0]) / 2)

    # Create a black background image with the same dimensions as the background image
    black_image = np.zeros_like(background_image)


    # Add the scaled image on top of the black background image
    background_image[start_y:start_y+scaled_image.shape[0], start_x:start_x+scaled_image.shape[1]] = scaled_image
    """
    cv2.imshow("Result", scaled_image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

    return scaled_image 

def resize_image(img1, resize_to):
    # load resized image as grayscale
    h, w, _= img1.shape

    # load background image as grayscale
    hh, ww, _ = resize_to.shape

    ratio = hh/h 
    img1 = cv2.resize(img1, (int(round(w * ratio)), int(round(h * ratio))), interpolation= cv2.INTER_LINEAR)
    
    h, w, _= img1.shape

    ratio = ww/w
    if ratio < 1:
        img1 = cv2.resize(img1, (int(round(w * ratio)), int(round(h * ratio))), interpolation= cv2.INTER_LINEAR)
    
    return img1

def insert_to_middle(back, img):
    if len(img.shape)<3:
        img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)

    # if back.shape[0] < img.shape[0] or back.shape[1] < img.shape[1]:s
    #     img = rescale(img, scale_percent=50)

    img = resize_image(img, resize_to=back)


    h, w, _= img.shape
    hh, ww, _ = back.shape
    
    
# compute xoff and yoff for placement of upper left corner of resized image   
    yoff = round(hh-h)
    xoff = round((ww-w)/2)
    if yoff<0:
        yoff=0
    if xoff<0:
        xoff=0

# use numpy indexing to place the resized image in the center of background image
    image = back.copy()
    image[yoff:yoff+h, xoff:xoff+w] = img[:hh, :ww]

    return image

def rescale(img, scale_percent):
    width = int(img.shape[1] * scale_percent / 100)
    height = int(img.shape[0] * scale_percent / 100)
    dim = (width, height)

    img = cv2.resize(img, dim, interpolation = cv2.INTER_AREA)
    
    return img

def get_mask(image, background, session):
    image = image.copy()
    background = background.copy()

    step_images = []
    
    # resize image
    # image = rescale(image, scale_percent=40)
    # background = rescale(background, scale_percent=40)
    image = resize_image(image, resize_to=background)
    image = remove(image, session=session)

    gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    b, g, r, a = cv2.split(image)
    image = cv2.merge((b, g, r))

    image = cv2.inRange(gray_image, (1), (255))

    image = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
    kernel = np.ones((3,3),np.uint8)
    image = cv2.erode(image,kernel,iterations = 2)

    return image 

def apply_mask(image, mask, virtual_background):
    image = image.copy()
    image= resize_image(image, resize_to=virtual_background)
    # image[mask<128] = [0,0,0]


    frame = np.zeros((virtual_background.shape[0], virtual_background.shape[1], 3), np.uint8) # RGBA
    image = insert_to_middle(frame, image)
    mask_resized = insert_to_middle(frame, mask)

    image[mask_resized<128] = virtual_background[mask_resized<128]
    return image

def apply_background(image, background, session=session):
    mask = get_mask(image, background, session)
    image = apply_mask(image, mask, background)

    return image

def main():
    background = cv2.imread("bg.jpg")
    overlay_image = cv2.imread("data/test/layak/DSC_1181.JPG")
    # virtual_background = rescale(background, scale_percent=40)
    mask = get_mask(overlay_image, background, session)
    image = apply_mask(overlay_image, mask, background)

    cv2.imshow('image', overlay_image)
    cv2.imshow('background', background)
    cv2.imshow('segmentation', mask)
    cv2.imshow('image', image)

    while(True):
        #Read each frame and flip it, and convert to grayscale
        if(cv2.waitKey(1) & 0xFF == ord('q')):
            break

    cv2.destroyAllWindows()


if __name__ == '__main__':
    main()
