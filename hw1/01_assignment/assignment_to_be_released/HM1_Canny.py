import numpy as np
from HM1_Convolve import Gaussian_filter, Sobel_filter_x, Sobel_filter_y, padding
from utils import read_img, write_img
from HM1_HarrisCorner import rectangle_filter

def compute_gradient_magnitude_direction(x_grad, y_grad):
    """
        The function you need to implement for Q2 a).
        Inputs:
            x_grad: array(float) 
            y_grad: array(float)
        Outputs:
            magnitude_grad: array(float)
            direction_grad: array(float) you may keep the angle of the gradient at each pixel
    """
    magnitude_grad = np.sqrt(x_grad**2 + y_grad**2)
    magnitude_grad = magnitude_grad / np.max(magnitude_grad)
    direction_grad = np.arctan2(y_grad, x_grad)
    return magnitude_grad, direction_grad 



def non_maximal_suppressor(grad_mag, grad_dir):
    """
        The function you need to implement for Q2 b).
        Inputs:
            grad_mag: array(float) 
            grad_dir: array(float)
        Outputs:
            output: array(float)
    """   
    grad_mag = padding(grad_mag, 1, "replicatePadding")
    grad_dir = padding(grad_dir, 1, "replicatePadding")
    NMS_output = np.zeros_like(grad_mag)
    grad_dir = grad_dir * 180 / np.pi
    # NMS_output[1:-1, 1:-1] = grad_mag[1:-1, 1:-1]
    grad_dir = np.mod(grad_dir, 180)
    east_west = (grad_dir <= 22.5) | (grad_dir > 157.5)
    north_south = (grad_dir > 67.5) & (grad_dir <= 112.5)
    northeast_southwest = (grad_dir > 22.5) & (grad_dir <= 67.5)
    northwest_southeast = (grad_dir > 112.5) & (grad_dir <= 157.5)
    
    NMS_output[1:-1, 1:-1][east_west[1:-1, 1:-1]] = np.where(
        (grad_mag[1:-1, 1:-1] > grad_mag[1:-1, :-2]) & (grad_mag[1:-1, 1:-1] > grad_mag[1:-1, 2:]),
        grad_mag[1:-1, 1:-1],
        0)[east_west[1:-1, 1:-1]]
    NMS_output[1:-1, 1:-1][north_south[1:-1, 1:-1]] = np.where(
        (grad_mag[1:-1, 1:-1] > grad_mag[:-2, 1:-1]) & (grad_mag[1:-1, 1:-1] > grad_mag[2:, 1:-1]),
        grad_mag[1:-1, 1:-1],
        0)[north_south[1:-1, 1:-1]]
    NMS_output[1:-1, 1:-1][northeast_southwest[1:-1, 1:-1]] = np.where(
        (grad_mag[1:-1, 1:-1] > grad_mag[:-2, :-2]) & (grad_mag[1:-1, 1:-1] > grad_mag[2:, 2:]),
        grad_mag[1:-1, 1:-1],
        0)[northeast_southwest[1:-1, 1:-1]]
    NMS_output[1:-1, 1:-1][northwest_southeast[1:-1, 1:-1]] = np.where(
        (grad_mag[1:-1, 1:-1] > grad_mag[2:, :-2]) & (grad_mag[1:-1, 1:-1] > grad_mag[:-2, 2:]),
        grad_mag[1:-1, 1:-1],
        0)[northwest_southeast[1:-1, 1:-1]]

    NMS_output = NMS_output[1:-1, 1:-1]
    return NMS_output 
            


def hysteresis_thresholding(img) :
    """
        The function you need to implement for Q2 c).
        Inputs:
            img: array(float) 
        Outputs:
            output: array(float)
    """

    #you can adjust the parameters to fit your own implementation 
    img = img > 0.15
    low_ratio = 0.1
    high_ratio = 0.3
    max_value = high_ratio * np.average(img)
    min_value = low_ratio * np.average(img)
    img = padding(img, 1, "zeroPadding")
    output = np.zeros_like(img)
    for i in range(1, img.shape[0]-1):
        for j in range(1, img.shape[1]-1):
            if img[i, j] > max_value:
                output[i, j] = img[i, j]
            elif img[i, j] < min_value:
                output[i, j] = 0
            else:
                if np.any(img[i-1:i+1, j-1:j+1] > max_value):
                    output[i, j] = img[i, j]
                else:
                    output[i, j] = 0
    return output[1:-1, 1:-1]



if __name__=="__main__":

    #Load the input images
    input_img = read_img("Lenna.png")/255
    #Apply gaussian blurring
    blur_img = Gaussian_filter(input_img)
    x_grad = Sobel_filter_x(blur_img)
    y_grad = Sobel_filter_y(blur_img)

    #Compute the magnitude and the direction of gradient
    magnitude_grad, direction_grad = compute_gradient_magnitude_direction(x_grad, y_grad)

    #NMS
    NMS_output = non_maximal_suppressor(magnitude_grad, direction_grad)
    # NMS_output = magnitude_grad

    #Edge linking with hysteresis
    output_img = hysteresis_thresholding(NMS_output)
    
    write_img("result/HM1_Canny_result.png", output_img*255)
