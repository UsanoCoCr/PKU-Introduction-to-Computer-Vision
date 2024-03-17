import numpy as np
from utils import  read_img, draw_corner
from HM1_Convolve import convolve, Sobel_filter_x,Sobel_filter_y,padding



def corner_response_function(input_img, window_size, alpha, threshold):
    """
        The function you need to implement for Q3.
        Inputs:
            input_img: array(float)
            window_size: int
            alpha: float
            threshold: float
        Outputs:
            corner_list: array
    """
    # please solve the corner_response_function of each window,
    # and keep windows with theta > threshold.
    # you can use several functions from HM1_Convolve to get 
    # I_xx, I_yy, I_xy as well as the convolution result.
    # for details of corner_response_function, please refer to the slides.
    I_x = Sobel_filter_x(input_img)
    I_y = Sobel_filter_y(input_img)
    I_xx = I_x**2
    I_yy = I_y**2
    I_xy = I_x*I_y    

    def rectangle_filter(img, window_size):
        padding_img = padding(img, 1, "zeroPadding")
        rectangle_kernel = np.ones((window_size, window_size))
        output = convolve(padding_img, rectangle_kernel)
        return output

    I_xx = rectangle_filter(I_xx, window_size)
    I_yy = rectangle_filter(I_yy, window_size)
    I_xy = rectangle_filter(I_xy, window_size)

    det_M = I_xx*I_yy - I_xy**2
    trace_M = I_xx + I_yy
    theta = det_M - alpha * (trace_M**2) - threshold
    theta[theta <= 0] = 0
    corner_x, corner_y = np.where(theta > 0)
    corner_list = np.array([corner_x, corner_y, theta[corner_x, corner_y]]).T 
    return corner_list # array, each row contains information about one corner, namely (index of row, index of col, theta)



if __name__=="__main__":

    #Load the input images
    input_img = read_img("hand_writting.png")/255.

    #you can adjust the parameters to fit your own implementation 
    window_size = 5
    alpha = 0.04
    threshold = 10

    corner_list = corner_response_function(input_img,window_size,alpha,threshold)

    # NMS
    corner_list_sorted = sorted(corner_list, key = lambda x: x[2], reverse = True)
    NML_selected = [] 
    NML_selected.append(corner_list_sorted[0][:-1])
    dis = 10
    for i in corner_list_sorted :
        for j in NML_selected :
            if(abs(i[0] - j[0]) <= dis and abs(i[1] - j[1]) <= dis) :
                break
        else :
            NML_selected.append(i[:-1])


    #save results
    draw_corner("hand_writting.png", "result/HM1_HarrisCorner.png", NML_selected)
