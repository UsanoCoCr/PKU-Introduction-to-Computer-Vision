import numpy as np
from utils import read_img, write_img

def padding(img, padding_size, type):
    """
        The function you need to implement for Q1 a).
        Inputs:
            img: array(float)
            padding_size: int
            type: str, zeroPadding/replicatePadding
        Outputs:
            padding_img: array(float)
    """
    if type=="zeroPadding":
        padding_img = np.zeros((img.shape[0]+2*padding_size, img.shape[1]+2*padding_size))
        padding_img[padding_size:padding_size+img.shape[0], padding_size:padding_size+img.shape[1]] = img
        return padding_img
    elif type=="replicatePadding":
        padding_img = np.zeros((img.shape[0]+2*padding_size, img.shape[1]+2*padding_size))
        padding_img[padding_size:padding_size+img.shape[0], padding_size:padding_size+img.shape[1]] = img
        padding = np.full((padding_size, img.shape[1]), img[0,:])
        padding_img[:padding_size, padding_size:padding_size+img.shape[1]] = padding
        padding = np.full((padding_size, img.shape[1]), img[-1,:])
        padding_img[padding_size+img.shape[0]:, padding_size:padding_size+img.shape[1]] = padding
        padding = np.full((img.shape[0]+2*padding_size, padding_size), padding_img[:,padding_size:padding_size+1])
        padding_img[:, :padding_size] = padding
        padding = np.full((img.shape[0]+2*padding_size, padding_size), padding_img[:,padding_size+img.shape[1]-1:padding_size+img.shape[1]])
        padding_img[:, padding_size+img.shape[1]:] = padding
        return padding_img

def convol_with_Toeplitz_matrix(img, kernel):
    """
        The function you need to implement for Q1 b).
        Inputs:
            img: array(float) 6*6
            kernel: array(float) 3*3
        Outputs:
            output: array(float)
    """
    #zero padding
    padding_img = padding(img, 1, "zeroPadding")
    #build the Toeplitz matrix and compute convolution
    from scipy.linalg import toeplitz
    h = kernel.reshape(-1, 1)
    H1 = toeplitz(np.concatenate((h[0].reshape(-1), [0,0,0,0,0])), np.concatenate((h[0:3].reshape(-1), [0,0,0,0,0])))
    H2 = toeplitz(np.concatenate((h[3].reshape(-1), [0,0,0,0,0])), np.concatenate((h[3:6].reshape(-1), [0,0,0,0,0])))
    H3 = toeplitz(np.concatenate((h[6].reshape(-1), [0,0,0,0,0])), np.concatenate((h[6:9].reshape(-1), [0,0,0,0,0])))
    H_zero = np.zeros((6, 8))
    row1 = np.hstack((H1, H2, H3, H_zero, H_zero, H_zero, H_zero, H_zero))
    row2 = np.hstack((H_zero, H1, H2, H3, H_zero, H_zero, H_zero, H_zero))
    row3 = np.hstack((H_zero, H_zero, H1, H2, H3, H_zero, H_zero, H_zero))
    row4 = np.hstack((H_zero, H_zero, H_zero, H1, H2, H3, H_zero, H_zero))
    row5 = np.hstack((H_zero, H_zero, H_zero, H_zero, H1, H2, H3, H_zero))
    row6 = np.hstack((H_zero, H_zero, H_zero, H_zero, H_zero, H1, H2, H3))
    H = np.vstack((row1, row2, row3, row4, row5, row6))
    output = np.dot(H, padding_img.reshape(-1, 1)).reshape(6, 6)
    return output


def convolve(img, kernel):
    """
        The function you need to implement for Q1 c).
        Inputs:
            img: array(float)
            kernel: array(float)
        Outputs:
            output: array(float)
    """
    
    #build the sliding-window convolution here
    img_size_row, img_size_col = img.shape
    kernel_size_row, kernel_size_col = kernel.shape
    output_size_row, output_size_col = img_size_row-kernel_size_row+1, img_size_col-kernel_size_col+1
    output = np.zeros((output_size_row, output_size_col))
    
    bound_row, bound_col = np.meshgrid(np.arange(output_size_row), np.arange(output_size_col))
    inside_row, inside_col = np.meshgrid(np.arange(kernel_size_row), np.arange(kernel_size_col))
    window = img[bound_row + inside_row.reshape(-1, 1, 1), bound_col + inside_col.reshape(-1, 1, 1)]
    window = window.reshape(kernel_size_row*kernel_size_col, -1).T
    # print(window)
    h = kernel.T.reshape(-1, 1)
    # print(h)
    output = np.dot(window, h).reshape(output_size_col, output_size_row).T
    # output = np.array([11,21,31,12,22,32,13,23,33,14,24,34]).reshape(output_size_col, output_size_row).T
    # print(output)
    return output


def Gaussian_filter(img):
    padding_img = padding(img, 1, "replicatePadding")
    gaussian_kernel = np.array([[1/16,1/8,1/16],[1/8,1/4,1/8],[1/16,1/8,1/16]])
    output = convolve(padding_img, gaussian_kernel)
    return output

def Sobel_filter_x(img):
    padding_img = padding(img, 1, "replicatePadding")
    sobel_kernel_x = np.array([[-1,0,1],[-2,0,2],[-1,0,1]])
    output = convolve(padding_img, sobel_kernel_x)
    return output

def Sobel_filter_y(img):
    padding_img = padding(img, 1, "replicatePadding")
    sobel_kernel_y = np.array([[-1,-2,-1],[0,0,0],[1,2,1]])
    output = convolve(padding_img, sobel_kernel_y)
    return output



if __name__=="__main__":

    np.random.seed(111)
    input_array=np.random.rand(6,6)
    # input_array = np.array([[1,2,3,4,5,6],[7,8,9,10,11,12],[13,14,15,16,17,18],[19,20,21,22,23,24],[25,26,27,28,29,30]])
    input_kernel=np.random.rand(3,3)
    # input_kernel = np.array([[1,2,3],[4,5,6],[7,8,9]])

    # task1: padding
    zero_pad =  padding(input_array,1,"zeroPadding")
    np.savetxt("result/HM1_Convolve_zero_pad.txt",zero_pad)

    replicate_pad = padding(input_array,1,"replicatePadding")
    np.savetxt("result/HM1_Convolve_replicate_pad.txt",replicate_pad)


    #task 2: convolution with Toeplitz matrix
    result_1 = convol_with_Toeplitz_matrix(input_array, input_kernel)
    np.savetxt("result/HM1_Convolve_result_1.txt", result_1)

    # task 3: convolution with sliding-window
    result_2 = convolve(input_array, input_kernel)
    np.savetxt("result/HM1_Convolve_result_2.txt", result_2)

    #task 4/5: Gaussian filter and Sobel filter
    input_img = read_img("lenna.png")/255

    img_gadient_x = Sobel_filter_x(input_img)
    img_gadient_y = Sobel_filter_y(input_img)
    img_blur = Gaussian_filter(input_img)

    write_img("result/HM1_Convolve_img_gadient_x.png", img_gadient_x*255)
    write_img("result/HM1_Convolve_img_gadient_y.png", img_gadient_y*255)
    write_img("result/HM1_Convolve_img_blur.png", img_blur*255)
