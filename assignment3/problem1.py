import numpy as np
from scipy.ndimage import convolve, maximum_filter

import matplotlib.pyplot as plt

def gauss2d(sigma, fsize):
    """ Create a 2D Gaussian filter

    Args:
        sigma: width of the Gaussian filter
        fsize: (w, h) dimensions of the filter
    Returns:
        *normalized* Gaussian filter as (h, w) np.array
    """
    m, n = fsize
    x = np.arange(-m / 2 + 0.5, m / 2)
    y = np.arange(-n / 2 + 0.5, n / 2)
    xx, yy = np.meshgrid(x, y, sparse=True)
    g = np.exp(-(xx ** 2 + yy ** 2) / (2 * sigma ** 2))
    return g / np.sum(g)


def derivative_filters():
    """ Create derivative filters for x and y direction

    Returns:
        fx: derivative filter in x direction
        fy: derivative filter in y direction
    """
    fx = np.array([[0.5, 0, -0.5]])
    fy = fx.transpose()
    return fx, fy


def compute_hessian(img, gauss, fx, fy):
    """ Compute elements of the Hessian matrix

    Args:
        img:
        gauss: Gaussian filter
        fx: derivative filter in x direction
        fy: derivative filter in y direction

    Returns:
        I_xx: (h, w) np.array of 2nd derivatives in x direction
        I_yy: (h, w) np.array of 2nd derivatives in y direction
        I_xy: (h, w) np.array of 2nd derivatives in x-y direction
    """
    #print(gauss.shape)
    #print(fx)
    #print()
    #print(fy)
    #print(fx*fy)
    
    #Use the gaussion filter to smooth the image
    g = convolve(img,gauss, mode = 'mirror')
    # Use the derivative filter to get the first derivative
    Ix = convolve(g,fx, mode = 'mirror')
    Iy = convolve(g,fy, mode = 'mirror')
    # Use the derivative filter again to get the second derivative
    Ixx = convolve(Ix,fx, mode = 'mirror')
    Iyy = convolve(Iy,fy, mode = 'mirror')
    Ixy = convolve(Ix,fy, mode = 'mirror')
    Iyx = convolve(Iy,fx, mode = 'mirror')
    """
    fig =plt.figure()
    fig.add_subplot(4,2,1)
    plt.imshow(img)
    
    fig.add_subplot(4,2,2)
    plt.imshow(g)
    
    fig.add_subplot(4,2,3)
    plt.imshow(Ix)
    
    fig.add_subplot(4,2,4)
    plt.imshow(Iy)
    
    fig.add_subplot(4,2,5)
    plt.imshow(Ixx)
    
    fig.add_subplot(4,2,6)
    plt.imshow(Iyy)
    
    fig.add_subplot(4,2,7)
    plt.imshow(Ixy)
    
    fig.add_subplot(4,2,8)
    plt.imshow(Iyx)
    
    plt.show()
    """
    return Ixx,Iyy,Ixy
    #
    # You code here
    #


def compute_criterion(I_xx, I_yy, I_xy, sigma):
    """ Compute criterion function

    Args:
        I_xx: (h, w) np.array of 2nd derivatives in x direction
        I_yy: (h, w) np.array of 2nd derivatives in y direction
        I_xy: (h, w) np.array of 2nd derivatives in x-y direction
        sigma: scaling factor

    Returns:
        criterion: (h, w) np.array of scaled determinant of Hessian matrix
    """
    
    criterion = (I_xx*I_yy - np.power(I_xy,2) )* sigma**4
    #print(np.abs(criterion[0:2][0:4]))
    #print(criterion[0:2][0:4])
    #plt.imshow(np.abs(criterion) )
    #plt.show()
    #return np.abs(criterion)
    return criterion
    #
    # You code here
    #


def nonmaxsuppression(criterion, threshold):
    """ Apply non-maximum suppression to criterion values
        and return Hessian interest points

        Args:
            criterion: (h, w) np.array of criterion function values
            threshold: criterion threshold
        Returns:
            rows: (n,) np.array with y-positions of interest points
            cols: (n,) np.array with x-positions of interest points
    """
    
    max_f = maximum_filter(criterion, size = 5, mode = 'mirror')
   
    local_max = np.where( max_f == criterion, criterion,0)
    """
    max_filter will get the maximum number of a certain area and replace all the number in this area with this maximum value.
    For example : for 3*3 kernal
    
    original:
        
    3 3 0 0 3   max_filter  3 3 3 3 3 
    0 0 2 1 3       ->      3 3 3 3 3
    0 1 1 1 2               3 3 2 3 3
    
    Comparing with the original array, we can know which values are local maximum and replace other values with zero.
    
    bool array of which values are local maximum :
    
    T T F F T
    F F F F T
    F F F F F

    local maximum array : 
    
    3 3 0 0 3
    0 0 0 0 3
    0 0 0 0 0
    """
    # x
            
    
    #r, c = np.nonzero(max_f > threshold)
    """
    fig =plt.figure()
    fig.add_subplot(1,2,1)
    plt.imshow(criterion)
    fig.add_subplot(1,2,2)
    plt.imshow(local_max )
    plt.show()
    """
    r, c = np.nonzero(local_max > threshold)
    return r,c
    #
    # You code here
    #
