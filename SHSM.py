import numpy as np
import cv2
import math
from scipy import ndimage


def r2p(x):
    return np.abs(x), np.angle(x)


def ssim(img1, img2):

    """ Modified from http://mubeta06.github.io/python/sp/_modules/sp/ssim.html
    Return the Structural Similarity Map corresponding to input images img1
    and img2 (images are assumed to be uint8)

    This function attempts to mimic precisely the functionality of ssim.m a
    MATLAB provided by the author's of SSIM
    https://ece.uwaterloo.ca/~z70wang/research/ssim/ssim_index.m
    """

    img1 = img1.astype(np.float64)
    img2 = img2.astype(np.float64)

    # The following hyper-parameters are from the original SSIM paper.
    sigma = 1.5
    k1 = 0.01
    k2 = 0.03

    l = 255  # bit depth of image
    c1 = (k1*l)**2
    c2 = (k2*l)**2

    mu1 = ndimage.filters.gaussian_filter(img1, sigma=sigma,  mode='nearest')
    mu2 = ndimage.filters.gaussian_filter(img2, sigma=sigma, mode='nearest')

    mu1_sq = mu1*mu1
    mu2_sq = mu2*mu2
    mu1_mu2 = mu1*mu2

    sigma1_sq = ndimage.filters.gaussian_filter((img1*img1), sigma=sigma, mode='nearest') - mu1_sq
    sigma2_sq = ndimage.filters.gaussian_filter((img2*img2), sigma=sigma, mode='nearest') - mu2_sq
    sigma12 = ndimage.filters.gaussian_filter(img1*img2, sigma=sigma, mode='nearest') - mu1_mu2

    return ((2*mu1_mu2 + c1)*(2*sigma12 + c2))/((mu1_sq + mu2_sq + c1)*(sigma1_sq + sigma2_sq + c2))


def shsm(img1, img2):

    negligable_grad_x_0 = 5
    x_0 = 5
    sigma = 1.5

    k1 = 0.03

    img1 = img1.astype(np.float64)

    cg1 = 1/(1+np.exp(-1*(img1[:, :, 2]-negligable_grad_x_0)))
    img1 = img1[:, :, 1]  # set img1 to Hue channel only.

    img2 = img2.astype(np.float64)

    cg2 = 1/(1+np.exp(-1*(img2[:, :, 2]-negligable_grad_x_0)))
    img2 = img2[:, :, 1]  # set img2 to Hue channel only.


    kernel = np.array([[1,0,0],[0,0,0],[0,0,-1]])
    img1_der = ndimage.filters.convolve(img1, kernel, mode='nearest')  # get gradients of hue channel of image 1
    img1_der = img1_der*cg1  # Gradients of pixels with negligable chrominance are suppressed.
    img1_der = np.abs(img1_der)
    img1_der = np.reshape(img1_der,(65536))
    img1_der[img1_der>128] = 256 - img1_der[img1_der>128]
    img1_der = np.reshape(img1_der,(256,256))
    img1_der = 1/(1+np.exp(-(img1_der-x_0)))
    deriv1_a = ndimage.filters.gaussian_filter(img1_der, sigma=sigma, mode='nearest')

    img2_der = ndimage.filters.convolve(img2, kernel, mode='nearest') #get gradients of hue channel of image 2
    img2_der = img2_der*cg2 #Gradients of pixels with negligable chrominance are supressed.
    img2_der = np.abs(img2_der)
    img2_der = np.reshape(img2_der,(65536))
    img2_der[img2_der>128] = 256 - img2_der[img2_der>128]
    img2_der = np.reshape(img2_der,(256,256))
    img2_der= 1/(1+np.exp(-(img2_der-x_0)))
    deriv2_a = ndimage.filters.gaussian_filter(img2_der, sigma=sigma, mode='nearest')

    kernel = np.array([[0,0,1],[0,0,0],[-1,0,0]]) # we need to check gradients in two orthogonal directions.
    img1_der = ndimage.filters.convolve(img1, kernel, mode='nearest') #get gradients of hue channel of image 1
    img1_der = img1_der*cg1 #Gradients of pixels with negligable chrominance are supressed.
    img1_der = np.sqrt(img1_der**2)
    img1_der = np.reshape(img1_der,(65536))
    img1_der[img1_der>128] = 256 - img1_der[img1_der>128]
    img1_der = np.reshape(img1_der,(256,256))
    img1_der = 1/(1+np.exp(-(img1_der-x_0)))
    deriv1_b = ndimage.filters.gaussian_filter(img1_der, sigma=sigma, mode='nearest')

    img2_der = ndimage.filters.convolve(img2, kernel, mode='nearest')  # get gradients of hue channel of image 2
    img2_der = img2_der*cg2  # Gradients of pixels with negligable chrominance are supressed.
    img2_der = np.abs(img2_der)
    img2_der = np.reshape(img2_der,(65536))
    img2_der[img2_der>128] = 256 - img2_der[img2_der>128]
    img2_der = np.reshape(img2_der,(256,256))
    img2_der= 1/(1+np.exp(-(img2_der-x_0)))
    deriv2_b = ndimage.filters.gaussian_filter(img2_der, sigma=sigma, mode='nearest')

    return (2*deriv1_a*deriv2_a+k1)/(deriv1_a**2+deriv2_a**2+k1)*(2*deriv1_b*deriv2_b+k1)/(deriv1_b**2+deriv2_b**2+k1)


def create_output_map_images(gt_bgr, cl_bgr, gt_lhc, cl_lhc, res_l, res_h, res_c):
    res_lim = np.uint8(res_l*255)
    res_lim = cv2.merge((res_lim, res_lim, res_lim))
    res_him = np.uint8(res_h*255)
    res_him = cv2.merge((res_him,res_him,res_him))
    res_cim = np.uint8(res_c*255)
    res_cim = cv2.merge((res_cim, res_cim, res_cim))

    res_lhc = res_l*res_h*res_c
    res_lhcim = np.uint8(res_lhc*255)
    res_lhcim = cv2.merge((res_lhcim, res_lhcim, res_lhcim))

    white_v_divider = 255*np.ones((256,5,3))
    full_fig = np.hstack([gt_bgr, white_v_divider])
    full_fig = np.hstack([full_fig, cl_bgr])
    full_fig = np.hstack([full_fig, white_v_divider])
    full_fig = np.hstack([full_fig, res_lim])
    full_fig = np.hstack([full_fig, white_v_divider])
    full_fig = np.hstack([full_fig, res_him])
    full_fig = np.hstack([full_fig, white_v_divider])
    full_fig = np.hstack([full_fig, res_cim])
    full_fig = np.hstack([full_fig, white_v_divider])
    full_fig = np.hstack([full_fig, res_lhcim])
    cv2.imwrite('./Lab.png', full_fig)

    white_h_divider=255*np.ones((5,778,3))
    full_fig = np.hstack([gt_bgr, white_v_divider])
    full_fig = np.hstack([full_fig, cl_bgr])
    full_fig = np.hstack([full_fig, white_v_divider])
    full_fig = np.hstack([full_fig, res_lhcim])

    # We don't actually use this next section (next four lines) in the output but you can add the l below
    l = np.hstack([cv2.merge((gt_lhc[:, :, 0],gt_lhc[:, :, 0],gt_lhc[:, :, 0])), white_v_divider])
    l = np.hstack([l,cv2.merge( (cl_lhc[:, :, 0], cl_lhc[:, :, 0], cl_lhc[:, :, 0]))])
    l = np.hstack([l, white_v_divider])
    l = np.hstack([l, res_lim])

    h = np.hstack([cv2.merge((gt_lhc[:,:,1], gt_lhc[:,:,1], gt_lhc[:,:,1])), white_v_divider])
    h = np.hstack([h,cv2.merge( (cl_lhc[:, :, 1], cl_lhc[:, :, 1], cl_lhc[:, :, 1]))])
    h = np.hstack([h, white_v_divider])
    h = np.hstack([h, res_him])

    c = np.hstack([cv2.merge((gt_lhc[:,:,2]*2, gt_lhc[:, :, 2]*2, gt_lhc[:, :, 2]*2)), white_v_divider])
    c = np.hstack([c, cv2.merge( (cl_lhc[:, :, 2]*2, cl_lhc[:, :, 2]*2, cl_lhc[:, :, 2]*2))])
    c = np.hstack([c, white_v_divider])
    c = np.hstack([c, res_cim])

    full_fig = np.vstack([full_fig, white_h_divider])
    full_fig = np.vstack([full_fig, h])

    full_fig = np.vstack([full_fig, white_h_divider])
    full_fig = np.vstack([full_fig, c])

    cv2.imwrite('./SSIM_SSHM_map.png', full_fig)


def compare(gt_bgr, cl_bgr, create_images=False):
    gt_lab = cv2.cvtColor(gt_bgr, cv2.COLOR_BGR2LAB) #convert the two images to CIEL*a*b*
    cl_lab = cv2.cvtColor(cl_bgr, cv2.COLOR_BGR2LAB)

    # change Ground truth image to Lhc space
    gt_comp = np.array(gt_lab[:, :, 1:3], np.float64)  # change ab to floating point
    gt_comp = (gt_comp[:, :, 0]-128) + (1j * (gt_comp[:, :, 1]-128))  # convert to complex number (cartesian) format
    gt_c, gt_h = r2p(gt_comp)  # convert to polar form gt_c is magnitude, gt_h is hue
    gt_h = np.array(gt_h/math.pi*128+128, np.dtype(np.uint8))  # change the range of gt_h from (-pi, +pi) to (0,255)
    gt_c = np.array(gt_c, np.uint8)     # just change to uint8
    gt_lhc = cv2.merge((gt_lab[:, :, 0], gt_h, gt_c))         # merge the three channels

    # Repeat process for the colourisation
    cl_comp = np.array(cl_lab[:,:,1:3], np.float64)
    cl_comp = (cl_comp[:,:,0]-128) + (1j * (cl_comp[:,:,1]-128))
    cl_c, cl_h = r2p(cl_comp)
    cl_h = np.array(cl_h/math.pi*128+128, np.dtype(np.uint8))
    cl_c = np.array(cl_c, np.uint8)
    cl_lhc = cv2.merge((cl_lab[:, :, 0], cl_h, cl_c))

    resL = ssim(gt_lhc[:,:,0], cl_lhc[:,:,0])
    resc = ssim(gt_lhc[:,:,2], cl_lhc[:,:,2])

    resh = shsm(gt_lhc, cl_lhc)

    reshc = resh*resc

    if create_images:
        create_output_map_images(gt_bgr, cl_bgr, gt_lhc, cl_lhc, resL, resh, resc)

    return np.array([np.mean(resL), np.mean(resh), np.mean(resc), np.mean(reshc)])
