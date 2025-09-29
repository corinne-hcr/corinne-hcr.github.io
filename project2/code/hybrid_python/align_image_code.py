import math
import numpy as np
import matplotlib.pyplot as plt
import skimage.transform as sktr
from PIL import Image




def get_points(im1, im2):
    print('Please select 2 points in each image for alignment.')
    plt.imshow(im1)
    p1, p2 = plt.ginput(2)
    plt.close()
    plt.imshow(im2)
    p3, p4 = plt.ginput(2)
    plt.close()
    return (p1, p2, p3, p4)

def recenter(im, r, c):
    R, C, _ = im.shape
    rpad = (int) (np.abs(2*r+1 - R))
    cpad = (int) (np.abs(2*c+1 - C))
    return np.pad(
        im, [(0 if r > (R-1)/2 else rpad, 0 if r < (R-1)/2 else rpad),
             (0 if c > (C-1)/2 else cpad, 0 if c < (C-1)/2 else cpad),
             (0, 0)], 'constant')

def find_centers(p1, p2):
    cx = np.round(np.mean([p1[0], p2[0]]))
    cy = np.round(np.mean([p1[1], p2[1]]))
    return cx, cy

def align_image_centers(im1, im2, pts):
    p1, p2, p3, p4 = pts
    h1, w1, b1 = im1.shape
    h2, w2, b2 = im2.shape
    
    cx1, cy1 = find_centers(p1, p2)
    cx2, cy2 = find_centers(p3, p4)

    im1 = recenter(im1, cy1, cx1)
    im2 = recenter(im2, cy2, cx2)
    return im1, im2

def rescale_images(im1, im2, pts):
    p1, p2, p3, p4 = pts
    len1 = np.sqrt((p2[1] - p1[1])**2 + (p2[0] - p1[0])**2)
    len2 = np.sqrt((p4[1] - p3[1])**2 + (p4[0] - p3[0])**2)
    dscale = len2/len1
    if dscale < 1:
        # im1 = sktr.rescale(im1, dscale)
        im1 = sktr.rescale(im1, dscale, multichannel=True)
    else:
        # im2 = sktr.rescale(im2, 1./dscale)
        im2 = sktr.rescale(im2, 1./dscale, multichannel=True)

    return im1, im2

def rotate_im1(im1, im2, pts):
    p1, p2, p3, p4 = pts
    theta1 = math.atan2(-(p2[1] - p1[1]), (p2[0] - p1[0]))
    theta2 = math.atan2(-(p4[1] - p3[1]), (p4[0] - p3[0]))
    dtheta = theta2 - theta1
    # im1 = sktr.rotate(im1, dtheta*180/np.pi)
    im1 = sktr.rotate(im1, dtheta*180/np.pi, preserve_range=True, mode='edge')

    return im1, dtheta

def match_img_size(im1, im2):
    # Make images the same size
    h1, w1, c1 = im1.shape
    h2, w2, c2 = im2.shape

    target_h = min(h1, h2)
    target_w = min(w1, w2)


    # if h1 < h2:
    #     im2 = im2[int(np.floor((h2-h1)/2.)) : -int(np.ceil((h2-h1)/2.)), :, :]
    # elif h1 > h2:
    #     im1 = im1[int(np.floor((h1-h2)/2.)) : -int(np.ceil((h1-h2)/2.)), :, :]
    # if w1 < w2:
    #     im2 = im2[:, int(np.floor((w2-w1)/2.)) : -int(np.ceil((w2-w1)/2.)), :]
    # elif w1 > w2:
    #     im1 = im1[:, int(np.floor((w1-w2)/2.)) : -int(np.ceil((w1-w2)/2.)), :]
    # assert im1.shape == im2.shape

    if h1 > target_h:
        start = (h1 - target_h) // 2
        im1 = im1[start:start + target_h, :, :]
    if w1 > target_w:
        start = (w1 - target_w) // 2
        im1 = im1[:, start:start + target_w, :]
        
    
    if h2 > target_h:
        start = (h2 - target_h) // 2
        im2 = im2[start:start + target_h, :, :]
    if w2 > target_w:
        start = (w2 - target_w) // 2
        im2 = im2[:, start:start + target_w, :]
    im1 = im1[:target_h, :target_w, :]
    im2 = im2[:target_h, :target_w, :]

    assert im1.shape == im2.shape, f"Still mismatch: {im1.shape} vs {im2.shape}"




    return im1, im2

def align_images(im1, im2):
    # if len(im1.shape) == 2:
    #     im1 = np.stack([im1]*3, axis=2)
    # if len(im2.shape) == 2:
    #     im2 = np.stack([im2]*3, axis=2)
    
   
    # if im1.shape[2] == 4:
    #     im1 = im1[:, :, :3]
    # if im2.shape[2] == 4:
    #     im2 = im2[:, :, :3]
    
   
    #     im1 = np.stack([im1[:,:,0]]*3, axis=2)
    # if im2.shape[2] == 2:
    #     im2 = np.stack([im2[:,:,0]]*3, axis=2)


    pts = get_points(im1, im2)
    im1, im2 = align_image_centers(im1, im2, pts)
    im1, im2 = rescale_images(im1, im2, pts)
    im1, angle = rotate_im1(im1, im2, pts)
    im1, im2 = match_img_size(im1, im2)
    return im1, im2


if __name__ == "__main__":
    # 1. load the image
    # 2. align the two images by calling align_images
    # Now you are ready to write your own code for creating hybrid images!
    # derek = plt.imread('DerekPicture.jpg') / 255.0
    # nutmeg = plt.imread('nutmeg.jpg') / 255.0
    

    cat = np.array(Image.open('cat.jpg').convert('RGB').resize((512, 512))) / 255.0
    dog = np.array(Image.open('dog.jpg').convert('RGB').resize((512, 512))) / 255.0
    tiger = np.array(Image.open('tiger.jpg').convert('RGB').resize((512, 512))) / 255.0

    # nutmeg_aligned, derek_aligned = align_images(nutmeg, derek)
    cat_aligned_t, tiger_aligned = align_images(cat, tiger)
    cat_aligned_d, dog_aligned = align_images(cat, dog)

    # plt.imsave('derek_aligned.jpg', derek_aligned)
    # plt.imsave('nutmeg_aligned.jpg', nutmeg_aligned)
    plt.imsave('cat_aligned_t.jpg', cat_aligned_t)
    plt.imsave('tiger_aligned.jpg', tiger_aligned)
    plt.imsave('cat_aligned_d.jpg', cat_aligned_d)
    plt.imsave('dog_aligned.jpg', dog_aligned)
    



