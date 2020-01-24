import os, cv2
import numpy as np
import scipy
import scipy.misc
import openslide
import pandas as pd
from PIL import ImageFilter, Image
import matplotlib.pyplot as plt
from skimage.transform import PiecewiseAffineTransform, warp
from skimage.color import rgb2hsv,hsv2rgb,rgb2lab,lab2rgb
from skimage import exposure

def save_wsi_thumbnail_mask(filename):
    try: filename = filename.numpy()
    except: filename = filename
    wsi = openslide.OpenSlide(filename)

    def find_tissue_mask():
        thumbnail = wsi.get_thumbnail((thumbnail_size,thumbnail_size))
        thumbnail_blurred = np.array(thumbnail.filter(ImageFilter.GaussianBlur(radius=10)))
        ret2,mask = cv2.threshold(thumbnail_blurred[:,:,0],0,255,cv2.THRESH_BINARY+cv2.THRESH_OTSU)
        kernel = np.ones((5,5),np.uint8)
        mask = cv2.erode(mask,kernel,iterations = 1)
        mask[mask==0] = 1
        mask[mask==255] = 0
        return mask

    l_dims = wsi.level_dimensions
    thumbnail_size = min(2000., max(l_dims[0]))

    mask_path = '{}_MASK.png'.format(filename.split('.')[0])

    if not os.path.isfile(mask_path):
        print(filename)
        mask = find_tissue_mask()*255
        mask_PIL = Image.fromarray(mask)
        mask_PIL.save(mask_path)


def get_random_wsi_batch(filename, patch_size, augment=False, downsample=4):
    '''
    takes a wsi and returns a random patch of patch_size
    downsample must be a multiple of 2

    '''
    try: augment = augment.numpy()
    except: augment = augment
    try: patch_size = patch_size.numpy()
    except: patch_size = patch_size
    try: filename = filename.numpy()
    except: filename = filename
    try: downsample = downsample.numpy()
    except: downsample = downsample

    if augment:
        # pad for affine
        patch_size = patch_size+4

    wsi = openslide.OpenSlide(filename)

    l_dims = wsi.level_dimensions
    level = wsi.get_best_level_for_downsample(downsample + 0.1)

    mask_path = '{}_MASK.png'.format(filename.decode().split('.')[0])
    mask = np.array(Image.open(mask_path))

    def get_random_patch(wsi, l_dims, level, mask, patch_size, filename, downsample, augment):
        iter = 0
        while True:

            if iter > 200: # use next level if stuck in loop
                print('{} broken for level {}'.format(filename,level))
                level -= 1
                print('\ttrying level {}'.format(level))
                if level == -1: # if no resolution works return white region
                    print('{} broken | using white patch...'.format(filename))
                    return np.ones((patch_size, patch_size,3))*255
                iter = 0

            if iter == 0: # calculate dimensions and scales
                try:
                    level_dims = l_dims[level]
                    level_downsample = wsi.level_downsamples[level]
                    thumbnail_size = float(max(mask.shape))
                    mask_scale = thumbnail_size/max(level_dims)
                    scale_factor = int(round(downsample / level_downsample))
                except:
                    print('{} broken | using white patch...'.format(filename))
                    return np.ones((patch_size, patch_size,3))*255

            iter += 1
            x_start = int(np.random.uniform(low=0, high=level_dims[0]-patch_size*scale_factor))
            y_start = int(np.random.uniform(low=0, high=level_dims[1]-patch_size*scale_factor))

            mask_x_start = int(x_start*mask_scale)
            mask_x_stop = int((x_start + patch_size*scale_factor)*mask_scale)
            mask_y_start = int(y_start*mask_scale)
            mask_y_stop = int((y_start + patch_size*scale_factor)*mask_scale)

            if np.sum(mask[mask_y_start:mask_y_stop, mask_x_start:mask_x_stop]) > (mask_y_stop-mask_y_start) * (mask_x_stop-mask_x_start) * 255 / 3:

                try:
                    region = wsi.read_region((int(x_start*level_downsample),int(y_start*level_downsample)), level, (patch_size*scale_factor,patch_size*scale_factor))

                    if scale_factor > 1:
                        region = region.resize((patch_size, patch_size), resample=1)
                    region = np.array(region)[:,:,:3]

                    def colorshift(img, hbound=0.025, lbound=0.015): #Shift Hue of HSV space and Lightness of LAB space
                        hShift=np.random.normal(0,hbound)
                        lShift=np.random.normal(1,lbound)
                        img=rgb2hsv(img)
                        img[:,:,0]=(img[:,:,0]+hShift)
                        img=hsv2rgb(img)
                        img=rgb2lab(img)
                        img[:,:,0]=exposure.adjust_gamma(img[:,:,0],lShift)
                        img=lab2rgb(img)
                        return img

                    def PiecewiseAffine(img, points=8):
                        ### piecwise affine ###
                        rows, cols = img.shape[0], img.shape[1]
                        src_cols = np.linspace(0, cols, points)
                        src_rows = np.linspace(0, rows, points)
                        src_rows, src_cols = np.meshgrid(src_rows, src_cols)
                        src = np.dstack([src_cols.flat, src_rows.flat])[0]
                        # add offset
                        dst_rows = np.zeros(src[:, 1].shape) + src[:, 1]
                        for i in list(range(points))[1:-1]:
                            dst_rows[i::points] += np.random.normal(loc=0, scale=rows/(points*10), size=dst_rows[i::points].shape)
                        dst_cols = np.zeros(src[:, 0].shape) + src[:, 0]
                        dst_cols[points:-points] += np.random.normal(loc=0,scale=rows/(points*10), size=dst_cols[points:-points].shape)
                        dst = np.vstack([dst_cols, dst_rows]).T
                        # compute transform
                        tform = PiecewiseAffineTransform()
                        tform.estimate(src, dst)
                        # apply transform
                        img = warp(img, tform, output_shape=(rows, cols))
                        return img

                    if augment:
                        if np.random.random() > .5:
                            # augment image
                            region = (region/255.).astype(np.float64)
                            region = colorshift(region)
                            region, patch_mask = PiecewiseAffine(region)
                            region = (region*255.)
                        # unpad
                        region = region[2:-2,2:-2,:]

                    return region

                except: pass

    region = get_random_patch(wsi, l_dims, level, mask, patch_size, filename, downsample, augment)
    region_t = np.transpose(region, (2,0,1)) # [CWH]
    return region_t

def get_slide_label(filename, data_label_xlsx):
    data_label_xlsx = str(data_label_xlsx.numpy())
    filename = str(filename.numpy())
    # get slide label
    df = pd.read_excel(data_label_xlsx)
    name = filename.split('/')[-1]
    index = df.index[df['wsi']==name].tolist()
    if index == []:
        label = np.array([-1])
    else:
        label = np.array(df['class'][index])
    return label
