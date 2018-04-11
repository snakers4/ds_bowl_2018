import os
import cv2
import time
import glob

import collections
import numpy as np
import pandas as pd

from scipy import ndimage
import torch.utils.data as data
from skimage.draw import circle
from skimage.morphology import thin
from skimage.io import imread,imread_collection
from skimage.segmentation import find_boundaries
from sklearn.model_selection import train_test_split
from sklearn.model_selection import StratifiedKFold
import imgaug as ia
from imgaug import augmenters as iaa
from skimage.morphology import binary_dilation
from skimage.morphology import disk

cv2.setNumThreads(0)

# for testing
resolution_dict2 = {
    # resolutions divisible by 16
    '1024x1024':[[1024,1024],[1024,1024]],
    '1040x1388':[[1040,1388],[1056,1376]],    
}

resolution_dict = {
    # resolutions divisible by 16
    '256x256':[[256,256],[256,256]],
    '256x320':[[256,320],[256,320]],
    '1024x1024':[[1024,1024],[1024,1024]],
    '512x640':[[512,640],[512,640]],   
    '260x347':[[260,347],[288,352]], 
    '520x696':[[520,696],[544,704]],    
    '360x360':[[360,360],[384,384]],
    '603x1272':[[603,1272],[608,1280]],
    '1040x1388':[[1040,1388],[1056,1376]],  

    # additional test resolutions    
    '512x680':[[512,680],[544,704]], 
    '519x253':[[519,253],[544,256]],     
    '524x348':[[524,348],[544,352]],  
    '520x348':[[520,348],[544,352]],
    '519x162':[[519,162],[544,160]],
    '519x161':[[519,161],[544,160]],
    '390x239':[[390,239],[416,256]],          
}

resolution_list = [
    # resolutions divisible by 16
    ['256x256',[256,256],[256,256]],
    ['360x360',[360,360],[384,384]],  
    ['520x696',[520,696],[544,704]],   
    ['256x320',[256,320],[256,320]],    
    ['260x347',[260,347],[288,352]], 
    ['512x640',[512,640],[512,640]],   
    ['603x1272',[603,1272],[608,1280]],    
    ['1040x1388',[1040,1388],[1056,1376]],  
    ['1024x1024',[1024,1024],[1024,1024]], 
]

resolution_list_wo1024 = [
    # resolutions divisible by 16
    ['256x256',[256,256],[256,256]],
    ['360x360',[360,360],[384,384]],  
    ['520x696',[520,696],[544,704]],   
    ['256x320',[256,320],[256,320]],    
    ['260x347',[260,347],[288,352]], 
    ['512x640',[512,640],[512,640]],   
    ['603x1272',[603,1272],[608,1280]],    
    ['1040x1388',[1040,1388],[1056,1376]],  
]

resolution_list_train_small = [
    # resolutions divisible by 16
    ['256x256',[256,256],[256,256]],
    ['360x360',[360,360],[384,384]],  
    ['520x696',[520,696],[544,704]],   
    ['256x320',[256,320],[256,320]],    
    ['260x347',[260,347],[288,352]], 
    ['512x640',[512,640],[512,640]],   
]


sometimes = lambda aug: iaa.Sometimes(0.5, aug)

# Define our sequence of augmentation steps that will be applied to every image.
seq = iaa.Sequential(
    [
        # Execute 0 to 5 of the following (less important) augmenters per
        # image. Don't execute all of them, as that would often be way too
        # strong.
        #
        iaa.SomeOf((0, 5),
            [
                # Blur each image with varying strength using
                # gaussian blur (sigma between 0 and 3.0),
                # average/uniform blur (kernel size between 2x2 and 7x7)
                # median blur (kernel size between 3x3 and 11x11).
                iaa.OneOf([
                    iaa.GaussianBlur((0, 3.0)),
                    iaa.AverageBlur(k=(2, 7)),
                ]),
                # Sharpen each image, overlay the result with the original
                # image using an alpha between 0 (no sharpening) and 1
                # (full sharpening effect).
                iaa.Sharpen(alpha=(0, 1.0), lightness=(0.75, 1.5)),
                # Same as sharpen, but for an embossing effect.
                iaa.Emboss(alpha=(0, 1.0), strength=(0, 2.0)),
                # Search in some images either for all edges or for
                # directed edges. These edges are then marked in a black
                # and white image and overlayed with the original image
                # using an alpha of 0 to 0.7.
                sometimes(iaa.OneOf([
                    iaa.EdgeDetect(alpha=(0, 0.7)),
                    iaa.DirectedEdgeDetect(
                        alpha=(0, 0.7), direction=(0.0, 1.0)
                    ),
                ])),
                # Add gaussian noise to some images.
                # In 50% of these cases, the noise is randomly sampled per
                # channel and pixel.
                # In the other 50% of all cases it is sampled once per
                # pixel (i.e. brightness change).
                iaa.AdditiveGaussianNoise(
                    loc=0, scale=(0.0, 0.05*255), per_channel=0.5
                ),
                # Either drop randomly 1 to 10% of all pixels (i.e. set
                # them to black) or drop them on an image with 2-5% percent
                # of the original size, leading to large dropped
                # rectangles.
                iaa.OneOf([
                    iaa.Dropout((0.01, 0.1), per_channel=0.5),
                    iaa.CoarseDropout(
                        (0.03, 0.15), size_percent=(0.02, 0.05),
                        per_channel=0.2
                    ),
                ]),
                # Invert each image's chanell with 5% probability.
                # This sets each pixel value v to 255-v.
                iaa.Invert(0.05, per_channel=True), # invert color channels
                # Add a value of -10 to 10 to each pixel.
                iaa.Add((-10, 10), per_channel=0.5),
                # Change brightness of images (50-150% of original value).
                iaa.Multiply((0.5, 1.5), per_channel=0.5),
                # Improve or worsen the contrast of images.
                iaa.ContrastNormalization((0.5, 2.0), per_channel=0.5),
                # Convert each image to grayscale and then overlay the
                # result with the original with random alpha. I.e. remove
                # colors with varying strengths.
                iaa.Grayscale(alpha=(0.0, 1.0)),
            ],
            # do all of the above augmentations in random order
            random_order=True
        )
    ],
    # do all of the above augmentations in random order
    random_order=True
)

def thin_region_fast(mask,
                    iterations):
    
    min_x, max_x = np.argwhere(mask > 0)[:,0].min(),np.argwhere(mask > 0)[:,0].max()
    min_y, max_y = np.argwhere(mask > 0)[:,1].min(),np.argwhere(mask > 0)[:,1].max()
   
    empty = np.zeros_like(mask)
    try:
        empty[min_x:max_x,min_y:max_y] = thin(mask[min_x:max_x,min_y:max_y],max_iter=iterations)
        return empty
    except:
        return empty
    
def distance_transform_fast(mask,
                           return_indices=False):
    
    min_x, max_x = np.argwhere(mask > 0)[:,0].min(),np.argwhere(mask > 0)[:,0].max()
    min_y, max_y = np.argwhere(mask > 0)[:,1].min(),np.argwhere(mask > 0)[:,1].max()
    
    if return_indices == False:
        empty = np.zeros_like(mask)
        try:
            empty[min_x:max_x,min_y:max_y] = ndimage.distance_transform_edt(mask[min_x:max_x,min_y:max_y])
            return empty
        except:
            return empty
    else:
        min_x = max(min_x-5,0)
        min_y = max(min_y-5,0)
        max_x = max_x+5
        max_y = max_y+5
        
        empty = np.zeros_like(mask)
        indices = np.zeros_like(np.vstack([[mask]*2])) 
        
        try:
            empty[min_x:max_x,min_y:max_y],indices[:,min_x:max_x,min_y:max_y] = ndimage.distance_transform_edt(mask[min_x:max_x,min_y:max_y],return_indices=True)
            indices[0] = indices[0] + min_x
            indices[1] = indices[1] + min_y
            return empty,indices
        except:
            return empty,indices        
    
def mask2vectors(mask):
    distances, indices = distance_transform_fast(mask,return_indices=True)
    # avoid division by zero for blank areas  when normalizing
    grid_indices = np.indices((mask.shape[0],mask.shape[1]))
    distances[distances==0]=1
    return (indices*(mask>255//2) - grid_indices*(mask>255//2)) / np.asarray([distances,distances])

class BDataset(data.Dataset):
    def __init__(self,
                 df = None,
                 transforms = None,
                 fold_num = 0, # which fold to use
                 mode = 'train', # 'train', 'val' or 'test'
                 dset_resl = '256x256',
                 erosion_type = 'boundary', # erode or boundary
                 nuclei_size = 4):
        
        self.df = df
        self.df = self.df.reset_index()
        self.transforms = transforms
        self.fold_num = fold_num
        self.mode = mode
        self.nuclei_size = nuclei_size
        self.erosion_type = erosion_type
        self.resolution_dict = resolution_dict
        if self.mode in ['train','val']:
            # set indices
            skf = StratifiedKFold(n_splits=4,
                                  shuffle = True,
                                  random_state = 42)

            # stratify fold by cluster number
            f1, f2, f3, f4 = skf.split(self.df.index.values,self.df.cluster.values)
            self.folds = [f1, f2, f3, f4]
            # all train images
            self.train_idx = self.folds[self.fold_num][0]
            self.val_idx = self.folds[self.fold_num][1]
            
            # leave only images with particular resolution
            self.train_idx = list(set(self.df[self.df.w_h==dset_resl].index.values).intersection(set(self.train_idx)))
            self.val_idx = list(set(self.df[self.df.w_h==dset_resl].index.values).intersection(set(self.val_idx)))
        else:
            self.train_idx = list(set(self.df[self.df.w_h==dset_resl].index.values))
            
    def __len__(self):
        if self.mode == 'train':
            return len(self.train_idx)
        elif self.mode == 'val':
            return len(self.val_idx)          
        elif self.mode == 'test':  
            return len(self.train_idx) 

    def __getitem__(self, idx):
        if self.mode == 'train':
            img_glob = glob.glob('../data/stage1_train/{}/images/*.png'.format(self.df.loc[self.train_idx[idx],'sample']))
            mask_glob = glob.glob('../data/stage1_train/{}/masks/*.png'.format(self.df.loc[self.train_idx[idx],'sample']))
            
            or_resl = self.resolution_dict[self.df.loc[self.train_idx[idx],'w_h']][0]
            target_resl = self.resolution_dict[self.df.loc[self.train_idx[idx],'w_h']][1]            
            
            masks = imread_collection(mask_glob).concatenate()
            mask = np.sum(np.stack(masks, 0), 0)/255.0
            
            gt_labels = np.zeros((masks.shape[1], masks.shape[2]), np.uint16)
            for index in range(0, masks.shape[0]):
                gt_labels[masks[index] > 0] = index + 1
                
            boundaries = find_boundaries(gt_labels, connectivity=1, mode='outer', background=0) 
           
            if self.erosion_type == 'erode':
                # erode a few layers
                masks_thin = np.asarray([(thin(image=mask,max_iter=3)) for mask in masks])
                # replace mask with the eroded version
                mask = np.sum(np.stack(masks_thin, 0), 0)/255.0            
            else:
                # just remove the boundaries
                mask[boundaries] = 0
            
            # boundaries-only mask
            boundaries = boundaries * 1
            img = imread(img_glob[0])[:,:,0:3]
            
            nuclei_centers_coord = [(ndimage.measurements.center_of_mass(_) ) for _ in masks]
            nuclei_centers_coord = [(int(_[0]),int(_[1])) for _ in nuclei_centers_coord]
            nuclei_centers = np.zeros((masks.shape[1], masks.shape[2]), dtype=int)
            
            for coord in nuclei_centers_coord: 
                rr, cc = circle(coord[0], coord[1], self.nuclei_size, mask.shape)
                nuclei_centers[rr, cc] = 1
        elif self.mode == 'val':
            img_glob = glob.glob('../data/stage1_train/{}/images/*.png'.format(self.df.loc[self.val_idx[idx],'sample']))
            mask_glob = glob.glob('../data/stage1_train/{}/masks/*.png'.format(self.df.loc[self.val_idx[idx],'sample']))
            
            or_resl = self.resolution_dict[self.df.loc[self.val_idx[idx],'w_h']][0]
            target_resl = self.resolution_dict[self.df.loc[self.val_idx[idx],'w_h']][1]            
            
            masks = imread_collection(mask_glob).concatenate()
            mask = np.sum(np.stack(masks, 0), 0)/255.0
            
            gt_labels = np.zeros((masks.shape[1], masks.shape[2]), np.uint16)
            for index in range(0, masks.shape[0]):
                gt_labels[masks[index] > 0] = index + 1
                
            boundaries = find_boundaries(gt_labels, connectivity=1, mode='outer', background=0) 

            if self.erosion_type == 'erode':
                # erode a few layers
                masks_thin = np.asarray([(thin(image=mask,max_iter=3)) for mask in masks])
                # replace mask with the eroded version
                mask = np.sum(np.stack(masks_thin, 0), 0)/255.0            
            else:
                # just remove the boundaries
                mask[boundaries] = 0            
            
            # boundaries-only mask
            boundaries = boundaries * 1
            img = imread(img_glob[0])[:,:,0:3]
            
            nuclei_centers_coord = [(fn.measurements.center_of_mass(_) ) for _ in masks]
            nuclei_centers_coord = [(int(_[0]),int(_[1])) for _ in nuclei_centers_coord]
            nuclei_centers = np.zeros((masks.shape[1], masks.shape[2]), dtype=int)
            
            for coord in nuclei_centers_coord: 
                rr, cc = circle(coord[0], coord[1], self.nuclei_size, mask.shape)
                nuclei_centers[rr, cc] = 1
        elif self.mode == 'test':
            img_glob = glob.glob('../data/stage1_test/{}/images/*.png'.format(self.df.loc[self.train_idx[idx],'sample']))
            img = imread(img_glob[0])[:,:,0:3]
            
            mask = None
            boundaries = None
            nuclei_centers = None
            or_resl = self.resolution_dict[self.df.loc[self.train_idx[idx],'w_h']][0]
            target_resl = self.resolution_dict[self.df.loc[self.train_idx[idx],'w_h']][1]
            
        if self.transforms is not None:
            if mask is not None:            
                msk = np.stack((mask,boundaries,nuclei_centers),axis=2)
                img, msk = self.transforms(img, msk, target_resl)
            else:
                msk = 0
                img, _ = self.transforms(img, None, target_resl)
        
        # if the image gets flipped, flip it back
        if img.shape[0]!=target_resl[0]:
            img = img.transpose(1, 0, 2)
            if mask is not None:
                msk = msk.transpose(1, 0, 2)

        return img,msk,or_resl,target_resl

class BDatasetResize(data.Dataset):
    def __init__(self,
                 df = None,
                 transforms = None,
                 fold_num = 0, # which fold to use
                 mode = 'train', # 'train', 'val' or 'test'
                 dset_resl = '256x256',
                 erosion_type = 'boundary', # erode or boundary
                 nuclei_size = 4):
        
        bad_idx = [53]
        self.df = df
        self.df = self.df.reset_index()
        self.transforms = transforms
        self.fold_num = fold_num
        self.mode = mode
        self.nuclei_size = nuclei_size
        self.erosion_type = erosion_type
        self.resolution_dict = resolution_dict
        if self.mode in ['train','val']:
            # set indices
            skf = StratifiedKFold(n_splits=4,
                                  shuffle = True,
                                  random_state = 42)

            # stratify fold by cluster number
            f1, f2, f3, f4 = skf.split(self.df.index.values,self.df.cluster.values)
            self.folds = [f1, f2, f3, f4]
            # all train images
            self.train_idx = self.folds[self.fold_num][0]
            self.val_idx = self.folds[self.fold_num][1]
            
            # leave only images with particular resolution
            # also remove idx 53 - broken data there
            self.train_idx = list(set(self.df[(self.df.w_h==dset_resl)&(~self.df.index.isin(bad_idx))].index.values).intersection(set(self.train_idx)))
            self.val_idx = list(set(self.df[(self.df.w_h==dset_resl)&(~self.df.index.isin(bad_idx))].index.values).intersection(set(self.val_idx)))
        else:
            self.train_idx = list(set(self.df[self.df.w_h==dset_resl].index.values))
            
    def __len__(self):
        if self.mode == 'train':
            return len(self.train_idx)
        elif self.mode == 'val':
            return len(self.val_idx)          
        elif self.mode == 'test':  
            return len(self.train_idx) 

    def __getitem__(self, idx):
        if self.mode == 'train':
            img_glob = glob.glob('../data/stage1_train/{}/images/*.png'.format(self.df.loc[self.train_idx[idx],'sample']))
            mask_glob = glob.glob('../data/stage1_train/{}/masks/*.png'.format(self.df.loc[self.train_idx[idx],'sample']))

            masks = imread_collection(mask_glob).concatenate()
            mask = np.sum(np.stack(masks, 0), 0)/255.0
            
            gt_labels = np.zeros((masks.shape[1], masks.shape[2]), np.uint16)
            for index in range(0, masks.shape[0]):
                gt_labels[masks[index] > 0] = index + 1
                
            boundaries = find_boundaries(gt_labels, connectivity=1, mode='outer', background=0) 
           
            if self.erosion_type == 'erode':
                # erode a few layers
                masks_thin = np.asarray([(thin(image=mask,max_iter=3)) for mask in masks])
                # replace mask with the eroded version
                mask = np.sum(np.stack(masks_thin, 0), 0)/255.0            
            else:
                # just remove the boundaries
                mask[boundaries] = 0
            
            # boundaries-only mask
            boundaries = boundaries * 1
            img = imread(img_glob[0])[:,:,0:3]
            
            nuclei_centers_coord = [(ndimage.measurements.center_of_mass(_) ) for _ in masks]
            nuclei_centers_coord = [(int(_[0]),int(_[1])) for _ in nuclei_centers_coord]
            nuclei_centers = np.zeros((masks.shape[1], masks.shape[2]), dtype=int)
            
            for coord in nuclei_centers_coord: 
                rr, cc = circle(coord[0], coord[1], self.nuclei_size, mask.shape)
                nuclei_centers[rr, cc] = 1
        elif self.mode == 'val':
            img_glob = glob.glob('../data/stage1_train/{}/images/*.png'.format(self.df.loc[self.val_idx[idx],'sample']))
            mask_glob = glob.glob('../data/stage1_train/{}/masks/*.png'.format(self.df.loc[self.val_idx[idx],'sample']))
            
            masks = imread_collection(mask_glob).concatenate()
            mask = np.sum(np.stack(masks, 0), 0)/255.0
            
            gt_labels = np.zeros((masks.shape[1], masks.shape[2]), np.uint16)
            for index in range(0, masks.shape[0]):
                gt_labels[masks[index] > 0] = index + 1
                
            boundaries = find_boundaries(gt_labels, connectivity=1, mode='outer', background=0) 

            if self.erosion_type == 'erode':
                # erode a few layers
                masks_thin = np.asarray([(thin(image=mask,max_iter=3)) for mask in masks])
                # replace mask with the eroded version
                mask = np.sum(np.stack(masks_thin, 0), 0)/255.0            
            else:
                # just remove the boundaries
                mask[boundaries] = 0            
            
            # boundaries-only mask
            boundaries = boundaries * 1
            img = imread(img_glob[0])[:,:,0:3]
            
            nuclei_centers_coord = [(ndimage.measurements.center_of_mass(_) ) for _ in masks]
            nuclei_centers_coord = [(int(_[0]),int(_[1])) for _ in nuclei_centers_coord]
            nuclei_centers = np.zeros((masks.shape[1], masks.shape[2]), dtype=int)
            
            for coord in nuclei_centers_coord: 
                rr, cc = circle(coord[0], coord[1], self.nuclei_size, mask.shape)
                nuclei_centers[rr, cc] = 1
        elif self.mode == 'test':
            img_glob = glob.glob('../data/stage1_test/{}/images/*.png'.format(self.df.loc[self.train_idx[idx],'sample']))
            img = imread(img_glob[0])[:,:,0:3]
            
            img_id = self.df.loc[self.train_idx[idx],'img'].split('.')[0]
            
            mask = None
            boundaries = None
            nuclei_centers = None
            
        # estimage image divisibility by 32
        # if the image is not divisible in any of the dimensions, upscale this dimensions
        w,h = img.shape[0],img.shape[1]
        
        if w % 32 != 0:
            target_w = (w//32 + 1) * 32
        else:
            target_w = w
        if h % 32 != 0:
            target_h = (h//32 + 1) * 32
        else:
            target_h = h

        if w != target_w or h != target_h:
            img = cv2.resize(img, (target_h,target_w), interpolation=cv2.INTER_LINEAR)
            
            if mask is not None: 
                msk = np.stack((mask,boundaries,nuclei_centers),axis=2)
                msk = cv2.resize(msk, (target_h,target_w), interpolation=cv2.INTER_LINEAR)
        else:
            if mask is not None: 
                msk = np.stack((mask,boundaries,nuclei_centers),axis=2)
        
        if self.transforms is not None:
            if mask is not None:            
                # msk = np.stack((mask,boundaries,nuclei_centers),axis=2)
                img, msk = self.transforms(img, msk)
            else:
                msk = 0
                img, _ = self.transforms(img, None)
        
        # if the image gets flipped, flip it back
        if img.shape[0]!=target_w:
            img = img.transpose(1, 0, 2)
            if mask is not None:
                msk = msk.transpose(1, 0, 2)
        if self.mode in ['train','val']:
            return img,msk,(w,h),(target_w,target_h)
        else:
            return img,msk,(w,h),(target_w,target_h),img_id
        
class BDatasetResizeSeed(data.Dataset):
    def __init__(self,
                 df = None,
                 transforms = None,
                 fold_num = 0, # which fold to use
                 mode = 'train', # 'train', 'val' or 'test'
                 dset_resl = '256x256',
                 erosion_type = 'boundary', # erode or boundary
                 boundary_mode = 'outer', # outer inner or thick
                 nuclei_size = 4,
                 factor = 64,
                 is_crop = False):
        
        self.is_crop = is_crop
        bad_idx = [53]
        self.df = df
        self.factor = 64
        self.df = self.df.reset_index()
        self.transforms = transforms
        self.fold_num = fold_num
        self.mode = mode
        self.nuclei_size = nuclei_size
        self.erosion_type = erosion_type
        self.resolution_dict = resolution_dict
        self.boundary_mode = boundary_mode
        
        if self.mode in ['train','val']:
            # set indices
            skf = StratifiedKFold(n_splits=4,
                                  shuffle = True,
                                  random_state = 42)

            # stratify fold by cluster number
            f1, f2, f3, f4 = skf.split(self.df.index.values,self.df.cluster.values)
            self.folds = [f1, f2, f3, f4]
            # all train images
            self.train_idx = self.folds[self.fold_num][0]
            self.val_idx = self.folds[self.fold_num][1]
            
            # leave only images with particular resolution
            # also remove idx 53 - broken data there
            self.train_idx = list(set(self.df[(self.df.w_h==dset_resl)&(~self.df.index.isin(bad_idx))].index.values).intersection(set(self.train_idx)))
            self.val_idx = list(set(self.df[(self.df.w_h==dset_resl)&(~self.df.index.isin(bad_idx))].index.values).intersection(set(self.val_idx)))
        else:
            self.train_idx = list(set(self.df[self.df.w_h==dset_resl].index.values))
            
    def __len__(self):
        if self.mode == 'train':
            return len(self.train_idx)
        elif self.mode == 'val':
            return len(self.val_idx)          
        elif self.mode == 'test':  
            return len(self.train_idx) 

    def __getitem__(self, idx):
        if self.mode == 'train':
            img_glob = glob.glob('../data/stage1_train/{}/images/*.png'.format(self.df.loc[self.train_idx[idx],'sample']))
            mask_glob = glob.glob('../data/stage1_train/{}/masks/*.png'.format(self.df.loc[self.train_idx[idx],'sample']))

            masks = imread_collection(mask_glob).concatenate()
            mask = np.sum(np.stack(masks, 0), 0)/255.0
            seed = np.copy(mask)
            img_sample = self.df.loc[self.train_idx[idx],'sample']
            
            gt_labels = np.zeros((masks.shape[1], masks.shape[2]), np.uint16)
            for index in range(0, masks.shape[0]):
                gt_labels[masks[index] > 0] = index + 1
                
            boundaries = find_boundaries(gt_labels, connectivity=1, mode=self.boundary_mode, background=0) 
           
            if self.erosion_type == 'erode':
                # erode a few layers
                masks_thin = np.asarray([(thin(image=mask,max_iter=3)) for mask in masks])
                # replace mask with the eroded version
                mask = np.sum(np.stack(masks_thin, 0), 0)/255.0            
            else:
                # just remove the boundaries
                seed[boundaries] = 0
            
            # boundaries-only mask
            boundaries = boundaries * 1
            img = imread(img_glob[0])[:,:,0:3]
            
        elif self.mode == 'val':
            img_glob = glob.glob('../data/stage1_train/{}/images/*.png'.format(self.df.loc[self.val_idx[idx],'sample']))
            mask_glob = glob.glob('../data/stage1_train/{}/masks/*.png'.format(self.df.loc[self.val_idx[idx],'sample']))
            
            masks = imread_collection(mask_glob).concatenate()
            mask = np.sum(np.stack(masks, 0), 0)/255.0
            seed = np.copy(mask)
            img_sample = self.df.loc[self.val_idx[idx],'sample']            
            
            gt_labels = np.zeros((masks.shape[1], masks.shape[2]), np.uint16)
            for index in range(0, masks.shape[0]):
                gt_labels[masks[index] > 0] = index + 1
                
            boundaries = find_boundaries(gt_labels, connectivity=1, mode=self.boundary_mode, background=0) 

            if self.erosion_type == 'erode':
                # erode a few layers
                masks_thin = np.asarray([(thin(image=mask,max_iter=3)) for mask in masks])
                # replace mask with the eroded version
                mask = np.sum(np.stack(masks_thin, 0), 0)/255.0            
            else:
                # just remove the boundaries
                seed[boundaries] = 0            
            
            # boundaries-only mask
            boundaries = boundaries * 1
            img = imread(img_glob[0])[:,:,0:3]

        elif self.mode == 'test':
            img_glob = glob.glob('../data/stage1_test/{}/images/*.png'.format(self.df.loc[self.train_idx[idx],'sample']))
            img = imread(img_glob[0])[:,:,0:3]
            
            img_id = self.df.loc[self.train_idx[idx],'img'].split('.')[0]
            
            mask = None
            boundaries = None
            seed = None
            
        # estimage image divisibility by 32
        # if the image is not divisible in any of the dimensions, upscale this dimensions
        w,h = img.shape[0],img.shape[1]
        
        if self.is_crop == False:
            if w % self.factor != 0:
                target_w = (w//self.factor + 1) * self.factor
            else:
                target_w = w
            if h % self.factor != 0:
                target_h = (h//self.factor + 1) * self.factor
            else:
                target_h = h

            if w != target_w or h != target_h:
                img = cv2.resize(img, (target_h,target_w), interpolation=cv2.INTER_LINEAR)

                if mask is not None: 
                    msk = np.stack((mask,boundaries,seed),axis=2)
                    msk = cv2.resize(msk, (target_h,target_w), interpolation=cv2.INTER_LINEAR)
            else:
                if mask is not None: 
                    msk = np.stack((mask,boundaries,seed),axis=2)
        else:
            if mask is not None: 
                msk = np.stack((mask,boundaries,seed),axis=2)            
        
        if self.transforms is not None:
            if mask is not None:            
                # msk = np.stack((mask,boundaries,nuclei_centers),axis=2)
                img, msk = self.transforms(img, msk)
            else:
                msk = 0
                img, _ = self.transforms(img, None)

        if self.is_crop == False:
            # if the image gets flipped, flip it back
            if img.shape[0]!=target_w:
                img = img.transpose(1, 0, 2)
                if mask is not None:
                    msk = msk.transpose(1, 0, 2)

            if self.mode in ['train','val']:
                return img,msk,(w,h),(target_w,target_h),img_sample
            else:
                return img,msk,(w,h),(target_w,target_h),img_id
        else:
            if self.mode in ['train','val']:
                return img,msk,(w,h),(256,256),img_sample
            else:
                return img,msk,(w,h),(256,256),img_id               
            
class BDatasetResizeSeedErode(data.Dataset):
    def __init__(self,
                 df = None,
                 transforms = None,
                 fold_num = 0, # which fold to use
                 mode = 'train', # 'train', 'val' or 'test'
                 dset_resl = '256x256',
                 nuclei_size = 1,
                 factor = 64,
                 is_crop = False,
                 is_img_augs = False,
                 is_distance_transform = False,
                 is_center = False,
                 is_boundaries = False,
                 is_vectors = False,
                 boundary_mode = 'thick'):
        
        self.is_crop = is_crop
        self.is_boundaries = is_boundaries
        self.is_img_augs = is_img_augs
        self.boundary_mode = boundary_mode
        self.is_vectors = is_vectors
        self.is_distance_transform = is_distance_transform
        bad_idx = [53]
        self.df = df
        self.factor = 64
        self.df = self.df.reset_index()
        self.transforms = transforms
        self.fold_num = fold_num
        self.mode = mode
        self.is_center = is_center
        self.nuclei_size = nuclei_size
        self.resolution_dict = resolution_dict
       
        if self.mode in ['train','val']:
            # set indices
            skf = StratifiedKFold(n_splits=4,
                                  shuffle = True,
                                  random_state = 42)

            # stratify fold by cluster number
            f1, f2, f3, f4 = skf.split(self.df.index.values,self.df.cluster.values)
            self.folds = [f1, f2, f3, f4]
            # all train images
            self.train_idx = self.folds[self.fold_num][0]
            self.val_idx = self.folds[self.fold_num][1]
            
            # leave only images with particular resolution
            # also remove idx 53 - broken data there
            self.train_idx = list(set(self.df[(self.df.w_h==dset_resl)&(~self.df.index.isin(bad_idx))].index.values).intersection(set(self.train_idx)))
            self.val_idx = list(set(self.df[(self.df.w_h==dset_resl)&(~self.df.index.isin(bad_idx))].index.values).intersection(set(self.val_idx)))
        else:
            self.train_idx = list(set(self.df[self.df.w_h==dset_resl].index.values))
            
    def __len__(self):
        if self.mode == 'train':
            return len(self.train_idx)
        elif self.mode == 'val':
            return len(self.val_idx)          
        elif self.mode == 'test':  
            return len(self.train_idx) 

    def __getitem__(self, idx):
        if self.mode == 'train':
            img_glob = glob.glob('../data/stage1_train/{}/images/*.png'.format(self.df.loc[self.train_idx[idx],'sample']))
            mask_glob = glob.glob('../data/stage1_train/{}/masks/*.png'.format(self.df.loc[self.train_idx[idx],'sample']))

            masks = imread_collection(mask_glob).concatenate()
            masks = [((mask).astype('uint8')) for mask in masks]
            
            mask = np.sum(np.stack(masks, 0), 0)
            img_sample = self.df.loc[self.train_idx[idx],'sample']

            masks_thin1 = np.asarray([(thin_region_fast(_,1)) for _ in masks])
            masks_thin2 = np.asarray([(thin_region_fast(_,3)) for _ in masks])
            masks_thin3 = np.asarray([(thin_region_fast(_,5)) for _ in masks])
            
            if self.is_center == True:
                nuclei_centers_coord = [(ndimage.measurements.center_of_mass(_) ) for _ in masks]
                nuclei_centers_coord = [(int(_[0]),int(_[1])) for _ in nuclei_centers_coord]
                nuclei_centers = np.zeros((mask.shape[0], mask.shape[1]), dtype=int)

                for coord in nuclei_centers_coord: 
                    nuclei_centers[coord[0],coord[1]] = 1

                nuclei_centers = binary_dilation(nuclei_centers, selem=disk(self.nuclei_size))
                mask0 = (nuclei_centers * 1).astype('uint8')                
            else:
                masks_thin0 = np.asarray([(thin_region_fast(_,7)) for _ in masks])
                mask0 = np.sum(np.stack(masks_thin0, 0), 0).astype('uint8')
            
            if self.is_boundaries == True:
                gt_labels = np.zeros((mask.shape[0], mask.shape[1]), np.uint16)
                for index in range(0, len(masks)):
                    gt_labels[masks[index] > 0] = index + 1
                boundaries = find_boundaries(gt_labels, connectivity=1, mode=self.boundary_mode, background=0)
                boundaries = (boundaries * 1).astype('uint8')
            
            # normalize distances by image size
            # this seems to be a natural way
            if self.is_distance_transform == True:
                masks_distance = np.asarray([(distance_transform_fast(_)) for _ in masks])
                # we lose some accuracy here
                mask_distance = np.sum(np.stack(masks_distance, 0), 0).astype('uint8')
                
            if self.is_vectors == True:
                vectors = np.asarray([(mask2vectors(_)) for _ in masks])
                vectors = (np.sum(np.stack(vectors, 0), 0))
                # encode as int for augmentations
                vectors = ((vectors+1)*125).astype('uint8')
            
            mask1 = np.sum(np.stack(masks_thin1, 0), 0).astype('uint8')
            mask2 = np.sum(np.stack(masks_thin2, 0), 0).astype('uint8')   
            mask3 = np.sum(np.stack(masks_thin3, 0), 0).astype('uint8')
            mask = (mask/255).astype('uint8')

            img = imread(img_glob[0])[:,:,0:3]
            
        elif self.mode == 'val':
            img_glob = glob.glob('../data/stage1_train/{}/images/*.png'.format(self.df.loc[self.val_idx[idx],'sample']))
            mask_glob = glob.glob('../data/stage1_train/{}/masks/*.png'.format(self.df.loc[self.val_idx[idx],'sample']))
            
            masks = imread_collection(mask_glob).concatenate()
            masks = [((mask).astype('uint8')) for mask in masks]
            
            mask = np.sum(np.stack(masks, 0), 0)
            img_sample = self.df.loc[self.val_idx[idx],'sample']            
            
            masks_thin1 = np.asarray([(thin_region_fast(_,1)) for _ in masks])
            masks_thin2 = np.asarray([(thin_region_fast(_,3)) for _ in masks])
            masks_thin3 = np.asarray([(thin_region_fast(_,5)) for _ in masks])
            
            if self.is_center == True:
                nuclei_centers_coord = [(ndimage.measurements.center_of_mass(_) ) for _ in masks]
                nuclei_centers_coord = [(int(_[0]),int(_[1])) for _ in nuclei_centers_coord]
                nuclei_centers = np.zeros((mask.shape[0], mask.shape[1]), dtype=int)

                for coord in nuclei_centers_coord: 
                    nuclei_centers[coord[0],coord[1]] = 1

                nuclei_centers = binary_dilation(nuclei_centers, selem=disk(self.nuclei_size))
                mask0 = (nuclei_centers * 1).astype('uint8')                
            else:
                masks_thin0 = np.asarray([(thin_region_fast(_,7)) for _ in masks])
                mask0 = np.sum(np.stack(masks_thin0, 0), 0).astype('uint8')
            
            if self.is_boundaries == True:
                gt_labels = np.zeros((mask.shape[0], mask.shape[1]), np.uint16)
                for index in range(0, len(masks)):
                    gt_labels[masks[index] > 0] = index + 1
                boundaries = find_boundaries(gt_labels, connectivity=1, mode=self.boundary_mode, background=0)
                boundaries = (boundaries * 1).astype('uint8')             
            
            # normalize distances by image size
            # this seems to be a natural way
            if self.is_distance_transform == True:
                masks_distance = np.asarray([(distance_transform_fast(_)) for _ in masks])
                # we lose some accuracy here
                mask_distance = np.sum(np.stack(masks_distance, 0), 0).astype('uint8')
                
            if self.is_vectors == True:
                vectors = np.asarray([(mask2vectors(_)) for _ in masks])
                vectors = (np.sum(np.stack(vectors, 0), 0))
                # encode as int for augmentations
                vectors = ((vectors+1)*125).astype('uint8')                
            
            mask1 = np.sum(np.stack(masks_thin1, 0), 0).astype('uint8')
            mask2 = np.sum(np.stack(masks_thin2, 0), 0).astype('uint8')   
            mask3 = np.sum(np.stack(masks_thin3, 0), 0).astype('uint8')
            mask = (mask/255).astype('uint8')
            
            img = imread(img_glob[0])[:,:,0:3]

        elif self.mode == 'test':
            img_glob = glob.glob('../data/stage1_test/{}/images/*.png'.format(self.df.loc[self.train_idx[idx],'sample']))
            img = imread(img_glob[0])[:,:,0:3]
            
            img_id = self.df.loc[self.train_idx[idx],'img'].split('.')[0]
            
            mask = None
            boundaries = None
            seed = None
            
        # estimage image divisibility by 32
        # if the image is not divisible in any of the dimensions, upscale this dimensions
        w,h = img.shape[0],img.shape[1]
        
        if self.is_crop == False:
            if w % self.factor != 0:
                target_w = (w//self.factor + 1) * self.factor
            else:
                target_w = w
            if h % self.factor != 0:
                target_h = (h//self.factor + 1) * self.factor
            else:
                target_h = h
           
            if w != target_w or h != target_h:
                img = cv2.resize(img, (target_h,target_w), interpolation=cv2.INTER_LINEAR)

                if mask is not None:
                    
                    if self.is_vectors == True and self.is_boundaries == True:
                        msk = np.stack((mask,mask1,mask2,mask3,mask0,boundaries,vectors[0],vectors[1]),axis=2)
                    else:
                        if self.is_distance_transform == True:
                            if self.is_boundaries == True:
                                msk = np.stack((mask,mask1,mask2,mask3,mask0,mask_distance,boundaries),axis=2)
                            else:
                                msk = np.stack((mask,mask1,mask2,mask3,mask0,mask_distance),axis=2)
                        else:
                            msk = np.stack((mask,mask1,mask2,mask3,mask0),axis=2)
                        
                    msk = cv2.resize(msk, (target_h,target_w), interpolation=cv2.INTER_LINEAR)
            else:
                if mask is not None: 
                    
                    if self.is_vectors == True and self.is_boundaries == True:
                        msk = np.stack((mask,mask1,mask2,mask3,mask0,boundaries,vectors[0],vectors[1]),axis=2)
                    else:
                        if self.is_distance_transform == True:
                            if self.is_boundaries == True:
                                msk = np.stack((mask,mask1,mask2,mask3,mask0,mask_distance,boundaries),axis=2)
                            else:
                                msk = np.stack((mask,mask1,mask2,mask3,mask0,mask_distance),axis=2)
                        else:
                            msk = np.stack((mask,mask1,mask2,mask3,mask0),axis=2)
        else:
            if mask is not None: 
                
                if self.is_vectors == True and self.is_boundaries == True:
                    msk = np.stack((mask,mask1,mask2,mask3,mask0,boundaries,vectors[0],vectors[1]),axis=2)
                else:
                    if self.is_distance_transform == True:
                        if self.is_boundaries == True:
                            msk = np.stack((mask,mask1,mask2,mask3,mask0,mask_distance,boundaries),axis=2)
                        else:
                            msk = np.stack((mask,mask1,mask2,mask3,mask0,mask_distance),axis=2)
                    else:
                        msk = np.stack((mask,mask1,mask2,mask3,mask0),axis=2)       

        if self.transforms is not None:
            if mask is not None:    
                if self.is_img_augs == True:
                    img = seq.augment_image(img)
                img, msk = self.transforms(img, msk)
            else:
                msk = 0
                if self.is_img_augs == True:
                    img = seq.augment_image(img)                
                img, _ = self.transforms(img, None)

        
        # do not forget to normalize the max possible distance
        if self.is_distance_transform == True:
            msk = msk.astype(float)
            max_dist = msk[:,:,5].max()
            if max_dist == 0:
                max_dist = 1
            msk[:,:,5] = msk[:,:,5] / max_dist
            
        # do not forget to convert vectors back to float  
        if self.is_vectors == True and self.is_boundaries == True:
            msk = msk.astype(float)
            msk[:,:,6] = msk[:,:,6]/125-1
            msk[:,:,7] = msk[:,:,7]/125-1
                
        if self.is_crop == False:
            # if the image gets flipped, flip it back
            if img.shape[0]!=target_w:
                img = img.transpose(1, 0, 2)
                if mask is not None:
                    msk = msk.transpose(1, 0, 2)

            if self.mode in ['train','val']:
                return img,msk,(w,h),(target_w,target_h),img_sample
            else:
                return img,msk,(w,h),(target_w,target_h),img_id
        else:
            if self.mode in ['train','val']:
                return img,msk,(w,h),(256,256),img_sample
            else:
                return img,msk,(w,h),(256,256),img_id                           
            
class BDatasetPad(data.Dataset):
    def __init__(self,
                 df = None,
                 transforms = None,
                 fold_num = 0, # which fold to use
                 mode = 'train', # 'train', 'val' or 'test'
                 dset_resl = '256x256',
                 nuclei_size = 1,
                 factor = 64,
                 is_crop = False,
                 is_img_augs = False,
                 is_distance_transform = False,
                 is_center = False,
                 is_boundaries = False,
                 is_vectors = False,
                 boundary_mode = 'thick'):
        
        self.is_crop = is_crop
        self.is_boundaries = is_boundaries
        self.is_img_augs = is_img_augs
        self.boundary_mode = boundary_mode
        self.is_vectors = is_vectors
        self.is_distance_transform = is_distance_transform
        bad_idx = [53]
        self.df = df
        self.factor = 64
        self.df = self.df.reset_index()
        self.transforms = transforms
        self.fold_num = fold_num
        self.mode = mode
        self.is_center = is_center
        self.nuclei_size = nuclei_size
        self.resolution_dict = resolution_dict
       
        if self.mode in ['train','val']:
            # set indices
            skf = StratifiedKFold(n_splits=4,
                                  shuffle = True,
                                  random_state = 42)

            # stratify fold by cluster number
            f1, f2, f3, f4 = skf.split(self.df.index.values,self.df.cluster.values)
            self.folds = [f1, f2, f3, f4]
            # all train images
            self.train_idx = self.folds[self.fold_num][0]
            self.val_idx = self.folds[self.fold_num][1]
            
            # leave only images with particular resolution
            # also remove idx 53 - broken data there
            self.train_idx = list(set(self.df[(self.df.w_h==dset_resl)&(~self.df.index.isin(bad_idx))].index.values).intersection(set(self.train_idx)))
            self.val_idx = list(set(self.df[(self.df.w_h==dset_resl)&(~self.df.index.isin(bad_idx))].index.values).intersection(set(self.val_idx)))
        else:
            self.train_idx = list(set(self.df[self.df.w_h==dset_resl].index.values))
            
    def __len__(self):
        if self.mode == 'train':
            return len(self.train_idx)
        elif self.mode == 'val':
            return len(self.val_idx)          
        elif self.mode == 'test':  
            return len(self.train_idx) 

    def __getitem__(self, idx):
        if self.mode == 'train':
            img_glob = glob.glob('../data/stage1_train/{}/images/*.png'.format(self.df.loc[self.train_idx[idx],'sample']))
            mask_glob = glob.glob('../data/stage1_train/{}/masks/*.png'.format(self.df.loc[self.train_idx[idx],'sample']))

            masks = imread_collection(mask_glob).concatenate()
            masks = [((mask).astype('uint8')) for mask in masks]
            
            mask = np.sum(np.stack(masks, 0), 0)
            img_sample = self.df.loc[self.train_idx[idx],'sample']

            masks_thin1 = np.asarray([(thin_region_fast(_,1)) for _ in masks])
            masks_thin2 = np.asarray([(thin_region_fast(_,3)) for _ in masks])
            masks_thin3 = np.asarray([(thin_region_fast(_,5)) for _ in masks])
            
            if self.is_center == True:
                nuclei_centers_coord = [(ndimage.measurements.center_of_mass(_) ) for _ in masks]
                nuclei_centers_coord = [(int(_[0]),int(_[1])) for _ in nuclei_centers_coord]
                nuclei_centers = np.zeros((mask.shape[0], mask.shape[1]), dtype=int)

                for coord in nuclei_centers_coord: 
                    nuclei_centers[coord[0],coord[1]] = 1

                nuclei_centers = binary_dilation(nuclei_centers, selem=disk(self.nuclei_size))
                mask0 = (nuclei_centers * 1).astype('uint8')                
            else:
                masks_thin0 = np.asarray([(thin_region_fast(_,7)) for _ in masks])
                mask0 = np.sum(np.stack(masks_thin0, 0), 0).astype('uint8')
            
            if self.is_boundaries == True:
                gt_labels = np.zeros((mask.shape[0], mask.shape[1]), np.uint16)
                for index in range(0, len(masks)):
                    gt_labels[masks[index] > 0] = index + 1
                boundaries = find_boundaries(gt_labels, connectivity=1, mode=self.boundary_mode, background=0)
                boundaries = (boundaries * 1).astype('uint8')
            
            # normalize distances by image size
            # this seems to be a natural way
            if self.is_distance_transform == True:
                masks_distance = np.asarray([(distance_transform_fast(_)) for _ in masks])
                # we lose some accuracy here
                mask_distance = np.sum(np.stack(masks_distance, 0), 0).astype('uint8')
                
            if self.is_vectors == True:
                vectors = np.asarray([(mask2vectors(_)) for _ in masks])
                vectors = (np.sum(np.stack(vectors, 0), 0))
                # encode as int for augmentations
                vectors = ((vectors+1)*125).astype('uint8')
            
            mask1 = np.sum(np.stack(masks_thin1, 0), 0).astype('uint8')
            mask2 = np.sum(np.stack(masks_thin2, 0), 0).astype('uint8')   
            mask3 = np.sum(np.stack(masks_thin3, 0), 0).astype('uint8')
            mask = (mask/255).astype('uint8')

            img = imread(img_glob[0])[:,:,0:3]
            
        elif self.mode == 'val':
            img_glob = glob.glob('../data/stage1_train/{}/images/*.png'.format(self.df.loc[self.val_idx[idx],'sample']))
            mask_glob = glob.glob('../data/stage1_train/{}/masks/*.png'.format(self.df.loc[self.val_idx[idx],'sample']))
            
            masks = imread_collection(mask_glob).concatenate()
            masks = [((mask).astype('uint8')) for mask in masks]
            
            mask = np.sum(np.stack(masks, 0), 0)
            img_sample = self.df.loc[self.val_idx[idx],'sample']            
            
            masks_thin1 = np.asarray([(thin_region_fast(_,1)) for _ in masks])
            masks_thin2 = np.asarray([(thin_region_fast(_,3)) for _ in masks])
            masks_thin3 = np.asarray([(thin_region_fast(_,5)) for _ in masks])
            
            if self.is_center == True:
                nuclei_centers_coord = [(ndimage.measurements.center_of_mass(_) ) for _ in masks]
                nuclei_centers_coord = [(int(_[0]),int(_[1])) for _ in nuclei_centers_coord]
                nuclei_centers = np.zeros((mask.shape[0], mask.shape[1]), dtype=int)

                for coord in nuclei_centers_coord: 
                    nuclei_centers[coord[0],coord[1]] = 1

                nuclei_centers = binary_dilation(nuclei_centers, selem=disk(self.nuclei_size))
                mask0 = (nuclei_centers * 1).astype('uint8')                
            else:
                masks_thin0 = np.asarray([(thin_region_fast(_,7)) for _ in masks])
                mask0 = np.sum(np.stack(masks_thin0, 0), 0).astype('uint8')
            
            if self.is_boundaries == True:
                gt_labels = np.zeros((mask.shape[0], mask.shape[1]), np.uint16)
                for index in range(0, len(masks)):
                    gt_labels[masks[index] > 0] = index + 1
                boundaries = find_boundaries(gt_labels, connectivity=1, mode=self.boundary_mode, background=0)
                boundaries = (boundaries * 1).astype('uint8')             
            
            # normalize distances by image size
            # this seems to be a natural way
            if self.is_distance_transform == True:
                masks_distance = np.asarray([(distance_transform_fast(_)) for _ in masks])
                # we lose some accuracy here
                mask_distance = np.sum(np.stack(masks_distance, 0), 0).astype('uint8')
                
            if self.is_vectors == True:
                vectors = np.asarray([(mask2vectors(_)) for _ in masks])
                vectors = (np.sum(np.stack(vectors, 0), 0))
                # encode as int for augmentations
                vectors = ((vectors+1)*125).astype('uint8')                
            
            mask1 = np.sum(np.stack(masks_thin1, 0), 0).astype('uint8')
            mask2 = np.sum(np.stack(masks_thin2, 0), 0).astype('uint8')   
            mask3 = np.sum(np.stack(masks_thin3, 0), 0).astype('uint8')
            mask = (mask/255).astype('uint8')
            
            img = imread(img_glob[0])[:,:,0:3]

        elif self.mode == 'test':
            img_glob = glob.glob('../data/stage1_test/{}/images/*.png'.format(self.df.loc[self.train_idx[idx],'sample']))
            img = imread(img_glob[0])[:,:,0:3]
            
            img_id = self.df.loc[self.train_idx[idx],'img'].split('.')[0]
            
            mask = None
            boundaries = None
            seed = None
            
        # estimage image divisibility by 32
        # if the image is not divisible in any of the dimensions, upscale this dimensions
        w,h = img.shape[0],img.shape[1]
        
       
        if self.is_crop == False:
            if w % self.factor != 0:
                target_w = (w//self.factor + 1) * self.factor
            else:
                target_w = w
            if h % self.factor != 0:
                target_h = (h//self.factor + 1) * self.factor
            else:
                target_h = h
           
            if w != target_w or h != target_h:
                
                add_x = (target_w-w) // 2
                add_y = (target_h-h) // 2
                
                if (target_w-w)%2>0:
                    add_x_1 = 1
                else:
                    add_x_1 = 0
                if (target_h-h)%2>0:
                    add_y_1 = 1
                else:
                    add_y_1 = 0
                    
                img = np.pad(array=img,
                          pad_width=((add_x,add_x+add_x_1),(add_y,add_y_1+add_y),(0,0)),
                          mode='constant')

                if mask is not None:
                    
                    if self.is_vectors == True and self.is_boundaries == True:
                        msk = np.stack((mask,mask1,mask2,mask3,mask0,boundaries,vectors[0],vectors[1]),axis=2)
                    else:
                        if self.is_distance_transform == True:
                            if self.is_boundaries == True:
                                msk = np.stack((mask,mask1,mask2,mask3,mask0,mask_distance,boundaries),axis=2)
                            else:
                                msk = np.stack((mask,mask1,mask2,mask3,mask0,mask_distance),axis=2)
                        else:
                            msk = np.stack((mask,mask1,mask2,mask3,mask0),axis=2)
                        
                    msk = pad(array=msk,
                              pad_width=[(target_w-w)//2,(target_h-h)//2],
                              mode='constant')                    
            else:
                if mask is not None: 
                    
                    if self.is_vectors == True and self.is_boundaries == True:
                        msk = np.stack((mask,mask1,mask2,mask3,mask0,boundaries,vectors[0],vectors[1]),axis=2)
                    else:
                        if self.is_distance_transform == True:
                            if self.is_boundaries == True:
                                msk = np.stack((mask,mask1,mask2,mask3,mask0,mask_distance,boundaries),axis=2)
                            else:
                                msk = np.stack((mask,mask1,mask2,mask3,mask0,mask_distance),axis=2)
                        else:
                            msk = np.stack((mask,mask1,mask2,mask3,mask0),axis=2)
        else:
            if mask is not None: 
                
                if self.is_vectors == True and self.is_boundaries == True:
                    msk = np.stack((mask,mask1,mask2,mask3,mask0,boundaries,vectors[0],vectors[1]),axis=2)
                else:
                    if self.is_distance_transform == True:
                        if self.is_boundaries == True:
                            msk = np.stack((mask,mask1,mask2,mask3,mask0,mask_distance,boundaries),axis=2)
                        else:
                            msk = np.stack((mask,mask1,mask2,mask3,mask0,mask_distance),axis=2)
                    else:
                        msk = np.stack((mask,mask1,mask2,mask3,mask0),axis=2)       

        if self.transforms is not None:
            if mask is not None:    
                if self.is_img_augs == True:
                    img = seq.augment_image(img)
                img, msk = self.transforms(img, msk)
            else:
                msk = 0
                if self.is_img_augs == True:
                    img = seq.augment_image(img)                
                img, _ = self.transforms(img, None)

        # do not forget to normalize the max possible distance
        if self.is_distance_transform == True:
            msk = msk.astype(float)
            max_dist = msk[:,:,5].max()
            if max_dist == 0:
                max_dist = 1
            msk[:,:,5] = msk[:,:,5] / max_dist
            
        # do not forget to convert vectors back to float  
        if self.is_vectors == True and self.is_boundaries == True:
            msk = msk.astype(float)
            msk[:,:,6] = msk[:,:,6]/125-1
            msk[:,:,7] = msk[:,:,7]/125-1
                
        if self.is_crop == False:
            # if the image gets flipped, flip it back
            if img.shape[0]!=target_w:
                img = img.transpose(1, 0, 2)
                if mask is not None:
                    msk = msk.transpose(1, 0, 2)

            if self.mode in ['train','val']:
                return img,msk,(w,h),(target_w,target_h),img_sample
            else:
                return img,msk,(w,h),(target_w,target_h),img_id
        else:
            if self.mode in ['train','val']:
                return img,msk,(w,h),(256,256),img_sample
            else:
                return img,msk,(w,h),(256,256),img_id            