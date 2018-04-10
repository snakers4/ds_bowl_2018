import glob
import numpy as np

from scipy import ndimage as ndi
import cv2

import skimage
from skimage.morphology import square
from skimage.morphology import dilation
from skimage.morphology import watershed
from skimage.morphology import binary_erosion
from skimage.feature import peak_local_max
from skimage.io import imread,imread_collection
from skimage.segmentation import find_boundaries
from skimage.filters import sobel
from skimage.measure import label,regionprops
from skimage import exposure
from skimage.morphology import thin
from skimage.filters import threshold_otsu
from skimage.morphology import binary_dilation
from skimage.morphology import disk
from skimage.feature import peak_local_max
from skimage.feature import blob_log

from scipy import ndimage as ndi

cv2.setNumThreads(0)

def wt_baseline(img = None,
              threshold = 0.5):

    # Make segmentation using edge-detection and watershed.
    edges = sobel(img)

    # Identify some background and foreground pixels from the intensity values.
    # These pixels are used as seeds for watershed.
    markers = np.zeros_like(img)
    foreground, background = 1, 2
    markers[img < threshold * 255 // 2] = background
    markers[img > threshold * 255] = foreground

    ws = watershed(edges, markers)
    labels = label(ws == foreground)
    
    return labels

def wt_seeds(img = None,
             seed = None,
             threshold = 0.5):

    # img and seed are 0-255 images

    # threshold the img
    img = 1 * (img > 255 * threshold)
    
    # filter low confidence values for the seed
    # seed[seed < 255 * threshold] = 0
    seed = 1 * (seed > 255 * threshold)
    
    seed_labels = label(seed)
    
    # remove the background label and count seed objects
    seeds = []
    # ignore the background
    for i in range(1,seed_labels.max()):
        seeds.append((seed_labels==i)*1)

    # find nuclei centers by measuring mass center
    seed_center_coords = [(ndi.measurements.center_of_mass(_) ) for _ in seeds]
    seed_center_coords = [(int(_[0]),int(_[1])) for _ in seed_center_coords]

    seed_centers = np.zeros((img.shape[0], img.shape[1]), dtype=bool)
    
    # create a mask with seed centers
    for coord in seed_center_coords:
        seed_centers[coord[0],coord[1]] = True

    # pad 10 pixels to masks
    # border objects are found better this way
    
    img_mask = skimage.util.pad(array=img,
                                pad_width=10,
                                mode='constant')
    seed_centers = skimage.util.pad(array=seed_centers,
                                    pad_width=10,
                                    mode='constant',
                                    constant_values=[True])


    # Now we want to separate the two objects in image
    # Generate the markers as local maxima of the distance to the background
    distance = ndi.distance_transform_edt(img_mask)

    markers = ndi.label(seed_centers)[0]

    labels = watershed(-distance,
                       markers,
                       mask=img_mask)

    labels = labels[10:-10,10:-10]
    img_mask = img_mask[10:-10,10:-10]
    
    return labels

def wt_seeds2(img = None,
             seed = None,
             threshold = 0.5):

    # img and seed are 0-255 images

    # threshold the img
    img_ths = 1 * (img > 255 * threshold)
    
    # filter low confidence values for the seed
    seed_ths = 1 * (seed > 255 * threshold)

    # pad 10 pixels to masks
    # border objects are found better this way
    
    img_ths = skimage.util.pad(array=img_ths,
                                pad_width=10,
                                mode='constant')
    
    seed_ths = skimage.util.pad(array=seed_ths,
                                pad_width=10,
                                mode='constant')    


    # Now we want to separate the two objects in image
    # Generate the markers as local maxima of the distance to the background
    distance = ndi.distance_transform_edt(seed_ths)

    # also threshold the distance
    markers = ndi.label(distance>0.5*distance.max())[0]

    labels = watershed(-distance,
                       markers,
                       mask=img_ths)

    labels = labels[10:-10,10:-10]
    img_ths = img_ths[10:-10,10:-10]
    
    return labels

def wt_seeds3(img = None,
             seed = None,
             threshold = 0.5):
    
    mask = (img>255*threshold) * 1

    mask = skimage.util.pad(array=mask,
                            pad_width=10,
                            mode='constant')

    seed = skimage.util.pad(array=seed,
                            pad_width=10,
                            mode='constant')

    ret, thresh = cv2.threshold(seed,0,255,cv2.THRESH_OTSU)

    # noise removal
    kernel = np.ones((3,3),np.uint8)
    opening = cv2.morphologyEx(thresh,cv2.MORPH_OPEN,kernel, iterations = 2)
    # sure background area
    sure_bg = cv2.dilate(opening,kernel,iterations=3)

    # Finding sure foreground area
    dist_transform = cv2.distanceTransform(opening,cv2.DIST_L2,5)

    # adaptive hist for distance transform
    dist_transform = exposure.equalize_adapthist(((dist_transform / dist_transform.max()) * 255).astype('uint8'), clip_limit=0.03)

    ret, sure_fg = cv2.threshold(dist_transform,0.7*dist_transform.max(),255,0)

    # Finding unknown region
    sure_fg = np.uint8(sure_fg)
    unknown = cv2.subtract(sure_bg,sure_fg)

    # Marker labelling
    ret, markers = cv2.connectedComponents(sure_fg)

    distance = ndi.distance_transform_edt(mask)

    labels = watershed(-distance,
                       markers,
                       mask=mask)
    
    return labels[10:-10,10:-10]

def label_baseline(msk1 =None,
                   threshold = 0.5):
    
    labels = (np.copy(msk1)>255*threshold)*1
    labels = label(labels)
    
    return labels

def energy_baseline(msk = None,
                    energy = None,
                    threshold = 0.5,
                    thin_labels = False):

    msk_ths = (np.copy(msk)>255*threshold)*1
    energy_ths = (np.copy(energy)>255*0.4)*1

    distance = ndi.distance_transform_edt(msk_ths)
    
    # Marker labelling
    markers = label(energy_ths)    

    labels = watershed(-distance,
                       markers,
                       mask=msk_ths)

    if thin_labels == True:
        for i,lbl in enumerate(np.unique(labels)):
            if i == 0:
                # pass the background
                pass
            else:
                current_label = (labels==lbl) * 1
                thinned_label = thin(current_label,max_iter=1)
                labels[labels==lbl] = 0
                labels[thinned_label] = lbl

    return labels

def energy_baseline_thin(msk = None,
                    energy = None,
                    threshold = 0.5):

    msk_ths = (np.copy(msk)>255*threshold)*1
    energy_ths = (np.copy(energy)>255*0.4)*1

    distance = ndi.distance_transform_edt(msk_ths)
    
    energy_ths = thin(energy_ths,max_iter=3)
    
    # Marker labelling
    markers = label(energy_ths)    

    labels = watershed(-distance,
                       markers,
                       mask=msk_ths)
    
    return labels

def energy_baseline_clahe(msk = None,
                    energy = None,
                    threshold = 0.5):

    msk_ths = (np.copy(msk)>255*threshold)*1
    energy_ths = (np.copy(energy)>threshold_otsu(energy))*1

    distance = ndi.distance_transform_edt(msk_ths)
    
    # Marker labelling
    markers = label(energy_ths)    

    labels = watershed(-distance,
                       markers,
                       mask=msk_ths)
    return labels

def energy_baseline_markers(msk = None,
                    energy = None,
                    threshold = 0.5):

    msk_ths = (np.copy(msk)>255*threshold)*1
    
    energy_ths =  np.copy(energy)
    energy_ths[energy_ths<255*0.4] = 0
    coordinates = peak_local_max(energy_ths, min_distance=3)
    markers = np.zeros_like(energy_ths)

    for coord in coordinates:
        markers[coord[0],coord[1]] = 1

    markers = binary_dilation(markers, selem=disk(1)) 
    markers = label(markers)  

    distance = ndi.distance_transform_edt(msk_ths)
    
    # Marker labelling 
    labels = watershed(-distance,
                       markers,
                       mask=msk_ths)
    return labels

def energy_baseline_blob(msk = None,
                    energy = None,
                    threshold = 0.5,
                    energy_ths = 0.2):
    
    msk_ths = (np.copy(msk)>255*threshold)*1
    energy[energy < 255 * energy_ths] = 0
    energy = energy.astype('uint8')
    
    regions = regionprops(label(msk_ths))
    
    max_radius = 0
    min_radius = 100
    
    for props in regions:
        if props.equivalent_diameter/2 > max_radius:
            max_radius = props.equivalent_diameter/2
        if props.equivalent_diameter/2 < min_radius:
            min_radius = props.equivalent_diameter/2                
                        
    min_radius = max(min_radius,2)
        
    blobs_log = blob_log(energy,
                         min_sigma=min_radius,
                         max_sigma=max_radius,
                         num_sigma=10,
                         threshold=.1)
    
    markers = np.zeros_like(energy)

    for blob in blobs_log:
        markers[int(blob[0]),int(blob[1])] = 1

    # markers = binary_dilation(markers, selem=disk(1)) 
    markers = label(markers)  
    
    distance = ndi.distance_transform_edt(msk_ths)
    
    # Marker labelling 
    labels = watershed(-distance,
                       markers,
                       mask=msk_ths)
    
    return labels

def mixed_wt(msk = None,
            energy = None,
            threshold = 0.5,
            energy_ths = 0.2):
    
    msk_ths = (np.copy(msk)>255*threshold)*1
    regions = regionprops(label(msk_ths))
    
    max_radius = 0
    min_radius = 100
    
    for props in regions:
        if props.equivalent_diameter/2 > max_radius:
            max_radius = props.equivalent_diameter/2
    
    if max_radius > 20:
        return energy_baseline(msk = msk,energy = energy,threshold = threshold)
    else:
        return energy_baseline_blob(msk = msk,energy = energy,threshold = threshold)
    
def mixed_wt2(msk = None,
            energy = None,
            threshold = 0.5,
            energy_ths = 0.4):
    
    msk_ths = (np.copy(msk)>255*threshold)*1
    energy_threshold = np.copy(energy)

    distance = ndi.distance_transform_edt(msk_ths)

    # add local maxima to markers as nuclei "centers"

    energy[energy < 255 * energy_ths] = 0
    energy = energy.astype('uint8')

    regions = regionprops(label(msk_ths))

    max_radius = 0
    min_radius = 100

    for props in regions:
        if props.equivalent_diameter/2 > max_radius:
            max_radius = props.equivalent_diameter/2
        if props.equivalent_diameter/2 < min_radius:
            min_radius = props.equivalent_diameter/2                

    min_radius = max(min_radius,2)

    blobs_log = blob_log(energy,
                         min_sigma=min_radius,
                         max_sigma=max_radius,
                         num_sigma=10,
                         threshold=.1)

    markers = np.zeros_like(energy)

    for blob in blobs_log:
        markers[int(blob[0]),int(blob[1])] = 1

    markers = binary_dilation(markers, selem=disk(2))

    energy_threshold = markers*255/4 + energy_threshold*(3/4)
    energy_threshold = (energy_threshold>255*energy_ths)*1  

    # Marker labelling
    markers = label(energy_threshold)    

    labels = watershed(-distance,
                       markers,
                       mask=msk_ths)
    
    return labels

def energy_baseline_blob2(msk = None,
                    energy = None,
                    threshold = 0.5,
                    energy_ths = 0.4):

    msk_ths = (np.copy(msk)>255*threshold)*1
    energy[energy < 255 * energy_ths] = 0
    energy = energy.astype('uint8')

    regions = regionprops(label(msk_ths))

    max_radius = 0
    min_radius = 100

    for props in regions:
        if props.equivalent_diameter/2 > max_radius:
            max_radius = props.equivalent_diameter/2
        if props.equivalent_diameter/2 < min_radius:
            min_radius = props.equivalent_diameter/2                

    min_radius = max(min_radius,2)

    # estimate kernel nuclei centers
    blobs_log = blob_log(energy,
                         min_sigma=min_radius,
                         max_sigma=max_radius,
                         num_sigma=10,
                         threshold=.1)

    markers = []

    # draw nuclei centers with decay
    for blob in blobs_log:
        marker = np.zeros_like(energy).astype('float')
        marker[int(blob[0]),int(blob[1])] = 1

        dsk = disk(int(blob[2])).astype('float')
        for i in range(1,dsk.shape[0]//2+1):
            dsk[i:-i,i:-i] = dsk[i:-i,i:-i] * 1.4
        dsk = dsk/dsk.max()    

        # find a fix for a case then the center is touching the frame
        try:
            marker[int(blob[0])-dsk.shape[0]//2:int(blob[0])+dsk.shape[0]//2+1,int(blob[1])-dsk.shape[0]//2:int(blob[1])+dsk.shape[0]//2+1] = dsk
        except:
            pass
        # marker = dilation(marker, selem=disk(round(blob[2],0))) 
        markers.append(marker)

    try:
        markers = np.sum(np.stack(markers, 0), 0)
        a = 0.25
        b = 0.75
        markers = (a*markers*255+b*energy)>255*energy_ths        
    except:
        markers = energy>255*energy_ths


    markers = label(markers)  
    distance = ndi.distance_transform_edt(msk_ths)

    # Marker labelling 
    labels = watershed(-distance,
                       markers,
                       mask=msk_ths)
    
    return labels