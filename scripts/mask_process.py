import numpy as np
import json
from pycocotools import mask

def mask2rle(img):
    '''
    img: numpy array, 1 - mask, 0 - background
    Returns run length as string formated
    '''
    pixels = img.T.flatten()
    pixels = np.concatenate([[0], pixels, [0]])
    runs = np.where(pixels[1:] != pixels[:-1])[0] + 1
    runs[1::2] -= runs[::2]
    return ' '.join(str(x) for x in runs)
 
 
def trans_anns(anns):
    if len(anns) == 0:
        return
    sorted_anns = sorted(anns, key=(lambda x: x['area']), reverse=False)
    
    list = []
    index = 0
    # transfer every box
    for ann in sorted_anns:
        bool_array = ann['segmentation']
        int_array = bool_array.astype(int)
        # rel format
        rle = mask2rle(int_array)
        list.append({"index": index, "mask": rle})
        index += 1
    return list


def create_automask_rel_output(image,automask_data):
    mask_list = trans_anns(automask_data)
    coco_rle = coco_rle_mask(automask_data)
    mask_obj = {
        "mask_list": mask_list,
        "coco_rle": coco_rle
    }
    
    return mask_obj

def coco_rle_mask(mask_anns):
    # 处理sam mask ann
    if len(mask_anns) == 0:
        return
    sorted_anns = sorted(mask_anns, key=(lambda x: x['area']), reverse=False)
    
    list = []
    index = 0
    # transfer every box
    for ann in sorted_anns:
        bool_array = ann['segmentation']
        rle = mask.encode(np.asfortranarray(bool_array))
            
        list.append({"index": index, "mask": rle})
        index += 1
    return list    
    


