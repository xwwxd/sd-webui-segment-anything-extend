import numpy as np
import json

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
    # 对每个注释进行处理
    for ann in sorted_anns:
        bool_array = ann['segmentation']
        # 将boolean类型的数组转换为int类型
        int_array = bool_array.astype(int)
        # 转化为RLE格式
        rle = mask2rle(int_array)
        list.append({"index": index, "mask": rle})
        index += 1
    return list


def create_automask_rel_output(image,automask_data):
    # 处理sam返回的图层信息
    mask_list = trans_anns(automask_data)
    
    mask_obj = {
        # "height": image.shape[0],
        # "width": image.shape[1],
        "mask_list": mask_list
    }
    # print(json.dumps(mask_obj))
    
    return mask_obj

