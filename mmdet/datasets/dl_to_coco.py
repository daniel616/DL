import os.path as osp

import mmcv
import numpy as np
from skimage import io
from mmcv.parallel import DataContainer as DC
from torch.utils.data import Dataset

from .transforms import (ImageTransform, BboxTransform, MaskTransform,
                         SegMapTransform, Numpy2Tensor)
from .utils import to_tensor, random_scale
from .extra_aug import ExtraAugmentation


import pycocotools.mask as maskUtils
import pandas as pd
import json


def to_coco(csv,out_file):
    df= pd.read_csv(csv)
    df['id']=df.index
    df['file_name']=df.File_name.apply(_conv)
    intermed = [v for k, v in df.to_dict(orient='index').items()]
    annotations=[]
    img_infos=[]
    import ipdb; ipdb.set_trace()
    for x in intermed:
        #import pdb; pdb.set_trace()
        s_range= [int(t) for t in x['Slice_range'].split(", ")]
        key=x['Key_slice_index']
        adjacents=[max(key-1,s_range[0]),min(key+1,s_range[1])]
        adjacents=[str(t) for t in adjacents]
        directory="_".join(x['file_name'].split("_")[:-1])
        ctx1=osp.join(directory,adjacents[0])
        ctx2=osp.join(directory,adjacents[1])

        img_info={
            'file_name':x['file_name'],
            'ctx1':ctx1,
            'ctx2':ctx2,
            'height':512,
            'width':512,
            'id':x['id']
        }

        b=[ float(t) for t in x['Bounding_boxes'].split(", ")]
        box_coco=[b[0],b[1],b[2]-b[0],b[3]-b[1]]
        segpoints=get_seg(x['Measurement_coordinates'])
        rles=maskUtils.frPyObjects(segpoints,512,512)
        rle=maskUtils.merge(rles)
        area=maskUtils.area(rle)

        ann_info={
            "segmentation":segpoints,
            'image_id':x['id'],
            'id':x['id'],
            'iscrowd':0,
            'category_id':1,
            'area':float(area),
            'bbox':box_coco
        }
        img_infos.append(img_info)
        annotations.append(ann_info)

    categories=[{'supercategory':'lesion','id':1,'name':'lesion'}]

    dataset={
        "images":img_infos,
        'annotations':annotations,
        'categories':categories
    }
    
    with open(out_file,'w+') as json_file:
        json.dump(dataset,json_file)

def get_seg(st):
    v = [float(x) for x in st.split(", ")]

    return [[v[0],v[1],v[4],v[5],v[2],v[3],v[6],v[7]]]

def _conv(x):
    parts = x.split("_")
    pre = "_".join(parts[:-1])
    return osp.join(pre, parts[-1])
