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

import pandas as pd
import json


def to_coco(csv,out_file):
    df= pd.read_csv(csv,usecols=['File_name','Bounding_boxes'])
    df['id']=df.index
    df['file_name']=df.File_name.apply(_conv)
    intermed = [v for k, v in df.to_dict(orient='index').items()]
    annotations=[]
    img_infos=[]
    for x in intermed:
        box=[ float(t) for t in x['Bounding_boxes'].split(", ")]
        img_info={
            'file_name':x['File_name'],
            'height':512,
            'width':512,
            'id':x['id']
        }
        ann_info={
            'image_id':x['id'],
            'id':x['id'],
            'iscrowd':0,
            'category_id':1,
            'area':1,
            'bbox':box
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


def _conv(x):
    parts = x.split("_")
    pre = "_".join(parts[:-1])
    return osp.join(pre, parts[-1])
