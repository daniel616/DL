from .coco import CocoDataset
from .dl_to_coco import to_coco

import numpy as np
import torch


class DL_coco(CocoDataset):
    CLASSES = ('abnormal')

    def __init__(self,
                 ann_file,
                 img_prefix,
                 img_scale=(512, 512),
                 img_norm_cfg=dict(
                     mean=[0, 0, 0],
                     std=[1, 1, 1],
                     to_rgb=True),
                 with_mask=False,
                 with_crowd=False,
                 **kwargs):
        ftype = ann_file.split(".")[-1]
        assert ftype == 'csv' or ftype == 'json'

        if ftype == 'json':
            super(DL_coco, self).__init__(ann_file,img_prefix,img_scale,img_norm_cfg,
                                          with_mask=with_mask,with_crowd=with_crowd,
                                          **kwargs)
        else:
            new_name = to_json(ann_file)
            to_coco(ann_file, new_name)
            super(DL_coco, self).__init__(new_name,img_prefix,img_scale,img_norm_cfg,
                                          with_mask=with_mask,with_crowd=with_crowd,
                                          **kwargs)

    

    def __getitem__(self, idx):
        data=super(DL_coco,self).__getitem__(idx)
        image_dc=data['img']
        image_dc._data=DICOM_window(image_dc._data-32768)
        return data


def DICOM_window(x,min_w=-1024,max_w=1023):
    assert isinstance(x,np.ndarray) or isinstance(x,torch.Tensor)
    if isinstance(x,np.ndarray):
        x=x.astype('float32')
        x=np.clip(x,a_min=min_w,a_max=max_w)
        x=(x-min_w)/(max_w-min_w)
        return x
    else:
        x=x.type(torch.float32)
        x=x.clamp(min_w,max_w)
        x=(x-min_w)/(max_w-min_w)
        return x


def to_json(st):
    pre = st.split(".")[:-1]
    pre="".join(pre)
    return ".".join([pre, "json"])
