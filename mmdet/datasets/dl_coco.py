from .coco import CocoDataset
from .dl_to_coco import to_coco

import numpy as np
import torch

from mmcv.parallel import DataContainer as DC

from .utils import to_tensor, random_scale
import os.path as osp
from skimage import io

from file_locs import image_dir

from .registry import DATASETS

import torchvision.transforms.functional as TF

@DATASETS.register_module
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
                 use_context=False,
                 grabcut=False,
                 **kwargs):
        ftype = ann_file.split(".")[-1]
        assert ftype == 'json'
        self.use_context=use_context

        super(DL_coco, self).__init__(ann_file,img_prefix,img_scale,img_norm_cfg,
                                      with_mask=with_mask,with_crowd=with_crowd,
                                      **kwargs)
        assert self.proposals is None, "Not implmented"
        assert with_crowd== False, "Not implmented"
        assert self.with_seg== False, "Not implemented"

    def __getitem__(self, idx):
        if self.test_mode:
            return self.prepare_test_img(idx)
        while True:
            data = self.prepare_train_img(idx)
            if data is None:
                idx = self._rand_another(idx)
                continue
            return data

    def prepare_train_img(self, idx):
        img=self.get_img(idx)
        img_info=self.img_infos[idx]

        ann = self.get_ann_info(idx)
        gt_bboxes = ann['bboxes']
        gt_labels = ann['labels']

        # skip the image if there is no valid gt bbox
        if len(gt_bboxes) == 0:
            return None
        # extra augmentation
        if self.extra_aug is not None:
            img, gt_bboxes, gt_labels = self.extra_aug(img, gt_bboxes,
                                                       gt_labels)
        #TODO: check flipping is done correctly
        flip=False
        # randomly sample a scale
        img_scale = random_scale(self.img_scales, self.multiscale_mode)

        img, img_shape, pad_shape, scale_factor = self.img_transform(
            img, img_scale, flip, keep_ratio=self.resize_keep_ratio)
        img = img.copy()

        gt_bboxes = self.bbox_transform(gt_bboxes, img_shape, scale_factor,
                                        flip)

        ori_shape = (img_info['height'], img_info['width'], 3)
        img_meta = dict(
            ori_shape=ori_shape,
            img_shape=img_shape,
            pad_shape=pad_shape,
            scale_factor=scale_factor,
            flip=flip,
            )

        data = dict(
            img=DC(to_tensor(img), stack=True),
            img_meta=DC(img_meta, cpu_only=True),
            gt_bboxes=DC(to_tensor(gt_bboxes)))
        if self.with_label:
            data['gt_labels'] = DC(to_tensor(gt_labels))

        if self.with_mask:
            gt_masks = self.mask_transform(ann['masks'], pad_shape, scale_factor, False)
            data['gt_masks'] = DC(gt_masks, cpu_only=True)

        #img,box,gt_bboxes,gt_masks=d_transform(img,gt_bboxes,gt_masks)
        return data


    def prepare_test_img(self, idx):
        """Prepare an image for testing (multi-scale and flipping)"""
        img=self.get_img(idx)
        img_info=self.img_infos[idx]

        assert self.proposals is None
        proposal=None

        def prepare_single(img, scale, flip, proposal=None):
            _img, img_shape, pad_shape, scale_factor = self.img_transform(
                img, scale, flip, keep_ratio=self.resize_keep_ratio)
            _img = to_tensor(_img)
            _img_meta = dict(
                ori_shape=(img_info['height'], img_info['width'], 3),
                img_shape=img_shape,
                pad_shape=pad_shape,
                scale_factor=scale_factor,
                flip=flip)

            return _img, _img_meta

        imgs = []
        img_metas = []

        for scale in self.img_scales:
            _img, _img_meta = prepare_single(
                img, scale, False, proposal)
            #TODO: THIS IS NOT ROBUST TO AUGMENTATIONS
            imgs.append(_img)
            img_metas.append(DC(_img_meta, cpu_only=True))

        ann=self.get_ann_info(idx)
        gt_bboxes=ann['bboxes']
        data = dict(img=imgs, img_meta=img_metas,gt_bboxes=DC(to_tensor(gt_bboxes)))
        if self.with_mask:
            gt_masks=ann['masks']
            data['gt_masks'] = DC(gt_masks, cpu_only=True)

        data['file_name'] = DC(self.img_infos[idx]['file_name'])

        return data

    def get_img(self,idx):
        img_info = self.img_infos[idx]
        readim=lambda x:io.imread(osp.join(image_dir,x)).astype('int32')-32768
        img=readim(img_info['file_name'])
        if self.use_context:
            ctx1,ctx2=readim(img_info['ctx1']),readim(img_info['ctx2'])
            img=np.stack([img,ctx1,ctx2],axis=-1)
        img=DICOM_window(img)
        return img


    def _proposals_scores(self,idx):
        proposals = self.proposals[idx][:self.num_max_proposals]
        # TODO: Handle empty proposals properly. Currently images with
        # no proposals are just ignored, but they can be used for
        # training in concept.
        if len(proposals) == 0:
            return None
        if not (proposals.shape[1] == 4 or proposals.shape[1] == 5):
            raise AssertionError(
                'proposals should have shapes (n, 4) or (n, 5), '
                'but found {}'.format(proposals.shape))
        if proposals.shape[1] == 5:
            scores = proposals[:, 4, None]
            proposals = proposals[:, :4]
        else:
            scores = None

        return proposals,scores


#input: PIL images for img, gt_boxes,gt_masks
def pil_rotate(img,gt_bbox,gt_masks,angle=None):
    angle=5
    r_img= TF.rotate(img,angle)
    r_masks=TF.rotate(gt_masks,angle)
    r_box=TF.rotate(gt_bbox)
    r_box=np.array(r_box)
    pts=np.where(r_box==1)


    return r_img,r_masks

def d_transform(img,gt_bboxes,gt_masks):
    img=TF.to_pil_image(img)
    #gt_bboxes=TF.to_pil_image()
    pil_rotate(img,gt_bboxes,gt_masks)


def _conv(x):
    parts = x.split("_")
    pre = "_".join(parts[:-1])
    return osp.join(pre, parts[-1])



def DICOM_window(x,min_w=-275,max_w=175):
    x=np.clip(x,a_min=min_w,a_max=max_w)
    x=(x-min_w)/(max_w-min_w)
    return x

def pts_to_img(segpoints,img_shape=(512,512)):
    image=np.zeros(img_shape,dtype=np.uint8)
    image[segpoints]=1
    return image

def img_to_pts(img):
    pts=np.where(img==1)
    return pts


def to_json(st):
    pre = st.split(".")[:-1]
    pre="".join(pre)
    return ".".join([pre, "json"])
