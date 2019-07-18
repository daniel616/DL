import logging
from abc import ABCMeta, abstractmethod

import mmcv
import numpy as np
import torch
import torch.nn as nn
import pycocotools.mask as maskUtils
import cv2

from mmcv.image import imread, imwrite
from mmcv.visualization.color import color_val

from mmdet.core import tensor2imgs, get_classes, auto_fp16


import sys
sys.path.insert(0,"../../..")

class BaseDetector(nn.Module):
    """Base class for detectors"""

    __metaclass__ = ABCMeta

    def __init__(self):
        super(BaseDetector, self).__init__()
        self.fp16_enabled = False

    @property
    def with_neck(self):
        return hasattr(self, 'neck') and self.neck is not None

    @property
    def with_shared_head(self):
        return hasattr(self, 'shared_head') and self.shared_head is not None

    @property
    def with_bbox(self):
        return hasattr(self, 'bbox_head') and self.bbox_head is not None

    @property
    def with_mask(self):
        return hasattr(self, 'mask_head') and self.mask_head is not None

    @abstractmethod
    def extract_feat(self, imgs):
        pass

    def extract_feats(self, imgs):
        assert isinstance(imgs, list)
        for img in imgs:
            yield self.extract_feat(img)

    @abstractmethod
    def forward_train(self, imgs, img_metas, **kwargs):
        pass

    @abstractmethod
    def simple_test(self, img, img_meta, **kwargs):
        pass

    @abstractmethod
    def aug_test(self, imgs, img_metas, **kwargs):
        pass

    def init_weights(self, pretrained=None):
        if pretrained is not None:
            logger = logging.getLogger()
            logger.info('load model from: {}'.format(pretrained))

    def forward_test(self, imgs, img_metas, **kwargs):
        for var, name in [(imgs, 'imgs'), (img_metas, 'img_metas')]:
            if not isinstance(var, list):
                raise TypeError('{} must be a list, but got {}'.format(
                    name, type(var)))

        num_augs = len(imgs)
        if num_augs != len(img_metas):
            raise ValueError(
                'num of augmentations ({}) != num of image meta ({})'.format(
                    len(imgs), len(img_metas)))
        # TODO: remove the restriction of imgs_per_gpu == 1 when prepared
        imgs_per_gpu = imgs[0].size(0)
        assert imgs_per_gpu == 1

        if num_augs == 1:
            return self.simple_test(imgs[0], img_metas[0], **kwargs)
        else:
            return self.aug_test(imgs, img_metas, **kwargs)

    @auto_fp16(apply_to=('img',))
    def forward(self, img, img_meta, return_loss=True, **kwargs):
        if return_loss:
            return self.forward_train(img, img_meta, **kwargs)
        else:
            return self.forward_test(img, img_meta, **kwargs)

    def show_result(self,
                    data,
                    result,
                    out_file,
                    img_norm_cfg=None,
                    dataset=None,
                    score_thr=0):
        # asdf
        import pdb; pdb.set_trace()
        if isinstance(result, tuple):
            bbox_result, segm_result = result
        else:
            bbox_result, segm_result = result, None
        img_tensor = data['img'][0]
        img_tensor *= 255
        img_metas = data['img_meta'][0].data[0]
        if img_norm_cfg is None:
            num_imgs = img_tensor.size(0)
            imgs = []
            for img_id in range(num_imgs):
                img = img_tensor[img_id, ...].cpu().numpy().transpose(1, 2, 0)
                imgs.append(np.ascontiguousarray(img))
            imgs = imgs
        else:
            imgs = tensor2imgs(img_tensor, **img_norm_cfg)
        assert len(imgs) == len(img_metas)

        if dataset is None:
            class_names = self.CLASSES
        elif isinstance(dataset, str):
            class_names = get_classes(dataset)
        elif isinstance(dataset, (list, tuple)):
            class_names = dataset
        else:
            raise TypeError(
                'dataset must be a valid dataset name or a sequence'
                ' of class names, not {}'.format(type(dataset)))

        for idx, val in enumerate(zip(imgs, img_metas)):
            img = val[0]
            img_meta = val[1]
            img_shape = img_meta['img_shape']
            assert len(img_shape) == 2 or len(img_shape) == 3
            if len(img_shape) == 2:
                h, w, = img_shape
            else:
                h, w, _ = img_shape
            img_show = img[:h, :w, :]
            #img_show = scale_255(img_show)

            bboxes = np.vstack(bbox_result)
            # draw segmentation masks
            if segm_result is not None:
                segms = mmcv.concat_list(segm_result)
                inds = np.where(bboxes[:, -1] > score_thr)[0]
                for i in inds:
                    color_mask = np.random.randint(
                        0, 256, (1, 3), dtype=np.uint8)
                    mask = maskUtils.decode(segms[i]).astype(np.bool)
                    img_show[mask] = img_show[mask] * 0.5 + color_mask * 0.5
            # draw bounding boxes
            labels = [
                np.full(bbox.shape[0], i, dtype=np.int32)
                for i, bbox in enumerate(bbox_result)
            ]
            labels = np.concatenate(labels)

            show_det_bboxes(
                img_show,
                bboxes,
                labels,
                out_file,
                gt_bboxes=data['gt_bboxes'],
                class_names=class_names,
                score_thr=score_thr
            )
            '''
            mmcv.imshow_det_bboxes(
                img_show,
                bboxes,
                labels,
                show=False,
                out_file=out_dir+str(idx)+".png",
                class_names=class_names,
                score_thr=score_thr)
            '''


def show_det_bboxes(img,
                    bboxes,
                    labels,
                    out_file,
                    gt_bboxes=None,
                    class_names=None,
                    score_thr=0,
                    bbox_color='yellow',
                    gt_color='green',
                    top_k=20
                    ):

    assert bboxes.ndim == 2
    assert labels.ndim == 1
    assert bboxes.shape[0] == labels.shape[0]
    assert bboxes.shape[1] == 4 or bboxes.shape[1] == 5
    img = imread(img)
    if score_thr > 0:
        assert bboxes.shape[1] == 5
        scores = bboxes[:, -1]
        inds = scores > score_thr
        bboxes = bboxes[inds, :]
        labels = labels[inds]

    topinds=np.argsort(bboxes[:,4])
    topinds=topinds[:top_k] if len(topinds)>=top_k else topinds
    bboxes=bboxes[topinds,:]
    labels=labels[topinds]


    bbox_color = color_val(bbox_color)
    gt_color=color_val(gt_color)

    for bbox, label in zip(bboxes, labels):
        addBox(img,bbox,bbox_color,label,class_names=class_names)

    if gt_bboxes is not None:
        assert len(gt_bboxes)==1
        gt_bboxes=gt_bboxes[0]
        for gt_box in gt_bboxes:
            gt_box=gt_box.numpy()
            addBox(img,gt_box,gt_color,0,class_names=class_names)

    imwrite(img, out_file)

def addBox(img,bbox,color,label,class_names=None):
    bbox=bbox
    bbox_int = bbox.astype(np.int32)
    left_top = (bbox_int[0], bbox_int[1])
    right_bottom = (bbox_int[2], bbox_int[3])
    cv2.rectangle(
        img, left_top, right_bottom, color, 1)
    label_text = class_names[
        label] if class_names is not None else 'cls {}'.format(label)
    if len(bbox) > 4:
        label_text += '|{:.02f}'.format(bbox[-1])
    cv2.putText(img, label_text, (bbox_int[0], bbox_int[1] - 2),
                cv2.FONT_HERSHEY_COMPLEX, 0.5, color)



def to_numpy(tensor):
    return tensor.cpu().detach().numpy()
