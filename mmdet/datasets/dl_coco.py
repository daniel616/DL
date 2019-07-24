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

@DATASETS.register_module
class DL_coco(CocoDataset):
    CLASSES = ('abnormal')
    #CLASSES=['bleh' for x in range(81)]
    #CLASSES[0]='abnormal'

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
        assert ftype == 'csv' or ftype == 'json'
        self.use_context=use_context

        if ftype == 'json':
            super(DL_coco, self).__init__(ann_file,img_prefix,img_scale,img_norm_cfg,
                                          with_mask=with_mask,with_crowd=with_crowd,
                                          **kwargs)
        else:
            new_name = to_json(ann_file)
            to_coco(ann_file, new_name,use_grabcut=grabcut)
            super(DL_coco, self).__init__(new_name,img_prefix,img_scale,img_norm_cfg,
                                          with_mask=with_mask,with_crowd=with_crowd,
                                          **kwargs)


    '''
    def __getitem__(self, idx):
        data=super(DL_coco,self).__getitem__(idx)
        image_dc=data['img']

        if isinstance(image_dc,DC):
            image_dc._data=DICOM_window(image_dc._data-32768)
            return data
        elif isinstance(image_dc,list):
            for idx, im in enumerate(image_dc):
                image_dc[idx]=DICOM_window(im-32768)
            return data
        raise ValueError('Type must be DC(sigh) or list')
    '''

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
        # load proposals if necessary
        if self.proposals is not None:
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

        ann = self.get_ann_info(idx)
        gt_bboxes = ann['bboxes']
        gt_labels = ann['labels']
        if self.with_crowd:
            gt_bboxes_ignore = ann['bboxes_ignore']

        # skip the image if there is no valid gt bbox
        if len(gt_bboxes) == 0:
            return None

        # extra augmentation
        if self.extra_aug is not None:
            img, gt_bboxes, gt_labels = self.extra_aug(img, gt_bboxes,
                                                       gt_labels)

        # apply transforms
        flip = True if np.random.rand() < self.flip_ratio else False
        # randomly sample a scale
        img_scale = random_scale(self.img_scales, self.multiscale_mode)

        img, img_shape, pad_shape, scale_factor = self.img_transform(
            img, img_scale, flip, keep_ratio=self.resize_keep_ratio)
        img = img.copy()
        if self.with_seg:
            gt_seg = mmcv.imread(
                osp.join(self.seg_prefix, img_info['file_name'].replace(
                    'jpg', 'png')),
                flag='unchanged')
            gt_seg = self.seg_transform(gt_seg.squeeze(), img_scale, flip)
            gt_seg = mmcv.imrescale(
                gt_seg, self.seg_scale_factor, interpolation='nearest')
            gt_seg = gt_seg[None, ...]
        if self.proposals is not None:
            proposals = self.bbox_transform(proposals, img_shape, scale_factor,
                                            flip)
            proposals = np.hstack(
                [proposals, scores]) if scores is not None else proposals
        gt_bboxes = self.bbox_transform(gt_bboxes, img_shape, scale_factor,
                                        flip)
        if self.with_crowd:
            gt_bboxes_ignore = self.bbox_transform(gt_bboxes_ignore, img_shape,
                                                   scale_factor, flip)
        if self.with_mask:
            gt_masks = self.mask_transform(ann['masks'], pad_shape,
                                           scale_factor, flip)

        ori_shape = (img_info['height'], img_info['width'], 3)
        img_meta = dict(
            ori_shape=ori_shape,
            img_shape=img_shape,
            pad_shape=pad_shape,
            scale_factor=scale_factor,
            flip=flip,
            #gt_bboxes=gt_bboxes
            )

        data = dict(
            img=DC(to_tensor(img), stack=True),
            img_meta=DC(img_meta, cpu_only=True),
            gt_bboxes=DC(to_tensor(gt_bboxes)))
        if self.proposals is not None:
            data['proposals'] = DC(to_tensor(proposals))
        if self.with_label:
            data['gt_labels'] = DC(to_tensor(gt_labels))
        if self.with_crowd:
            data['gt_bboxes_ignore'] = DC(to_tensor(gt_bboxes_ignore))
        if self.with_mask:
            data['gt_masks'] = DC(gt_masks, cpu_only=True)
        if self.with_seg:
            data['gt_semantic_seg'] = DC(to_tensor(gt_seg), stack=True)


        #data['file_name']=self.img_infos[idx]['file_name']

        return data


    def prepare_test_img(self, idx):
        """Prepare an image for testing (multi-scale and flipping)"""
        img=self.get_img(idx)
        img_info=self.img_infos[idx]

        if self.proposals is not None:
            proposal = self.proposals[idx][:self.num_max_proposals]
            if not (proposal.shape[1] == 4 or proposal.shape[1] == 5):
                raise AssertionError(
                    'proposals should have shapes (n, 4) or (n, 5), '
                    'but found {}'.format(proposal.shape))
        else:
            proposal = None



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
            if proposal is not None:
                if proposal.shape[1] == 5:
                    score = proposal[:, 4, None]
                    proposal = proposal[:, :4]
                else:
                    score = None
                _proposal = self.bbox_transform(proposal, img_shape,
                                                scale_factor, flip)
                _proposal = np.hstack(
                    [_proposal, score]) if score is not None else _proposal
                _proposal = to_tensor(_proposal)
            else:
                _proposal = None
            return _img, _img_meta, _proposal

        imgs = []
        img_metas = []
        proposals = []

        for scale in self.img_scales:
            _img, _img_meta, _proposal = prepare_single(
                img, scale, False, proposal)
            #TODO: THIS IS NOT ROBUST TO AUGMENTATIONS
            imgs.append(_img)
            img_metas.append(DC(_img_meta, cpu_only=True))
            proposals.append(_proposal)

        ann=self.get_ann_info(idx)
        gt_bboxes=ann['bboxes']
        data = dict(img=imgs, img_meta=img_metas,gt_bboxes=DC(to_tensor(gt_bboxes)))
        if self.proposals is not None:
            data['proposals'] = proposals
        if self.with_mask:
            gt_masks=ann['masks']
            data['gt_masks'] = DC(gt_masks, cpu_only=True)

        #data['file_name']=self.img_infos[idx]['file_name']
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


def _conv(x):
    parts = x.split("_")
    pre = "_".join(parts[:-1])
    return osp.join(pre, parts[-1])



def DICOM_window(x,min_w=-275,max_w=175):
    x=np.clip(x,a_min=min_w,a_max=max_w)
    x=(x-min_w)/(max_w-min_w)
    return x


'''

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

'''

def to_json(st):
    pre = st.split(".")[:-1]
    pre="".join(pre)
    return ".".join([pre, "json"])
