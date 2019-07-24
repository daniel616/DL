import file_locs

from mmdet.datasets.dl_coco import DL_coco
from mmdet.datasets.coco import CocoDataset
from mmdet.models import build_detector


import configs.dan.retina_dl as cfg
import configs.dan.mask as coco_cfg
if __name__ == "__main__":
    gen=DL_coco(file_locs.csv_dir+"DL_test_toy.csv",file_locs.image_dir,with_mask=True)
    args=coco_cfg.data['train']
    '''coco_gen=CocoDataset(ann_file=args['ann_file'],
                         img_prefix=args['img_prefix'],
                         img_scale=args['img_scale'],
                         img_norm_cfg=args['img_norm_cfg']
                         )'''
    print(gen[0])
    model = build_detector(
        coco_cfg.model, train_cfg=coco_cfg.train_cfg, test_cfg=coco_cfg.test_cfg)
    import ipdb; ipdb.set_trace()
