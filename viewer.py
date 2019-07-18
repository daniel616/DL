import file_locs
import numpy as np
from mmdet.datasets import DL_coco
from mmcv.parallel.data_container import DataContainer
from PIL import Image, ImageStat, ImageDraw
import torch
from mmdet.models import build_detector
from mmcv.parallel import MMDataParallel

def view_dataset(gen,max_imgs=150):
    max_imgs=min(max_imgs,len(gen))
    for i in range(max_imgs):
        data=gen[i]
        img=data['img'][0]
        img=tonumpy(img)

        gt_bboxes=data['gt_bboxes']

        view_image(img,gt_bboxes,"markings/"+str(i)+".png",text=data['file_name'])

def view_image(img,gt_bboxes,out_file, dt_bboxes=None,gt_mask=None,text=None):
    if img.max()<=1:
        img=(img*255)
    img=img.astype(np.uint8)
    assert len(img.shape)==3

    for idx in range(img.shape[0]):
        if idx>0: continue
        slice=img[idx]
        slice=np.tile(slice,(3,1,1))
        slice=np.transpose(slice,[1,2,0])
        slice=Image.fromarray(slice).convert('RGB')
        draw=ImageDraw.Draw(slice)

        gt_bboxes=[[int(t) for t in x] for x in gt_bboxes]
        for box in gt_bboxes:
            draw.rectangle(((box[0],box[1]),(box[2],box[3])), outline='green')

        if dt_bboxes is not None:
            dt_bboxes=[[int(t) for t in x] for x in dt_bboxes]
            for box in dt_bboxes:
                draw.rectangle(((box[0],box[1]),(box[2],box[3])),
                               outline='yellow')

        if text is not None:
            draw.text((250,10),text)

        if gt_mask is not None:
            assert False,"not implemented yet"
            mask=mask*64
            mask=Image.fromarray(mask)
            overlay=Image.new(mode='RGB',size=img.size,color='RED')
            slice.paste(overlay,(0,0),mask=mask)
        slice.save(out_file)

def un(val):
    assert isinstance(val,DataContainer), "not DataContainer"
    return val.data

def tonumpy(val):
    if isinstance(val,torch.Tensor):
        return val.data.cpu().numpy()
    else:
        assert isinstance(val,np.ndarray)
        return val

if __name__ == "__main__":
    train=DL_coco(file_locs.csv_dir+"DL_train_toy.csv",file_locs.image_dir,
                with_mask=True,use_context=True,test_mode=False)
    test=DL_coco(file_locs.csv_dir+"DL_test.csv",file_locs.image_dir,
                with_mask=True,use_context=True,test_mode=True)

    import pandas as pd
    df= pd.read_csv(file_locs.csv_dir+"DL_test.csv")

    import configs.dan.retina_dl as cfg
    model = build_detector(cfg.model, train_cfg = cfg.train_cfg, test_cfg = cfg.test_cfg)
    dmodel=MMDataParallel(model)
    data1=test[0]
    view_dataset(test)

    data=test[0]

    #model.simple_test(img=data['img'].data.reshape(1, 3, 512, 512).cuda(), img_meta=data['img_meta'].data)
