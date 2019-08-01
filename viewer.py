import file_locs
import numpy as np
from mmdet.datasets import DL_coco
from mmcv.parallel.data_container import DataContainer
from PIL import Image, ImageStat, ImageDraw
import torch
from mmdet.models import build_detector
from mmcv.parallel import MMDataParallel
import matplotlib.pyplot as plt

def view_dataset_test(gen,max_imgs=150):
    max_imgs=min(max_imgs,len(gen))
    for i in range(max_imgs):
        data=gen[i]
        img=data['img'][0]
        img=tonumpy(img)

        gt_bboxes=data['gt_bboxes'].data

        gt_masks=None
        if 'gt_masks' in data:
            gt_masks=data['gt_masks'].data
        view_image(img,gt_bboxes,
                   "markings/"+str(i)+".png",
                   text=data['file_name'].data,
                   gt_masks=gt_masks)

def view_dataset_train(gen,max_imgs=150):
    max_imgs=min(max_imgs,len(gen))
    for i in range(max_imgs):

        data=gen[i]
        img=data['img'].data
        img=tonumpy(img)

        gt_bboxes=data['gt_bboxes'].data
        gt_bboxes=tonumpy(gt_bboxes)

        gt_masks=None
        if 'gt_masks' in data:
            gt_masks=data['gt_masks'].data
            gt_masks=tonumpy(gt_masks)
        view_image(img,gt_bboxes,
                   "markings/"+str(i)+".png",
                   text=None,
                   gt_masks=gt_masks)

def view_image(img,gt_bboxes,out_file, dt_bboxes=None,gt_masks=None,text=None):
    if img.max()<=1:
        img=(img*255)
    img=img.astype(np.uint8)
    assert len(img.shape)==3

    slice=img[0]
    slice=np.tile(slice,(3,1,1))
    slice=np.transpose(slice,[1,2,0])

    fig= plt.figure(figsize=(9,4.5))
    fig.add_subplot(1,2,1)
    plt.imshow(slice)
    plt.axis('off')

    slice=Image.fromarray(slice).convert('RGB')
    draw=ImageDraw.Draw(slice)
    gt_bboxes=[[int(t) for t in x] for x in gt_bboxes]
    for box in gt_bboxes:
        draw.rectangle(((box[0],box[1]),(box[2],box[3])), outline='green')

    if dt_bboxes is not None:
        dt_bboxes=[[int(t) for t in x] for x in dt_bboxes]
        for box in dt_bboxes:
            draw.rectangle(((box[0],box[1]),(box[2],box[3])),
                           outline='blue')

    if text is not None:
        draw.text((250,10),text)

    if gt_masks is not None:
        overlay=Image.new(mode='RGB',size=(512,512),color='RED')
        for mask in gt_masks:
            assert mask.max()<=1
            pil_mask = Image.fromarray(np.uint8(255 * mask))
            slice.paste(overlay,(0,0),mask=pil_mask)

    fig.add_subplot(1,2,2)
    plt.imshow(np.array(slice))
    plt.axis('off')
    plt.subplots_adjust(wspace=None, hspace=None)
    fig.subplots_adjust(left=0, right=1, bottom=0, top=1)

    mydpi=200
    plt.savefig(out_file,dpi=mydpi,pad_inches=0)
    plt.close()

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
    train=DL_coco(file_locs.csv_dir+"DL_train_toy.json",file_locs.image_dir,
                with_mask=True,use_context=True,test_mode=False,will_batch=False)
    test=DL_coco(file_locs.csv_dir+"DL_train_toy.json",file_locs.image_dir,
                with_mask=True,use_context=True,test_mode=True,will_batch=False)

    import pandas as pd
    data=test[0]
    view_dataset_train(train)
    #df= pd.read_csv(file_locs.csv_dir+"DL_test.csv")
    #data=test[0]

    #data1=train[0]

