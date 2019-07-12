import cv2
import file_locs
import numpy as np
from mmdet.datasets import DL_coco
from mmcv.parallel.data_container import DataContainer
from PIL import Image, ImageStat, ImageDraw
import torch
from mmcv import Config
from mmdet.models import build_detector
from mmcv.parallel import MMDataParallel

def view_dataset(gen,max_imgs=30):
    max_imgs=min(max_imgs,len(gen))
    for i in range(max_imgs):
        data=gen[i]
        img=un(data['img'])
        img=tonumpy(img)
        img=(img*255).astype(np.uint8)

        if len(img.shape)==3:
            img=img[0]
        assert len(img.shape)==2
        img=np.tile(img,(3,1,1))
        img=np.transpose(img,[1,2,0])


        img=Image.fromarray(img).convert('RGB')

        draw=ImageDraw.Draw(img)
        pts=[int(x) for x in un(data['gt_bboxes'])[0]]

        #import pdb; pdb.set_trace()
        draw.rectangle(((pts[0],pts[1]),(pts[2],pts[3])))

        if 'gt_masks' in data:
            mask=un(data['gt_masks']).reshape(512,512)
            mask=mask*64
            #mask=np.zeros((512,512),np.uint8)
            #import pdb; pdb.set_trace()
            #mask[:,:]=1
            mask=Image.fromarray(mask)
            overlay=Image.new(mode='RGB',size=img.size,color='RED')
            img.paste(overlay,(0,0),mask=mask)
            #draw.bitmap((0,0),mask,fill='WHITE')
            #img.paste(overlay,mask=mask)
            #draw.bitmap((0,0),mask,fill=(255, 255, 255, 255))
            #mask=np.tile(mask,(3,1,1))
            #mask=np.transpose(mask,[1,2,0])
            #img=cv2.addWeighted(img,0.4,mask,128,0)
        img.save("markings/"+str(i)+".png")

def polypoints(arr):
    pts=[[arr[i],arr[i]+1] for i in range(0,len(arr),2)]
    pts=np.array(pts,np.int32)
    pts=pts.reshape(-1,1,2)
    return pts

def un(val):
    if isinstance(val,DataContainer):
        return val._data
    else:
        return val

def tonumpy(val):
    if isinstance(val,torch.Tensor):
        return val.data.cpu().numpy()
    else:
        assert isinstance(val,np.ndarray)
        return val

if __name__ == "__main__":
    gen=DL_coco(file_locs.csv_dir+"DL_test_toy.csv",file_locs.image_dir,
                with_mask=True,use_context=True)


    import configs.dan.retina_dl as cfg
    model = build_detector(cfg.model, train_cfg = cfg.train_cfg, test_cfg = cfg.test_cfg)
    dmodel=MMDataParallel(model)

    #view_dataset(gen)
    data=gen[0]
    model.simple_test(img=data['img'].data.reshape(1, 3, 512, 512).cuda(), img_meta=data['img_meta'].data)
