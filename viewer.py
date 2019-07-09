import cv2
import file_locs
import numpy as np
from mmdet.datasets import DL_coco
from mmcv.parallel.data_container import DataContainer
from PIL import Image, ImageStat, ImageDraw
import torch
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
            import pdb; pdb.set_trace()
            mask=un(data['gt_masks']).reshape(512,512)
            mask=np.zeros((512,512),np.uint8)
            #import pdb; pdb.set_trace()
            mask[:,:]=1
            mask=Image.fromarray(mask,mode='1')
            overlay=Image.new('RGB',img.size,color='WHITE')
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
    gen=DL_coco(file_locs.csv_dir+"DL_test_toy.csv",file_locs.image_dir,with_mask=True)
    view_dataset(gen)