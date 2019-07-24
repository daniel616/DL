import os.path as osp


import pycocotools.mask as maskUtils
import pandas as pd
import json
import numpy as np

from skimage import io

import cv2
import sys
import datetime

sys.path.insert(0,"../..")
from file_locs import image_dir

def to_coco(csv,out_file,use_grabcut=False):
    df= pd.read_csv(csv)
    df=_filter(df)
    df['id']=df.index
    df['file_name']=df.File_name.apply(_conv)
    intermed = [v for k, v in df.to_dict(orient='index').items()]
    annotations=[]
    img_infos=[]

    for idx, x in enumerate(intermed):
        if use_grabcut and idx%100==0:
            print("Grabcutting...", idx)
            print(datetime.datetime.now())
        s_range= [int(t) for t in x['Slice_range'].split(", ")]
        key=x['Key_slice_index']
        adjacents=[max(key-1,s_range[0]),min(key+1,s_range[1])]
        adjacents=[pad(t) for t in adjacents]

        directory=x['file_name'].split("/")[0]
        ctx1=osp.join(directory,adjacents[0])
        ctx2=osp.join(directory,adjacents[1])

        img_info={
            'file_name':x['file_name'],
            'ctx1':ctx1,
            'ctx2':ctx2,
            'height':512,
            'width':512,
            'id':x['id']
        }

        b=[ int(float(t)) for t in x['Bounding_boxes'].split(", ")]
        box_coco=[b[0],b[1],b[2]-b[0],b[3]-b[1]]

        ann_info={
            'image_id':x['id'],
            'id':x['id'],
            'iscrowd':0,
            'category_id':1,
            'bbox':box_coco
        }
        spoints=get_seg(x['Measurement_coordinates'])

        if not use_grabcut:
            rle=maskUtils.frPyObjects([spoints],512,512)
            #rle = maskUtils.merge(rle)
            area=maskUtils.area(rle)[0]
            ann_info['area']=float(area)
            ann_info['segmentation']=[spoints]
        else:
            f_name=img_info['file_name']
            w = [float(t) for t in x['DICOM_windows'].split(", ")]
            img=get_img(f_name,w)

            pts=np.array([[[spoints[0],spoints[1]],
                            [spoints[2],spoints[3]],
                            [spoints[4],spoints[5]],
                            [spoints[6],spoints[7]]]],dtype=np.int32)

            big_box=[b[0]-5,b[1]-5,b[2]+5,b[3]+5]

            try:
                gpoints=grabseg(img,big_box,pts)
            except cv2.error:
                print("confused by:",f_name)
                gpoints=[spoints]
            if len(gpoints)==0:
                gpoints=[spoints]

            rle=maskUtils.frPyObjects(gpoints,512,512)
            area=maskUtils.area(rle)[0]
            ann_info['area']=float(area)
            ann_info['segmentation']=gpoints

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


def _filter(df):
    df=df[df['Possibly_noisy']==0]
    mask = df.apply(lambda x: x['Image_size'].split(", ")[0] == '512',axis=1)
    df= df[mask]
    return df




def pad(v):
    assert isinstance(v,int)
    if v<10:
        return "00"+str(v)+".png"
    if v<100:
        return "0"+str(v)+".png"
    return str(v)+".png"

def get_seg(st):
    v = [float(x) for x in st.split(", ")]

    return [v[0],v[1],v[4],v[5],v[2],v[3],v[6],v[7]]

def _conv(x):
    parts = x.split("_")
    pre = "_".join(parts[:-1])
    return osp.join(pre, parts[-1])


def grabseg(img,bbox,pts):
    img=np.asarray(img)
    mask = np.zeros(img.shape, dtype=np.uint8)#(zero is background)
    assert isinstance(pts,np.ndarray)
    cv2.rectangle(mask,(bbox[0],bbox[1]),(bbox[2],bbox[3]),cv2.GC_PR_BGD,
                  thickness=-1)
    cv2.fillPoly(mask,pts,cv2.GC_FGD)
    assert len(mask.shape)==2

    bgdModel = np.zeros((1, 65), np.float64)
    fgdModel = np.zeros((1, 65), np.float64)


    img=np.expand_dims(img,-1)

    img=np.tile(img,(1,1,3))

    cv2.grabCut(img, mask, None, bgdModel, fgdModel, 5, cv2.GC_INIT_WITH_MASK)
    mask2 = np.where((mask == 2) | (mask == 0), 0, 1).astype('uint8')
    mask2=np.asfortranarray(mask2)


    points, _ = cv2.findContours((mask2).astype(np.uint8), cv2.RETR_EXTERNAL,
                                                     cv2.CHAIN_APPROX_SIMPLE)

    ret=[]
    for i in range(len(points)):
        if len(points[i])>3:
            ret.append(points[i].reshape(-1).astype(np.int32).tolist())

    return ret



def DICOM_window(x,min_w=-275.,max_w=175.0):
    x=np.clip(x,a_min=min_w,a_max=max_w)
    x=(x-min_w)/(max_w-min_w)
    return x

def get_img(f_name,window):
    assert window[1]>window[0]
    img=io.imread(osp.join(image_dir,f_name)).astype(np.int32)
    img-=32768
    img=img.astype(np.float32)
    img=DICOM_window(img,max_w=window[1],min_w=window[0])
    img=(255*img).astype(np.uint8)
    return img

