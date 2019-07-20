import os.path as osp



import pycocotools.mask as maskUtils
import pandas as pd
import json
import numpy as np

import cv2
def to_coco(csv,out_file):
    df= pd.read_csv(csv)
    df=df[df['Possibly_noisy']==0]
    df['id']=df.index
    df['file_name']=df.File_name.apply(_conv)
    intermed = [v for k, v in df.to_dict(orient='index').items()]
    annotations=[]
    img_infos=[]
    for x in intermed:
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

        b=[ float(t) for t in x['Bounding_boxes'].split(", ")]
        box_coco=[b[0],b[1],b[2]-b[0],b[3]-b[1]]
        segpoints=get_seg(x['Measurement_coordinates'])
        #mask_rles=grabseg()
        rles=maskUtils.frPyObjects(segpoints,512,512)
        rle=maskUtils.merge(rles)
        area=maskUtils.area(rle)


        ann_info={
            "segmentation":segpoints,
            'image_id':x['id'],
            'id':x['id'],
            'iscrowd':0,
            'category_id':1,
            'area':float(area),
            'bbox':box_coco
        }
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

def pad(v):
    assert isinstance(v,int)
    if v<10:
        return "00"+str(v)+".png"
    if v<100:
        return "0"+str(v)+".png"
    return str(v)+".png"

def get_seg(st):
    v = [float(x) for x in st.split(", ")]

    return [[v[0],v[1],v[4],v[5],v[2],v[3],v[6],v[7]]]

def _conv(x):
    parts = x.split("_")
    pre = "_".join(parts[:-1])
    return osp.join(pre, parts[-1])


def grabseg(img,bbox,pts):
    mask = np.zeros(img.shape, dtype=np.uint8)#(zero is background)
    assert isinstance(pts,np.ndarray)
    cv2.fillPoly(mask,bbox,cv2.GC_PR_BGD)
    cv2.fillPoly(mask,pts,cv2.GC_FGD)

    #masked_image = cv2.bitwise_and(img, mask)
    #cv2.imwrite('image_masked.png', masked_image)


    bgdModel = np.zeros(img.shape,np.float64)
    fgdModel = np.zeros(img.shape,np.float64)

    cv2.grabCut(img, mask, None, bgdModel, fgdModel, 5, cv2.GC_INIT_WITH_MASK)
    mask2 = np.where((mask == 2) | (mask == 0), 0, 1).astype('uint8')
    return maskUtils.encode(mask2)


