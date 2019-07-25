import file_locs
from mmdet.datasets import DL_coco
from mmdet.core.evaluation.bbox_overlaps import bbox_overlaps
import torch
import numpy as np
import copy

import configs.dan.srs.retina_anchor as cfg
from mmdet.models import build_detector,build_head
import matplotlib.pyplot as plt
import time


import copy

def get_req():
    data=DL_coco(file_locs.csv_dir + "DL_valid.csv", file_locs.image_dir,
                 with_mask=True, use_context=True, test_mode=True)

    model_ = build_detector(cfg.model, train_cfg = cfg.train_cfg, test_cfg = cfg.test_cfg)


    im=data[0]['img'][0]
    im=im.reshape(-1,3,512,512)
    feats=model_.extract_feat(im)
    featmap_sizes = [featmap.size()[-2:] for featmap in feats]
    met=data[0]['img_meta'][0].data


    anns=data.coco.anns
    boxes= np.array([v['bbox'] for x, v in anns.items()])
    boxes[:,2]=boxes[:,0]+boxes[:,2]
    boxes[:,3]=boxes[:,1]+boxes[:,3]

    return featmap_sizes,met,boxes

featmap_sizes,met,boxes= get_req()


#inputs should already be denormalized
def de_func(arr):
    assert len(arr)==5
    scales=arr[:3].tolist()
    ratios=arr[3:].tolist()
    ratios.extend([1,1/ratios[0],1/ratios[1]])
    return eval_anchor_cfg(scales,ratios)


#bounds are only for initialization
def ml():
            idxs = [idx for idx in range(popsize) if idx != j]
            a, b, c = pop[np.random.choice(idxs, 3, replace = False)]

            mutant = a + mut * (b - c)
            #mutant = np.clip(a + mut * (b - c), 0, 1)
            cross_points = np.random.rand(dimensions) < crossp
            if not np.any(cross_points):
                cross_points[np.random.randint(0, dimensions)] = True
            trial = np.where(cross_points, mutant, pop[j])


            trial_denorm = min_b + trial * diff

            assert len(trial_denorm)==5
            scales=trial_denorm[:3].tolist()
            ratios=trial_denorm[3:].tolist()
            ratios.extend([1,1/ratios[0],1/ratios[1]])
            my_cfg=copy.deepcopy(cfg.model['bbox_head'])

            if not (len(ratios)==len(scales)==0):
                my_cfg['anchor_scales']=scales
                my_cfg['anchor_ratios']=ratios
            anchor_head=build_head(my_cfg)

            anchs, flags =anchor_head.get_anchors(featmap_sizes,[met])

            all_anchors=torch.cat(anchs[0])
            all_flags=torch.cat(flags[0])

            f_anchors=all_anchors[all_flags,:]
            a_mins=f_anchors.min(1)[0]
            f_anchors=f_anchors[a_mins>=0,:]

            x_max=f_anchors[:,2]
            f_anchors=f_anchors[x_max<=img_dim[0],:]
            y_max=f_anchors[:,3]
            f_anchors=f_anchors[y_max<=img_dim[1],:]

            f = fobj(trial_denorm)
            if f < fitness[j]:
                fitness[j] = f
                pop[j] = trial
                if f < fitness[best_idx]:
                    best_idx = j
                    best = trial_denorm

        end=time.time()
        print("iteration:",i," Time taken:",end-start,)
        print("best:",best," fitness:",fitness[best_idx])

def normalize(scales,ratios, bounds):
    prenorm=list(scales)
    prenorm.extend(ratios)
    assert len(prenorm)==len(bounds)
    norm=[(prenorm[i]-b[0])/(b[1]-b[0]) for i, b in enumerate(bounds)]
    norm=np.array(norm)
    assert norm.max()<=1 and norm.min()>=0
    return norm

def loss(gt_bboxes,anchor_boxes):
    pass

def eval_anchor_cfg(anchor_scales,
                    anchor_ratios,
                    img_dim=(512,512)):

    my_cfg=copy.deepcopy(cfg.model['bbox_head'])
    if not (len(anchor_ratios)==len(anchor_scales)==0):
        my_cfg['anchor_scales']=anchor_scales
        my_cfg['anchor_ratios']=anchor_ratios
    anchor_head=build_head(my_cfg)

    anchs, flags =anchor_head.get_anchors(featmap_sizes,[met])

    all_anchors=torch.cat(anchs[0])
    all_flags=torch.cat(flags[0])

    f_anchors=all_anchors[all_flags,:]
    a_mins=f_anchors.min(1)[0]
    f_anchors=f_anchors[a_mins>=0,:]

    x_max=f_anchors[:,2]
    f_anchors=f_anchors[x_max<=img_dim[0],:]
    y_max=f_anchors[:,3]
    f_anchors=f_anchors[y_max<=img_dim[1],:]



    f_anchors=f_anchors.cpu().numpy()

    s=score(boxes,f_anchors)

    #heights=boxes[:,3]-boxes[:,1]
    #widths=boxes[:,2]-boxes[:,0]
    #plt.scatter(widths,heights,s=0.5,c=s>0.8,cmap='coolwarm')
    #plt.savefig("out.png")
    return -s.sum()


def score(gt_boxes,anchors):
    overlaps=bbox_overlaps(gt_boxes,anchors)
    best=overlaps.max(1)
    #print(best)
    return best

#TAKES height w
def summarize(gt_boxes):
    heights=gt_boxes[:,3]-gt_boxes[:,1]
    widths=gt_boxes[:,2]-gt_boxes[:,0]
    plt.scatter(widths,heights,s=0.2)
    plt.savefig("out.png")




if __name__ == "__main__":

    de(de_func,[[0,1],[0,1],[1,2],[1,4],[1,5]],
       its=500, initializers=[
            ((0.5,0.5,1.5),(3,2))
        ])
    #print(list(it))
    print(eval_anchor_cfg([],[]))
    #summarize(boxes)



