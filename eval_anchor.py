import file_locs
from mmdet.datasets import DL_coco
from mmdet.core.evaluation.bbox_overlaps import bbox_overlaps
import torch
import numpy as np
import copy

import configs.dan.srs.retina_dl as cfg
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


def param_search():
    anchor_scales=[1]
    anchor_ratios=[1]

    for idx, i in enumerate(np.arange(0,3,0.3)):
        anchor_ratios.append(i)
        anchor_scales.append(i)
        if idx%10==9:
            print(anchor_scales,anchor_ratios)
            print(eval_anchor_cfg(anchor_scales,anchor_ratios))


def de_func(arr):
    assert len(arr)==5
    ratios=arr[:2].tolist()
    ratios.extend([1,1/ratios[0],1/ratios[1]])
    scales=arr[2:].tolist()
    return eval_anchor_cfg(scales,ratios)


#bounds are only for initialization
def de(fobj, bounds, mut=0.8, crossp=0.7, popsize=20, its=500,initializers=[]):
    dimensions = len(bounds)
    pop = np.random.rand(popsize, dimensions)
    assert len(initializers)<=popsize
    for i, x in enumerate(initializers):
        assert len(x)==2
        val=normalize(x[0],x[1],bounds)
        pop[i]=val

    min_b, max_b = np.asarray(bounds).T
    diff = np.fabs(min_b - max_b)
    pop_denorm = min_b + pop * diff
    fitness = np.asarray([fobj(ind) for ind in pop_denorm])
    best_idx = np.argmin(fitness)
    best = pop_denorm[best_idx]
    for i in range(its):

        start=time.time()
        for j in range(popsize):
            idxs = [idx for idx in range(popsize) if idx != j]
            a, b, c = pop[np.random.choice(idxs, 3, replace = False)]

            #mutant = a + mut * (b - c)
            mutant = np.clip(a + mut * (b - c), 0, 1)
            cross_points = np.random.rand(dimensions) < crossp
            if not np.any(cross_points):
                cross_points[np.random.randint(0, dimensions)] = True
            trial = np.where(cross_points, mutant, pop[j])
            trial_denorm = min_b + trial * diff
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

def normalize(scales,ratios,bounds):
    prenorm=scales.copy()
    prenorm.extend(ratios)
    assert len(prenorm)==len(bounds)
    norm=[(prenorm[i]-b[0])/(b[1]-b[0]) for i, b in enumerate(bounds)]
    norm=np.array(norm)
    assert norm.max()<=1 and norm.min()>=0
    return norm



def eval_anchor_cfg(anchor_scales,
                    anchor_ratios):

    my_cfg=copy.deepcopy(cfg.model['bbox_head'])
    if not (len(anchor_ratios)==len(anchor_scales)==0):
        my_cfg['anchor_scales']=anchor_scales
        my_cfg['anchor_ratios']=anchor_ratios
    anchor_head=build_head(my_cfg)
    anchs, _ =anchor_head.get_anchors(featmap_sizes,[met])
    all_anchors=torch.cat(anchs[0])
    all_anchors=all_anchors.cpu().numpy()

    s=score(boxes,all_anchors)


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

    de(de_func,[[1.5,2.5],[2,4],[3,4],[2,4],[2,4]],
       its=500, initializers=[])
    #print(list(it))
    print(eval_anchor_cfg([],[]))
    #summarize(boxes)



