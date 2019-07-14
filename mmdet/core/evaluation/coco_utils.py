import mmcv
import numpy as np
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval

from .recall import eval_recalls


def coco_eval(result_files, result_types, coco, max_dets=(4, 100, 300, 1000)):
    for res_type in result_types:
        assert res_type in [
            'proposal', 'proposal_fast', 'bbox', 'segm', 'keypoints'
        ]

    if mmcv.is_str(coco):
        coco = COCO(coco)
    assert isinstance(coco, COCO)

    if result_types == ['proposal_fast']:
        ar = fast_eval_recall(result_files, coco, np.array(max_dets))
        for i, num in enumerate(max_dets):
            print('AR@{}\t= {:.4f}'.format(num, ar[i]))
        return

    for res_type in result_types:
        result_file = result_files[res_type]
        assert result_file.endswith('.json')

        coco_dets = coco.loadRes(result_file)
        img_ids = coco.getImgIds()
        iou_type = 'bbox' if res_type == 'proposal' else res_type

        FROC_eval(coco, coco_dets)
        '''
        cocoEval = COCOeval(coco, coco_dets, iou_type)
        cocoEval.params.imgIds = img_ids
        if res_type == 'proposal':
            cocoEval.params.useCats = 0
        cocoEval.params.maxDets=list(max_dets)
        cocoEval.evaluate()
        cocoEval.accumulate()
        cocoEval.summarize()
        '''


def FROC_eval(coco, coco_dets):
    n_imgs = len(coco.getImgIds())
    ret = []
    for p_thrs in np.arange(0.0, 1.0, 0.05):
        t_positives = 0
        f_positives = 0
        for img_id in coco.getImgIds():
            gt_ann = coco.imgToAnns[img_id][0]
            assert len(coco.imgToAnns[img_id]) == 1, "DL only has one ann per image"
            gt_box = gt_ann['bbox']
            dt_boxes = [ann['bbox'] for ann in coco_dets.imgToAnns[img_id] if ann['score'] >= p_thrs]

            if len(dt_boxes) == 0:
                continue
           
            ious = [IOU(gt_box, box) for box in dt_boxes]
            successes = max([int(val >= 0.5) for val in ious])
            fails = sum([val < 0.5 for val in ious])

            t_positives += successes
            f_positives += fails

        tps, fps = t_positives / n_imgs, f_positives / n_imgs
        ret.append((fps, tps))

    print(ret)

#ASSUMES INPUTS ARE IN (x,y, width,height) format
def IOU(gt_box, det_box):
    # compute overlaps
    # intersection
    gts=[gt_box[0],gt_box[1],gt_box[0]+gt_box[2],gt_box[1]+gt_box[3]]
    box1 = [det_box[0], det_box[1], det_box[0] + det_box[2], det_box[1] + det_box[3]]
    ixmin = max(gts[0], box1[0])
    iymin = max(gts[1], box1[1])
    ixmax = max(gts[2], box1[2])
    iymax = max(gts[3], box1[3])
    iw = max(ixmax - ixmin + 1., 0.)
    ih = max(iymax - iymin + 1., 0.)
    inters = iw * ih

    # union
    uni = ((box1[2] - box1[0] + 1.) * (box1[3] - box1[1] + 1.) +
           (gts[2] - gts[0] + 1.) *
           (gts[3] - gts[1] + 1.) - inters)

    overlaps = inters / uni
    # ovmax = np.max(overlaps)
    # jmax = np.argmax(overlaps)
    return overlaps


def fast_eval_recall(results,
                     coco,
                     max_dets,
                     iou_thrs=np.arange(0.5, 0.96, 0.05)):
    if mmcv.is_str(results):
        assert results.endswith('.pkl')
        results = mmcv.load(results)
    elif not isinstance(results, list):
        raise TypeError(
            'results must be a list of numpy arrays or a filename, not {}'.
                format(type(results)))

    gt_bboxes = []
    img_ids = coco.getImgIds()
    for i in range(len(img_ids)):
        ann_ids = coco.getAnnIds(imgIds=img_ids[i])
        ann_info = coco.loadAnns(ann_ids)
        if len(ann_info) == 0:
            gt_bboxes.append(np.zeros((0, 4)))
            continue
        bboxes = []
        for ann in ann_info:
            if ann.get('ignore', False) or ann['iscrowd']:
                continue
            x1, y1, w, h = ann['bbox']
            bboxes.append([x1, y1, x1 + w - 1, y1 + h - 1])
        bboxes = np.array(bboxes, dtype=np.float32)
        if bboxes.shape[0] == 0:
            bboxes = np.zeros((0, 4))
        gt_bboxes.append(bboxes)

    recalls = eval_recalls(
        gt_bboxes, results, max_dets, iou_thrs, print_summary=False)
    ar = recalls.mean(axis=1)
    return ar


def xyxy2xywh(bbox):
    _bbox = bbox.tolist()
    return [
        _bbox[0],
        _bbox[1],
        _bbox[2] - _bbox[0] + 1,
        _bbox[3] - _bbox[1] + 1,
    ]


def proposal2json(dataset, results):
    json_results = []
    for idx in range(len(dataset)):
        img_id = dataset.img_ids[idx]
        bboxes = results[idx]
        for i in range(bboxes.shape[0]):
            data = dict()
            data['image_id'] = img_id
            data['bbox'] = xyxy2xywh(bboxes[i])
            data['score'] = float(bboxes[i][4])
            data['category_id'] = 1
            json_results.append(data)
    return json_results


def det2json(dataset, results):
    json_results = []
    for idx in range(len(dataset)):
        img_id = dataset.img_ids[idx]
        result = results[idx]
        for label in range(len(result)):
            bboxes = result[label]
            for i in range(bboxes.shape[0]):
                data = dict()
                data['image_id'] = img_id
                data['bbox'] = xyxy2xywh(bboxes[i])
                data['score'] = float(bboxes[i][4])
                data['category_id'] = dataset.cat_ids[label]
                json_results.append(data)
    return json_results


def segm2json(dataset, results):
    bbox_json_results = []
    segm_json_results = []
    for idx in range(len(dataset)):
        img_id = dataset.img_ids[idx]
        det, seg = results[idx]
        for label in range(len(det)):
            # bbox results
            bboxes = det[label]
            for i in range(bboxes.shape[0]):
                data = dict()
                data['image_id'] = img_id
                data['bbox'] = xyxy2xywh(bboxes[i])
                data['score'] = float(bboxes[i][4])
                data['category_id'] = dataset.cat_ids[label]
                bbox_json_results.append(data)

            # segm results
            # some detectors use different score for det and segm
            if len(seg) == 2:
                segms = seg[0][label]
                mask_score = seg[1][label]
            else:
                segms = seg[label]
                mask_score = [bbox[4] for bbox in bboxes]
            for i in range(bboxes.shape[0]):
                data = dict()
                data['image_id'] = img_id
                data['score'] = float(mask_score[i])
                data['category_id'] = dataset.cat_ids[label]
                segms[i]['counts'] = segms[i]['counts'].decode()
                data['segmentation'] = segms[i]
                segm_json_results.append(data)
    return bbox_json_results, segm_json_results


def results2json(dataset, results, out_file):
    result_files = dict()
    if isinstance(results[0], list):
        json_results = det2json(dataset, results)
        result_files['bbox'] = '{}.{}.json'.format(out_file, 'bbox')
        result_files['proposal'] = '{}.{}.json'.format(out_file, 'bbox')
        mmcv.dump(json_results, result_files['bbox'])
    elif isinstance(results[0], tuple):
        json_results = segm2json(dataset, results)
        result_files['bbox'] = '{}.{}.json'.format(out_file, 'bbox')
        result_files['proposal'] = '{}.{}.json'.format(out_file, 'bbox')
        result_files['segm'] = '{}.{}.json'.format(out_file, 'segm')
        mmcv.dump(json_results[0], result_files['bbox'])
        mmcv.dump(json_results[1], result_files['segm'])
    elif isinstance(results[0], np.ndarray):
        json_results = proposal2json(dataset, results)
        result_files['proposal'] = '{}.{}.json'.format(out_file, 'proposal')
        mmcv.dump(json_results, result_files['proposal'])
    else:
        raise TypeError('invalid type of results')
    return result_files
