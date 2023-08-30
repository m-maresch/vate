from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval
from typing import List, Tuple

from model import DetectionView

METRICS = ['mAP', 'mAP_50', 'mAP_75', 'mAP_s', 'mAP_m', 'mAP_l']
METRIC_TO_COCO_ID = {
    'mAP': 0,
    'mAP_50': 1,
    'mAP_75': 2,
    'mAP_s': 3,
    'mAP_m': 4,
    'mAP_l': 5,
}


def evaluate_detections(detections: List[DetectionView], annotations_path: str) -> Tuple[float, float]:
    results = [
        dict(
            image_id=detection.frame_id,
            bbox=[detection.x, detection.y, detection.w, detection.h],
            score=detection.score,
            category_id=detection.category
        ) for detection in detections
    ]

    img_ids_start = results[0]['image_id']
    img_ids_end = results[-1]['image_id']

    coco_gt = COCO(annotations_path)
    coco_dt = coco_gt.loadRes(results)
    coco_eval = COCOeval(coco_gt, coco_dt, 'bbox')
    coco_eval.params.catIds = list(range(0, 10))
    coco_eval.params.imgIds = list(range(img_ids_start, img_ids_end + 1))
    coco_eval.params.maxDets = [100, 500, 1000]

    coco_eval.evaluate()
    coco_eval.accumulate()
    coco_eval.summarize()

    eval_results = dict()
    for metric in METRICS:
        coco_id = METRIC_TO_COCO_ID[metric]
        stat = coco_eval.stats[coco_id]
        eval_results[metric] = float(f'{stat:.4f}')
    print(f'Bounding box evaluation results: {eval_results}')

    return eval_results['mAP'], eval_results['mAP_50']
