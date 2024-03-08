## Object detection models

The models can be trained using [mmdetection](https://github.com/open-mmlab/mmdetection/tree/2.x).
The dataset needs to be converted to the COCO format to do the training via mmdetection.

3 models were trained for this research:

- [Faster R-CNN](https://github.com/open-mmlab/mmdetection/tree/2.x/configs/faster_rcnn)
- [Swin Transformer](https://github.com/open-mmlab/mmdetection/tree/2.x/configs/swin)
- [MobileNetV2-SSD](https://github.com/open-mmlab/mmdetection/tree/2.x/configs/ssd)

For usage together with Nvidia Jetson, [dusty-nv/pytorch-ssd](https://github.com/dusty-nv/pytorch-ssd) can be leveraged.
The dataset needs to be converted to the Pascal VOC format to do the training via dusty-nv/pytorch-ssd.

The following snippet may be useful when training models for the VisDrone datasets using mmdetection:

```
dataset_type = 'COCODataset'
classes = ('pedestrian', 'people', 'bicycle', 'car', 'van', 'truck', 'tricycle', 'awning-tricycle',
           'bus', 'motor',)

...

data = dict(
    train=dict(
        classes=classes,
        ann_file='datasets/VisDrone/annotations-VisDrone2019-DET-train.json',
        img_prefix='datasets/VisDrone/VisDrone2019-DET-train/images/'),
    val=dict(
        classes=classes,
        ann_file='datasets/VisDrone/annotations-VisDrone2019-DET-val.json',
        img_prefix='datasets/VisDrone/VisDrone2019-DET-val/images/'),
    test=dict(
        classes=classes,
        ann_file='datasets/VisDrone/annotations-VisDrone2019-DET-test-dev.json',
        img_prefix='datasets/VisDrone/VisDrone2019-DET-test-dev/images/'))
```
