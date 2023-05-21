import json
from typing import Union

from model import AnnotationView, Image, ImageList, AnnotationsByImage


def annotations_available(video: Union[str, None], annotations_path: Union[str, None]) -> bool:
    return video is not None and annotations_path is not None


def load_annotations(video, annotations_path) -> tuple[ImageList, AnnotationsByImage]:
    with open(annotations_path) as annotations_file:
        annotations_json = json.load(annotations_file)
        images = _to_images_of_video(annotations_json['images'], video)
        images_sorted = sorted(images, key=lambda i: i.file_name)
        image_ids = [image.id for image in images_sorted]

        annotations = dict()
        for image_id in image_ids:
            image_annotations = [annotation for annotation in annotations_json['annotations']
                                 if annotation['image_id'] == image_id]

            annotations[image_id] = []
            for image_annotation in image_annotations:
                (x, y, w, h) = image_annotation['bbox']
                category = int(image_annotation['category_id'])
                annotations[image_id].append(AnnotationView(x, y, w, h, category))

        return images_sorted, annotations


def _to_images_of_video(images, video) -> ImageList:
    video_name = video.split("/")[-2]
    images = [image for image in images
              if video_name in image['file_name']]
    return [Image(id=int(image['id']), file_name=image['file_name']) for image in images]
