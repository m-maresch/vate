from detection import ObjectDetector
from model import DetectionType
from torch_serve_prediction import TorchServePredictor


def get_object_detector(jetson: bool):
    cloud_predictor = TorchServePredictor(
        detection_model_url="http://127.0.0.1:9093/predictions/faster_rcnn_visdrone",
        detection_timeout=30,
        detection_type=DetectionType.CLOUD
    )

    if jetson:
        from jetson_prediction import JetsonPredictor
        return ObjectDetector(
            edge_predictor=JetsonPredictor(
                model="mb2-ssd-lite-512-100-epochs.onnx",
                labels="onnxlabels.txt",
                threshold=0.2,
                input_width=512,
                input_height=512
            ),
            cloud_predictor=cloud_predictor
        )
    else:
        return ObjectDetector(
            edge_predictor=TorchServePredictor(
                detection_model_url="http://127.0.0.1:9090/predictions/mobilenetv2_ssd_visdrone",
                detection_timeout=10,
                detection_type=DetectionType.EDGE
            ),
            cloud_predictor=cloud_predictor
        )
