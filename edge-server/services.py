from prediction import Predictor
from torch_serve_prediction import TorchServePredictor


def get_predictor(jetson: bool) -> Predictor:
    if jetson:
        from jetson_prediction import JetsonPredictor
        return JetsonPredictor(
            model="mb2-ssd-lite-512-100-epochs.onnx",
            labels="onnxlabels.txt",
            threshold=0.2,
            input_width=512,
            input_height=512
        )
    else:
        return TorchServePredictor(
            detection_model_url="http://127.0.0.1:9090/predictions/mobilenetv2_ssd_visdrone",
            detection_timeout=10
        )
