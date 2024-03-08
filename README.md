# VATE: Edge-Cloud System for Object Detection in Real-Time Video Streams

This repository contains the source code of the VATE system.

The clip shows a video from the
[VisDrone-VID2019 dataset](https://github.com/VisDrone/VisDrone-Dataset) being processed by VATE.

All details on VATE can be found in the paper (more information on this will be added here soon).

Abstract:
> In the realm of edge intelligence, emerging video analytics applications are often based on resource constrained edge
> devices. These applications need systems which are able to provide both low-latency and high-accuracy video stream
> processing, such as for object detection in real-time video streams. State-of-the-art systems tackle this challenge by
> leveraging edge computing and cloud computing. Such edge-cloud approaches typically combine low-latency results from
> the edge and high accuracy results from the cloud when processing a frame of the video stream. However, the accuracy
> achieved so far leaves much room for improvement. Furthermore, using more accurate object detection often requires
> having more capable hardware. This limits the edge devices which can be used. Applications related to autonomous
> drones, with the drone being the edge device, give one example. A wide variety of objects needs to be detected
> reliably for drones to operate safely. Drones with more computing capabilities are often more expensive and suffer
> from short battery life, as they consume more energy.
>
> In this paper, we introduce VATE, a novel edge-cloud system for object detection in real-time video streams. An
> enhanced approach for edge-cloud fusion is presented, leading to improved object detection accuracy. A novel
> multi-object tracker is introduced, allowing VATE to run on less capable edge devices. The architecture of VATE
> enables it to be used when edge devices are capable of running on-device object detection frequently and when edge
> devices need to minimise on-device object detection to preserve battery life. Its performance is evaluated on a
> challenging, drone-based video dataset. The experimental results show that VATE improves accuracy by up to 27.5%
> compared to the state-of-the-art system, while running on less capable and cheaper hardware.

More information on the object detection models can be found in `detection-models/`.

## Edge device

The edge device component can be used as follows:

```
python main.py "/path/to/video-sequences" "/path/to/annotations.json" [--detection-rate int] [--sync] [--ipc]
```

This component takes the following arguments:

- `/path/to/video-sequences` to specify the videos which will be processed by VATE. Videos are given as sequences of
  JPEG files. The alternative is to use the camera of the device as the video source. The camera is used by default.
- `/path/to/annotations.json` can be used to specify annotations for the videos if available. The annotations are
  displayed along with the objects detected by the system and are used to evaluate the accuracy of the system.
  Annotations are given in COCO format as a JSON file.
- `detection-rate` influences how often frames are sent to the edge server.
- `ipc` to send edge server requests via IPC. The default is to communicate with the edge server via TCP on port 8000.
- `sync` to use sync mode. The default is to use async mode.

Examples using [VisDrone2019-VID](https://github.com/VisDrone/VisDrone-Dataset):

```
python main.py "../detection-models/datasets/VisDrone/VisDrone2019-VID-test-dev/sequences/*" "../detection-models/datasets/VisDrone/annotations-VisDrone2019-VID-test-dev.json" --detection-rate 5 --sync

python main.py "../detection-models/datasets/VisDrone/VisDrone2019-VID-test-dev/sequences/*" "../detection-models/datasets/VisDrone/annotations-VisDrone2019-VID-test-dev.json" --detection-rate 10

python main.py "../detection-models/datasets/VisDrone/VisDrone2019-VID-test-dev/sequences/uav0000009_03358_v" "../detection-models/datasets/VisDrone/annotations-VisDrone2019-VID-test-dev.json" --detection-rate 10

python main.py "../detection-models/datasets/VisDrone/VisDrone2019-VID-test-dev/sequences/uav0000009_03358_v" "../detection-models/datasets/VisDrone/annotations-VisDrone2019-VID-test-dev.json" --detection-rate 5 --sync
```

## Edge server

The edge server component can be used as follows:

```
python main.py --no-jetson [--ipc]

python main.py --jetson [--ipc]
```

This component takes the following arguments:

- `jetson` or `no-jetson`, to use either the Jetson predictor or the TorchServe predictor. The default is to use the
  Jetson predictor.
- `ipc` to receive requests via IPC. The default is to receive requests via TCP on port 8000.

## Cloud server

The cloud server corresponds to a model being served by TorchServe.

The steps provided [here](https://mmdetection.readthedocs.io/en/v2.28.1/useful_tools.html#model-serving) can be used to
serve an object detection model.

## Dependencies

Thanks to everyone contributing to any of the following projects:

- mmdetection
- dusty-nv/pytorch-ssd and qfgaohao/pytorch-ssd
- dusty-nv/jetson-inference
- PyTorch and TorchServe
- NumPy
- Matplotlib
- Requests
- OpenCV
- COCO API
- ZeroMQ
- SciPy
- Pydantic
- VisDrone-Dataset
