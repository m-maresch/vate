# Title (WIP)

Initial:
Distributed inference (or/and learning) for an object detection use case on the edge, leveraging serverless technology.

Using Raspberry Pis, AWS resources, Kubernetes and TensorFlow.

Iteration #1:
Federated learning for an object detection use case on the edge with serverless containers for model serving.

Iteration (#2):
Serverless containers for integrating and serving deep learning models at the edge.
- Using Knative and K3s/MicroK8s for connecting devices with their edge server
- Serving deep learning models on the edge server

Iteration (#3):
Semantic segmentation on edge containers for smart factories
- Serving a deep learning model on an Nvidia Jetson Nano
- Using two Raspberry Pis with cameras which capture video streams
- These connect to an Nvidia Jetson Nano where semantic segmentation on the video streams is done
- Using K3s/MicroK8s for edge containers
- Web-App where one can see the live video stream and the semantic segmentation results per device/Raspberry Pi with camera

Iteration (#4):
Building edge intelligence using stream processing
- Using two Raspberry Pis with cameras which capture video streams
- These connect to a more powerful machine (e.g. a laptop) in the local network where semantic segmentation on the video streams is done
- Raspberry Pis + laptop represent an edge site
- Using Python, Kubernetes, NATS, TensorFlow/PyTorch
- Web-App on edge site where one can see the live video stream and the semantic segmentation results per device/Raspberry Pi with camera
- Extension #1: System manager application runs in the cloud, used to manage edge sites, samples data from all edge sites and enables further stream processing in the cloud
- Extension #2: Compare technologies which can be used for edge stream processing, e.g. Kafka vs NATS vs Pulsar ...

Problems with:
- Distributed inference at the edge
    - Based on existing, state of the art architectures:
        - For agriculture, production sites: devices with connection to edge server (on-site)
        - For connected cars: cars with connection to edge infrastructure (MEC)
    - If the devices are too weak to do the inference themselves then they most likely have a edge server available to them which is capable of doing the inference. (e.g. agriculture, production sites)
    - Otherwise they likely do have the capability to do the inference themselves. (e.g. connected cars)
    - Not a lot of use cases where this does not hold.
- Federated learning at the edge: Not a lot of use cases, as you need all of:
    - Data cannot be shared for some reason
    - Data is generated locally
    - The participants can agree on the data format used (for our FL scenario)
    - The participants are sufficently powerful to train a model
    - The statistical challenges of FL are not a blocker

## Topic

From edge computing we have devices and edge servers along with potentially the cloud. For centralized federated learning we have a single server per model. This results in 2 possible variants for the system architecture which can be implemented:
- Variant 1: One model for the whole system, implies that
    - One FL server running in the cloud
    - Devices communicate with edge server
    - Edge server aggregates data from devices
    - Edge server participates in FL
    - Model is deployed in the cloud (or same model on every edge server)
- Variant 2: One model per edge server, implies that
    - One FL server running at every edge server
    - Devices participate in FL
    - Models are deployed at the edge servers
    - Cloud might not be used at all
    
Each variant is suitable for different Use Cases.

Serverless edge computing focused on FaaS and artificial intelligence don't go well together:
- AI processing (training, testing): Involves long-running tasks and is IO-intensive at the edge [6]
- Serverless functions: Are short-running functions, CPU-bound

There is a mismatch. [5]

When talking about serverless, we're referring to not having to manage servers, not FaaS. Specifically serverless containers will be evaluated, as they are suitable for long running tasks as well.

## Product

## System Design

Centralized federated learning will be done, thus there is a server for orchestration and coordination.

On Kubernetes-based edge architecture: See [this](https://www.lfedge.org/2021/02/11/kubernetes-is-paving-the-path-for-edge-computing-adoption/)

3 approaches:
 - Deploy Kubernetes cluster on edge nodes directly (in theory no cloud if not necessary, can be done using e.g. K3s or MicroK8s)
 - Control plane in cloud, manages edge nodes (can be done using KubeEdge)
 - Virtual kubelets: Cloud contains abstract of edge nodes/pods

Depends on the Use Case what is the most appropriate, e.g.:
- MicroK8s + Knative when only edge is needed or cloud and edge are mostly independant
- KubeEdge when cloud and edge are needed and edge depends on cloud

Alternative to Kubernetes-based edge architectures: Using AWS exclusively, see [here](https://aws.amazon.com/blogs/architecture/applying-federated-learning-for-ml-at-the-edge/) for a federated learning at the edge example.

### Technology evaluation

Various solutions:
- Knative Serving: In theory suitable for edge as is
    - Would be deployed on K3s/MicroK8s
    - Resource usage rather high (official recommendation: >4 GB of memory, see [here](https://knative.dev/docs/install/operator/knative-with-operators/#prerequisites))
    - ARM Support: Yes (although does not seem to be widely used), see [here](https://github.com/knative/serving/issues/8320)
- Kubeflow: Not suitable for edge as is
    - Would be deployed on K3s/MicroK8s
    - Resource usage very high (e.g. see [here](https://charmed-kubeflow.io/docs/operators-and-bundles), >8 GB of memory for kubeflow-lite)
    - ARM Support: No, see [here](https://github.com/kubeflow/kubeflow/issues/2337)

Federated learning
- TensorFlow Federated: No multi-machine support, only for simulations
- PySyft: OK
- Flower: OK, seems to be the simplest option for our scenario, was made with exactly this in mind (see [4])

Kubernetes for edge:
- K3s: OK, would involve edge only, no serverless infrastructure out of the box
- MicroK8s: OK, would involve edge only, no serverless infrastructure out of the box
- KubeEdge: OK, would involve cloud and edge
    - ARM Support: Yes, native support, see [here](https://kubeedge.io/en/)

Serverless infrastructure:
- faasd: No multi-node cluster support
- OpenFaaS: OK
- Knative: OK, see above

## Related research
1. [A Survey on Federated Learning and its Applications for Accelerating Industrial Internet of Things](https://arxiv.org/pdf/2104.10501.pdf)
2. [Exploiting Unlabeled Data in Smart Cities using Federated Edge Learning](https://arxiv.org/pdf/2001.04030.pdf)
3. [On-device federated learning with Flower](https://arxiv.org/pdf/2104.03042.pdf)
4. [Flower: A Friendly Federated Learning Framework](https://arxiv.org/pdf/2007.14390.pdf)
5. [Serverless Edge Computing: Vision and Challenges](https://dsg.tuwien.ac.at/team/sd/papers/AusPDC_2021_SD_Serverless.pdf)
6. [Serverless Computing: One Step Forward, Two Steps Back](https://arxiv.org/pdf/1812.03651.pdf)
