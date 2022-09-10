# Title (WIP)

Distributed inference (or/and learning) for an object detection use case on the edge, leveraging serverless technology.

Using Raspberry Pis, AWS resources, Kubernetes and TensorFlow.

## Topic

## Product

## System Design

On Kubernetes-based edge architecture: See [this](https://www.lfedge.org/2021/02/11/kubernetes-is-paving-the-path-for-edge-computing-adoption/)

3 approaches:
 - Deploy Kubernetes cluster on edge nodes directly (in theory no cloud if not necessary, can be done using e.g. K3s or MicroK8s)
 - Control plane in cloud, manages edge nodes (can be done using KubeEdge)
 - Virtual kubelets: Cloud contains abstract of edge nodes/pods

Depends on the Use Case what is the most appropriate.

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
- Flower: OK, seems to be the simplest option for our scenario, was made with exactly this in mind (see the Flower paper)

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
- [A Survey on Federated Learning and its Applications for Accelerating Industrial Internet of Things](https://arxiv.org/pdf/2104.10501.pdf)
- [Exploiting Unlabeled Data in Smart Cities using Federated Edge Learning](https://arxiv.org/pdf/2001.04030.pdf)
- [On-device federated learning with Flower](https://arxiv.org/pdf/2104.03042.pdf)
- [Flower: A Friendly Federated Learning Framework](https://arxiv.org/pdf/2007.14390.pdf)
