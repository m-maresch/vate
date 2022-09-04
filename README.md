# Title (WIP)

Distributed inference (or/and learning) for an object detection use case on the edge, leveraging serverless technology.

Using Raspberry Pis, AWS resources, Kubernetes and TensorFlow.

## Topic

## Product

## System Design

### Technology evaluation

Various solutions:
- Knative: Not suitable for edge as is, resource usage too high (official recommendation: >4 GB of memory, see [here](https://knative.dev/docs/install/operator/knative-with-operators/#prerequisites))
- Kubeflow: Not suitable for edge as is, same as above

Federated learning
- TensorFlow Federated: No multi-machine support, only for simulations
- PySyft: OK
- Flower: OK, seems like the simplest option for our scenario, was made with exactly this in mind (see the Flower paper)

Kubernetes for edge:
- K3s: OK, no serverless infrastructure
- MicroK8s: OK, no serverless infrastructure

Serverless infrastructure:
- faasd: No multi-node cluster support
- OpenFaaS: OK

## Related research
- [A Survey on Federated Learning and its Applications for Accelerating Industrial Internet of Things](https://arxiv.org/pdf/2104.10501.pdf)
- [Exploiting Unlabeled Data in Smart Cities using Federated Edge Learning](https://arxiv.org/pdf/2001.04030.pdf)
- [On-device federated learning with Flower](https://arxiv.org/pdf/2104.03042.pdf)
- [Flower: A Friendly Federated Learning Research Framework](https://arxiv.org/pdf/2007.14390.pdf)
