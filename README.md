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

Kubernetes for edge:
- K3s: OK, no serverless infrastructure
- MicroK8s: OK, no serverless infrastructure

Serverless infrastructure:
- faasd: No multi-node cluster support
- OpenFaaS: OK
