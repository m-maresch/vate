# Title (WIP)

Initial:
Distributed inference (or/and learning) for an object detection use case on the edge, leveraging serverless technology.

Using Raspberry Pis, AWS resources, Kubernetes and TensorFlow.

Current iteration:
Federated learning for an object detection use case on the edge with serverless containers for model serving.

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
