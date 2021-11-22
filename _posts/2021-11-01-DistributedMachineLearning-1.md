---
layout: post           									# (require) default post layout
title: "Distributed Machine Learning I"            # (require) a string title
date: 2021-11-01       									# (require) a post date
categories: [machineLearning]   # (custom) some categories, but makesure these categories already exists inside path of `category/`
tags: [DistributedML]           	# (custom) tags only for meta `property="article:tag"`


---



Distributed Machine Learning

## Nodes/Parameter Distribution

Node들이 어떻게 distribute되는지는 **network topology, bandwidth, communication latency, parameter update frequency, desired fault tolerance**에 의해 결정된다.

Nodes distribution scheme은 크게 두 가지 방식으로 구분된다. 

### Centralized

하나의 node (또는 하나의 group of nodes)가 따로 model parameter들의 synchronization을 위해 존재한다. 이 node는 "parameter server"라고 불린다. 이 방식에서는 더 쉽게 model parameter들을 synchronize할 수 있다는 장점이 있지만, parameter server가 거대한 cluster의 bottleneck이 되어 속도가 저하되는 위험이 존재한다. 이렇게 single point of failure가 문제를 이르키는 위험을 줄이기 위해서는 multiple parallel servers를 사용하고 적절한 storage redundancy가 적용되는지 확실하게 확보하는 방법등이 있다. 

아래 그림은 parallel SGD(in data parallelism) 가 parameter server를 사용할때에 algorithm이 workers(servers)에게 model을 broadcast하는 단계부터 시작한다. Each worker reads its own split from the mini-batch in each training iteration, computing its own gradients, and sending those gradients to one or more parameter servers. The parameter servers aggregate all the gradients from the workers and wait until all workers have completed before they calculate the new model for the next iteration, which is then broadcasted to all workers.

![centralized](C:\SJL\스터디_분산ML_system\figures\centralized.png)

### Decentralized

De-centralized 방식으로는 각각의 node가 다른 모든 node들과 직접 소통하여 parameter들을 update한다. 이렇게 peer-to-peer update방식은 빠르고, sparse updates를 만들어서 변경된 부분만 update할 수있고, 또 single point of failure가 존재하지 않는 장점이 있다.

parallel SGD(in data parallelism)의 경우 다음 그림과 같이 decentralized scheme을 사용한다. 이때 ring-allreduce방식에 의존하여 nodes들간의 parameter updates를 communicate한다. ring-allreduce architecture에는 workers로 부터 gradients를 aggregate하는 central server가 부재인 대신에, 각 training iteration에서 each worker read its own split for a mini-batch, calculates its gradients, sends it gradients to its successor neighbor on the ring, and receives gradients from its predecessor neighbor on the ring.

![decentralized](C:\SJL\스터디_분산ML_system\figures\decentralized.png)

Decentralized scheme으로 centralized scheme대비 performance를 향상시킨 cases:

[2017] SGD centralized vs. decentralized: https://www.scs.stanford.edu/17au-cs244b/labs/projects/addair.pdf (experimental demonstration code @ github: https://github.com/tgaddair/decentralizedsgd)

[2019] layered-SGD: https://arxiv.org/pdf/1906.05936.pdf

[2017]PSGD(parallel stochastic gradient descent) case study: https://proceedings.neurips.cc/paper/2017/file/f75526659f31040afeb61cb7133e4e6d-Paper.pdf



### Network Topologies

ML deployment를 설계하는데에 주요 요소 중 하나는 cluster안에 computer(a.k.a machine, node, or worker)들이 형성하는 구조이다. 이 구조는 network topology라고 명칭하고 어떤 parallelism(data vs. model)과 communication 방식을 적용해서 system을 어떤 수준의 degree of distribution으로 디자인 할지를 결정한다. 그리고 어떤 topology를 사용할지는 대응하려는 problem, data set, cluster size, 또 그외의 다른 주요 factor들에 알맞는 방법으로 설정해야한다.

Node간의 communication과 parameter updates 방식으로 distribution의 degree(수준)을 설정한다. 다음과 같이 4단계의 degrees of distribution으로 구분할 수 있다: centralized (ensembling), decentralized as tree, decentralized with parameter server, fully distributed.

![topology](C:\SJL\스터디_분산ML_system\figures\Distributed_ML_topologies.PNG)

Degree of distribution:

- **centralized(ensembling):**

  aggregation을 위한 hierarchical 방식이다. 

  Single model만으로 문제를 정확하게 해결할 수 없는 경우에는 **정확도를 개선**하기위해 여러개의 model을 합해서 ensemble Learning을 진행 할 수도 있다. 예를 들어서, inherently distributed data source를 기반으로 machine learning algorithm이 훈련되고 centralization이 가능한 옵션이 아닐때에는 두 개의 separate stages에서 훈련을 수행하는 것이 요구된다. First stage는 local sites (data가 stored되는 곳 ), second stage는 global site(first stage의 individual 결과들이 합해지는 곳)이다. 이런 통합은 global site에서 ensemble method를 통해 수행될 수 있다.

  대표적인 ensemble method들은 다음과 같다:

  - Bagging: 여러개의 classifier를 만들어서 이들을 하나로 통합한다.
  - Boosting: 이전 모델로 misclassified된 data를 기반으로 새로운 모델을 훈련시켜서 정확도를 개선해나아가는 방식이다.
  - Bucketing: 여러개의 다른 모델을 훈련시키고 결국 가장 좋은 성과를 내는 모델을 선택하는 방식이다.
  - Random Forests: 여러개의 decision tree model을 사용하고 각 tree model의 prediction의 평균값을 통해 전체적인 정확도를 개선하는 방식이다. forest를 구성하는 각각의 tree model들에게 동일한 voting power가 주어진다. (equally important?)
  - Stacking: variance를 감소시키기 위해 여러개의 classifier를 dataset에 훈련시키고, 여러개의 classifier의 output을 새로운 classifier에게 input해서 variance를 줄이는 방식이다.
  - Learning Classifier Systems(LCSs): modular system of learning process이다. LCS는 dataset 의 data points를 기반으로 반복한다. 각 iteration에서 learning process를 모두 complete한다. LCS는 제한된 개수의 rule을 기반으로 작동된다는 것이 주요 요소이다. LCS는 dataset에 따라서 매우 다른 특성을 가질 수 있다. 

- **decentralized:** decentralized system에서는 intermediate aggregation이 허용된다. 두 가지 종류로 나누어볼 수 있다.

  - decentralized tree: with replicated model that is consistently updated when the aggregate is broadcast to all nodes such as in tree topologies.
  - decentralized parameter server: Parameter server들을 기준으로 모델이 나누어져있음 (with a partitioned model that is sharded over multiple parameter servers)

- **fully distributed(peer to peer)**

  특정 node에 특정한 세부적인 역할이 주어지지 않는다. 독립적인 node들이 ensemble되어 함께 solution을 형성한다. 



Distributed machine learning clusters에서 주로 사용되는 topologies는 다음과 같다.

Topologies:

- **trees:** 나무와 비슷한 형태인 tree topologies는 scale과 manage하기 매우 쉽다. 이 구조에서는 각 node가 본인의 부모와 자식 node 들과만 소통하면 된다. (e.g., AllReduce paradigm에서 global gradient를 계산하기 위해서 tree의 nodes들이 본인의 child node들의 local gradients를 모아서 구한 합을 parent node에게 전달한다.) 

- **rings:** Communication system이 broadcast를 위한 충분한 지원을 제공하지 못하거나 또는 communication overhead를 최소 수준으로 유지해야하는 경우, AllReduce pattern을 위한 ring topologies는 오직 neighboring node들만 message를 통해 synchronize하도록 요구하면서 구조를 단순화한다. (이런 방식은 multiple GPU들 사이에서 주로 사용된다.)

  (Network topology에서 ring은 각 node가 only 두개의 neighboring nodes와 communicate하며, message가 대부분 한 방향으로 전송되는 특징이있음. 반면, P2P는 두개이상의 nodes간의 연결을 형성하는 형태이고 ring과는 다르게 message의 전달 방향이 하나의 방향으로 제한되어있지 않다.)

- **parameter server(PS):** 이 paradigm은 decentralized set of workers를 centralized set of masters와 함께 사용하면서 node들간의 shared state(공유하는 상태)를 유지한다. 각 Parameter Server의 shard(i.e., a horizontal data partition that contains a subset of the total dataset 또는partition된 model의 한 부분)에 model parameter들이 store된다. 여기에서 모든 client가 key-value store처럼 읽고 쓴다. Shard안의 모든 model parameter들이 globally 공유되는 memory에 포함되어있어서 model을 보다 쉽게 inspect할 수 있는 점이 장점이다. 이 paradigm의 단점은 model내의 communication을 모두 총괄하는 Parameter Server가 bottleneck이슈를 발생시킬 수 있다는 것이다. 이런 문제를 완화하기위해 computation과 communication을 bridging하는 technique들이 사용된다. 

- **peer to peer:** Fully distributed model에서는 centralized state과는 반대로 각 node가 parameters의 copy를 가지고있고 worker들은 직접적으로 서로 소통한다. 이 방식의 장점은 centralized model보다 보통 상대적으로 높은 scalability를 가지고 있다는 점과 system 내에서 single point of failure를 제거 한다는 점이 있다. 이 방식을 구현하는 peer-to-peer network을 예시로 들어보자면, node들이 모든 다른 node들에게 update를 broadcast 해서 data-parallel processing framework을 형성한다.

이 외에도 data 그리고/또는 program을 여러개의 machines들에 평등하게 distribute 하는 방법은 여러가지가 있는데, distribution 방법이 model 훈련에 얼마나 많은 communication이 필요한지를 결정한다. 



## Consistency in parallel processing

Data parallel방식이든, model parallel방식이든 system내 node들간의 communication 방식에 따라서 parameter들이 initialize되고, weights/ biases가 update되며 model의 consistency를 결정한다. 

잠깐 review - parallel processing이 concurrent process과 다른점은? : https://medium.com/@itIsMadhavan/concurrency-vs-parallelism-a-brief-review-b337c8dac350. Processing의 구분:

- sequential - doing one task at a time. begin the next only after the current task is complete
- concurrent - doing more than one task in progress at the same time, not at the same time instant
- parallel - doing more than one task simultaneously at exact same time instant

Weights, biases를 update하여 model consistency를 확보하는 방법이 두 가지가 있다: **synchronous vs. asynchronous updates**



### Synchronous

model내 parameter들을 synchronize하며 진행하는 방식이여서 as fast as the slowest worker (worker 중 old H/W가 있다면 속도 이슈가 발생할 수 있음)

예시:

-만약 10k images가 data set에 있고, 10개의 nodes로 분산하여 distributed training을 진행하고 있다면, 각각의 node에 1k images가 주어지고, first iteration이 완료되면 새롭게 update된 model parameter들이 parameter server에 보내진다. 이 방법에서는 server가 모든 node들이 완료할때까지 기다려야해서 만약 하나의 node라도 멈추거나 늦어지는 문제가 발생한다면, 전체적인 training의 속도를 저하시킬 수 있는 위험이 있다. 

-Each SGD iteration runs on a mini-batch of training samples. In synchronous training, all the devices train their local model using different parts of data from a single (large) mini-batch. They then communicate their locally calculated gradients (directly or indirectly) to all devices. Only after all devices have successfully computed and sent their gradients the model is updated. The updated model is then sent to all nodes along with splits from the next mini-batch.

장점: model consistency 보장되고, staleness(얼마나 update가 뒤쳐져있는지)가 낮음

단점: communication-intensive, overall 속도(as fast as the slowest worker), bottleneck 위험 있음



### Asynchronous

모든 node들이 완료될때까지 기다리지않고 각 node가 완료되는대로 공유된다. 이 방법은 cluster utilization(활용도)를 높이고 전체적인 훈련 시간을 단축시킬 수 있지만(as fast as how fast each worker can be), gradient problem을 발생 시킬 위험이 있다.

Distributed된 machine들 중 한 machine이 더 worn-out되었던지 느린 I/O를 가지고 있기때문에 전체의 속도에 영향을 줄 수 있어서 asynchronous 처럼 bottleneck을 방지할 수 있는 asynchronous 방식을 선호하기도 한다.

예시:

-In asynchronous training, no device waits for updates to the model from any other device. The devices can run independently and share results as peers or communicate through one or more central servers known as “parameter” servers. In the peer architecture, each device runs a loop that reads data, computes the gradients, sends them (directly or indirectly) to all devices, and updates the model to the latest version.

장점: overall 속도(as fast as how fast each worker can be)

단점: accuracy degradation due to delay in parameter update, gradient problem발생 위험 있음



### 주요 communication/synchronization 방식

현실적으로 synchronous 방식은 32~50 nodes규모의 model에 주로 사용되고 그 이상으로 더 큰 cluster 또는 heterogeneous environment를 위해서는 asynchronous방식이 사용된다. (according to survey:https://arxiv.org/pdf/1802.09941.pdf)

Synchronous와 asynchronous 두 가지 방법 모두 centralized 또는 decentralized 방식에 적용될 수 있다. 예를 들면, 만약 synchronous updates와 centralized scheme의 cluster를 setup 했다면, separate parameter service가 따로 존재하고 이쪽으로 모든 node들이 updates를 전송하고, parameter service가 모든 node들의 update를 전달받은 후, 새로운 weight가 계산되고, 다음 iteration을 위해 모든 nodes들에게 복사된다. (gets replicated across all the nodes for next iteration) 

parallel computation과 inter-worker communication의 interleaving을 가능하게 하는 몇가지 방법이 있다. 

#### 1. BSP(Bulk Synchronous Parallel)

parallel programs에서는 worker machine들 간의 exchange program이 요구되는데, MapReduce의 경우에는 Map worker들이 만든 key-value pair를 가지고, Reduce worker에게 해당 key값을 가진 모든 pair들을 transmit한다. key를 두개의 다른 Reducer에게 보내는것과같은 에러가 발생하면 안된다. operational correctness가 보장되어야하는데, parallel programming에서는 BSP를 통해 이를 보장한다. BSP는 computation이 inter- machine communication방식에 intertwine되어있는 방식이다. BSP 방식을 따르는 parallel program들은 computation phase와 communication phase(a.k.a synchronization)사이에서 왔다갔다(alternate)한다.

BSP방식은 다음 그림과 같이 computation과 communication phase사이의 clean separation을 형성한다. BSP 방식에서는 worker machine들에게 다음 synchronization에 도달하기 전까지는 각 machine의 computation phase가 보이지 않는다. 

 <img src="C:\SJL\스터디_분산ML_system\figures\BSP.PNG" alt="BSP" style="zoom: 50%;" />

BSP방식을 따르는 ML program들은 serializable하다. 즉, sequential ML program과 동일하다는 것을 의미한다. serializable BSP ML program들은 correctness가 guarantee되어있다. 

**장점**- serializable BSP ML program은 적확한(correct) solution output이 보장되어 있다. 그래서 operation-centric program이나 ML program에서 bridging model로 주로 사용된다.

consistency를 보장하는 가장 심플한 model이다. 각각의 computation과 communication phase간의 synchronization을 통해서 consistency를 보장한다.

**단점**- 느리다. BSP는 iteration throughput이 낮다고 표현하는데, 이것은 P개의 machine들이 P-fold increase in throughput을 확보하지 못한다는 것이다. Every synchronization barrier에서 먼저 완료된 worker은 모든 다른 worker들이 완료될때까지 기다려야 한다. 이런 문제는 몇 worker들이 progress가 늦은 경우에 overhead를 발생 시킬 수 있다. 특히 예측할 수 없는 현실적인 문제(temperature fluctuation in datacenter, network congestion, background tasks, etc)로 인해 특정 machine이 cluster내의 나머지 machine들보다 느려서 well balanced workloads가 확보되었어도 program 전체의 효율이 slowest machine과 match되도록 떨어지는 문제가 발생한다. Machine들간의 communication이 instantaneous하지 않기때문에 synchronization 자체가 시간을 많이 소모할 수 있다. 



#### 2. SSP(Stale Synchronous Parallel)

Bounded asynchronous 방식으로 지정된 제한(threshold)까지만 asynchronous 방식으로 진행되는 SSP(stale synchronous parallel)라는 방식인데 (SSP는 BSP가 더 포괄적이게 개선된 버젼으로 생각하면 된다.) 

programming interface는 BSP와 비슷하다. 동작 방식은 다음과 같다:

P개의 parallel worker들이 update과 aggregation과 같은 ML computation을 iterative하게 수행하는데, 각 iteration의 끝에서 SSP worker가 본인의 일은 끝났다고 signal을 보인다. 이때 BSP라면 sychronization barrier가 enact되어서 inter communication이 수행되겠지만, SSP에서는 이 barrier를 enact하는것 대신에 worker들을 각자 상황에 따라서 stop하거나 proceed하도록 허락한다. SSP will stop a worker if it is more than s iterations ahead of any other worker, where s=staleness threshold.

특정 횟수의 iteration 동안에는 더 빠른 worker들이 먼저 진행나아갈 수 있도록 허용하여 synchronization overhead를 완화한다. 이 특정 횟수를 넘기면, worker들 모두 쉬어야한다. Worker들이 data의 cached version으로 operate하고 각 작업 사이클(task cycle)의 끝에서 변경점들을 commit하기 때문에, 다른 worker들이 오래된(stale) data로 operate하게 된다. 

(Every worker machine keeps an  iteration counter t, and a local view of the model parameters A. SSP worker machines “commit” their updates Δ, and then invoke  a “clock()” function that: 

① signals that their iteration has ended,  

② increments their iteration counter t, and 

③ informs the SSP  system to start communicating Δ to other machines, so they can update their local views of A. 

This clock() is analogous to BSP’s  synchronization barrier, but is different in that updates from one  worker do not need to be immediately communicated to other  workers—as a consequence, workers may proceed even if they  have only received a partial subset of the updates. This means that the local views of A can become stale, if some updates have not been received yet.)

<img src="C:\SJL\스터디_분산ML_system\figures\SSP.PNG" alt="SSP" style="zoom: 50%;" />

SSP를 구현할때에 다음과 같은 bounded staleness condition들이 존재한다:

- bounded clock difference
- timestamped updates
- model state guarantees
- read-my-writes

**장점**- SSP는 any pair of workers사이의 maximum staleness를 threshold s로 제한하기때문에, data parallel 또는 model parallel방식에서 모두 strong theoretical model convergence가 guarantee된다.

**단점**- staleness가 너무 높아지면(slow down하는 machine의 비중이 너무 커지는 경우 발생), convergence rate이 빠르게 deteriorate한다. 

SSP 방식으로 운영되는 ML program(data parallel & model parallel 방식 모두)은 다음과 같은 두 개의 complementary theorem을 기반으로 near-ideal convergence progress per iteration을 확보할 수 있다.



##### SSP data parallel convergence theorem

BSP와 동일하게 correctness가 보장된다. 그러나 현실에서 적용할때에는 staleness와 asynchrony를 최소로 유지하는것이 매우 중요하다. complex production environment에서는 other tasks나 user등으로 인해 machine이 temporarily slow down할 수 있고 maximum staleness와 staleness variance가 너무 커져버리는 문제가 발생할 수 있다.

##### SSP model parallel asymptotic consistency

global view of parameter A가 결국 converge될것이고, stale local worker view of parameter또한 global view A로 converge될것이다라는 것을 말한다. 그리고 이렇게 converge된 값이 optimal solution이 될것이다라는 것을 이야기한다.



#### 3. ASP(Approximate Synchronous Parallel)

ASP는 SSP와는 반대로 parameter가 얼마나 inaccurate될 수 있는지를 제한하는 방식이다. (Parameter가 얼마나 (inaccurate)부정확해질 수 있는지를 제한한다. 이 점은 parameter가 얼마나 stale해지는지 제한하는 SSP와는 반대이다.) 이 방식에서는 만약 aggregated update가 중요한 수준이 아니라면, synchronization을 무한으로 연기할 수도 있다. 

**장점**- 축적된 update가 insignificant할때에 server가 synchronization을 무기한으로 연기할 수 있다.

**단점**- 어떤 parameter를 선택해야 update가 significant한지 아닌지를 구분하기가 어렵다.  



#### 4. Asynchronous parallel 

BSP와는 다르게 worker machine이 다른 machine들을 기다려주지 않는다. 각 iteration마다 model information을 communicate한다. Asynchronous execution은 보통 near-ideal P-fold increase in iteration throughput을 확보하지만, convergence progress per iteration은 감소한다. 이 방식에서는 machine들이 서로를 기다려주지 않기때문에 공유되는 model information이 delay되거나 stale되어서 computation에 error을 발생시키는 문제가 발생한다. 이 error를 제한하기위해 delays는 정교하게 bound되어야한다. 

<img src="C:\SJL\스터디_분산ML_system\figures\ASP.PNG" alt="Asynchronous" style="zoom: 50%;" />

**장점:** 속도가 빠르다. worker들이 기다림 없이 병렬로 communicate할 수 있다. 이 방식으로 가장 빠른 speedup을 얻을 수 있다는 것이 장점이다.

**단점:** staleness, incorrect result (due to risk that one machine could end up many iterations slower than the others, leading to unrecoverable error in ML programs).

Model convergence가 느리게 확보될수있는 risk가 있다. Model이 아얘 incorrect하게 develop될 수도 있다. BSP나 SSP와는 다르게 error가 delay와 함께 커질 수 있다. model이 느리게 converge하거나 BSP, SSP와는 다르게 error가 delay와 함께 커져서 model이 부정확하게(incorrectly) develop될 수도 있다는것이다. 

BAP(Barrierless Asynchronous Parallel)/ TAP(Total Asynchronous Parallel)로 세분화 할 수 있다. 이 방식에서는 기다림 없이 worker machine들이 바로 서로 병렬로 communicate한다. 



#### 5. Layered SGD

추가적으로, 최근 2019에는 asynchronous와 synchronous의 장단점을 적절하게 조절하기위해 "LSGD(layered SGD)"이라는 algorithm을 만들어서 decentralized synchronous SGD방식을 구현한 논문도 발표되었다. (paper link: https://arxiv.org/pdf/1906.05936.pdf) 

LSGD partitions computing resources into subgroups that each contain a communication layer (communicator) and a computation layer (worker). Each subgroup has centralized communication for parameter updates while communication between subgroups is handled by communicators. As a result, communication time is overlapped with I/O latency of workers. The efficiency of the algorithm is tested by training a deep network on the ImageNet classification task.

![LSGD_topology](C:\SJL\스터디_분산ML_system\figures\LSGD_topology.png)



# Distributed Machine Learning Systems/Frameworks

Cluster를 통해 large volume의 data를 process하는 문제는 machine learning분야 만이 아니라 distributed system과 database research 분야에서도 오랫동안 연구되어 왔다. 그 결과 Apache Spark와 같은 general purpose distributed platform이 distributed system의 현실적인 구현 방법으로 활용되었고, MLlib와 같은 최적화된 library가 제공되고 있다. 

General purpose와는 반대의 방향에는 purpose-built machine learning libraries가 있는데, 기존에는 single machine에서 동작하도록 design되었으나, 점점 더 distributed setting에서 실행될 수 있도록 개선되고 있다. 예를 들어, Keras는 Google의 TensorFlow와 Microsoft의 CNTK에서 실행 될 수 있도록 backends를 받았다. Nvidia 또한 그들의 machine learning stack을 더 발전시켜서 Collective Communication Library(NCCL)을 구현했는데, 기존에 동일 node에서 multiple GPU를 지원하는 것이였지만, version 2 부터는 multiple nodes에서 실행 될 수 있도록 발전 시켰다. 

아얘 처음부터 distributed machine learning을 위해 설계되고 만들어진 specific algorithm과 operational model이 distributed ML ecosystem의 중심에 있다. e.g., Distributed Ensemble Learning, Parallel Synchronous Stochastic Gradient Descent (SGD), 또는 Parameter Servers. 원래 대부분의 system들은 user 또는 on-premise로 운용(operate)되도록 의도되어왔으나, 점점 더 많고 다양한 machine learning services가 cloud delivery model을 통해 공급되고 있다. 이들은 established distributed machine learning system을 중심에 두고 surrounding platform으로 인해 개선되고있으며 해당 기술/technology가 data scientist나 결정권자들에게 더 쉽게 사용할 수 있도록 만들어가고 있다.

![DistributedML_Ecosystem](C:\SJL\스터디_분산ML_system\figures\DistributedML_Ecosystem.PNG)



## General Purpose Distributed Computing Frameworks

distributed system은 하나의 값비싼 large server보다는 다수의 commodity servers(each with relatively small storage capacity and computing power)를 활용하는 방식에 의존한다. 이 방식은 충분히 fault tolerance를 software에 build하는 방식이 확보되어 있는 한, 비싼 specialized hardware을 사용하는 것보다 훨씬 더 affordable하다는 것으로 확인된다. 그리고 scale-out model에서는 각 node의 I/O subsystem이 모두 포함되기 때문에, 소수의 more powerful machine을 사용할때와 비교해보면 결국 더 높은 aggregate I/O bandwidth를 제공한다. 이 특성은 data ingestion이 매우 중요한 data-intensive application에서 매우 큰 도움이 된다. 

### storage

현존하는 framework들의 storage layer는 GFS나 또는 이와 비슷한 implementation들에 기반되어있다. 

#### Google File System & Hadoop

GFS는 Google이 소유하고 사용하는 시스템으로 모든 Big Data storage 요구사항을 처리하기위해 사용된다. Block-based model인 GFS는 cluster로 upload된 data를 chunks로 나누어서 chunk servers로 distribute한다. chunk들은 machine failure발생시 사용할 수 없게되는 상황을 대비해서 미리 보호하기위해 replicate된다. User는 master를 통해서 chunk server들에 있는 data를  access할 수 있다. master는 name node 역할을 수행하는데 file의 모든 chunk의 위치를 알려준다. Hadoop이 GFS architecture을 adopt했고, 지금은 Apache Foundation이 maintain하고 있다. Hadoop Distributed File System (HDFS)은 GFS design과 거의 동일하고 이름(nomenclature)만 다르며, storage layer 역할을 수행한다.

(**GFS vs. HDFS:** Google File System은 Google이 만들고 소유하고있는 distributed file system이다. 이 시스템에서 사용할 수 있는 framework으로 Google이 만든것이 Mapreduce이다. Hadoop Mapreduce는 Google의 Mapreduce를 기반으로 만들어졌고 HDFS와 Mapreduce는 Apache가 진행하는 Hadoop project에 포함되어있다. *비교 논문: https://citeseerx.ist.psu.edu/viewdoc/download?doi=10.1.1.454.4159&rep=rep1&type=pdf*)

### compute

(storage와는 다르게) compute을 위한 frameworks는 다양하게 존재한다. 여러 framework들은 자원들(resources)을 compute하는 작업들을 여러 features 및 trade-offs와 함께 scheduling 또는 distributing하기 위해 서로 경쟁한다.

#### MapReduce

Distributed setting에서 big data를 처리(process)하기 위해 Google이 개발한 framework이다. Functional programming concept으로부터 아이디어를 빌려오고, multiple phase로 구성된 architecture를 가지고 있다. 

먼저 "map phase"동안, 모든 data는 tuple(a.k.a "key-value pairs")로 split된다. 이것은 functional programming에서 second-order function을 하나의 set과 mapping하는 것과 견주어볼 수 있다. "map phase"에서는 두 가지 다른 value들 사이에서 function을 mapping할때에 data dependencies가 없기때문에, 이 phase가 fully parallel하게 실행될 수 있다. 

그 다음, "shuffle phase"에서는 tuples가 node들 사이에서 exchange되고 전달된다(passed on). 보통 aggregation은 data dependencies를 가지고 있고, 정확도를 위해 동일한 key에서 기인한 모든 tuple들은 동일한 node로 처리되는것이 보장되어야하기때문에 필수적이다. 

그 다음 순서인 "reduce phase"에서는 각  key당 single output value를 생성하기위해 tuples에 **aggregation**을 실행시킨다. 이것은 functional programming의 fold operation과 비슷하다. fold operation은 single result value를 생성하는 second-order function을 사용하여 collection을 하나 roll up 한다. 그러나 fold의 구성을 보면, 각 fold step이 바로 전 step에 의존하기 때문에 fold는 parallelized되지 못한다. "Reduce phase"에서 parallelism을 가능하게하기 위해서는 data를 shuffling하고 key를 reducing 하는것이 필요하다. 

##### MapReduce진행 방식(Hadoop)

User는 Map & Reduce function만 쓰면 된다. (단, 매번 Map-Reduce operation이 실행될때에 disk I/O가 수행되어야함. 나중에 Spark가 개발되면서 disk I/O operation이 더 빠른 in-memory operation으로 대체됨.)

1. Map: mapper function pass thru data, stored in disk of each node, and increment word count in output dictionary (get executed independently on each distributed box)
2. Shuffle: Hadoop automatically moves data across LAN network (so same keys are grouped together in one box)
3. Reduce: function which will consume the dictionary and add up the values with same keys (compute the total count)



<img src="C:\SJL\스터디_분산ML_system\figures\MapReduce_Map.PNG" alt="MapReduce_diagram" style="zoom:50%;" />

MapReduce의 가장 큰 장점은 이 framework을 통해서  data dependencies없이 그리고 전부 parallel(병렬)방식으로 동일 phase의 작업(tasks)을 수행 하는 동시에 아주 많은 machine들에게 data가 distribute될 수 있다는 것이다. 동일 machine들은 GFS (또는 비슷한 타 storage cluster)의 node가 될수 있어서 data를 program으로 이동시키는 것대신에 program을 data로 옮기고 data locality와 performance를 개선하는 방법이있다. Program은 wire선상에서 보통 several orders of magnitude 더 작기때문에 전달하기에 매우 효율적이다. 

그리고 scale-out 방식의 관점으로 보면, MapReduce는 실패한 작업들(tasks)을 healthy node들에게 rescheduling하는 방식과 heartbeat message를 사용해서 worker node들의 health를 monitoring하면서 software에 fault-tolerance를 구현한다. 보통 하나의 작업(task)의 granularity는 input data set의 single block의 크기와 동일하기때문에 node failure은 전체 application의 작은 일부분만 영향을 끼친다. 전체 시스템은 손쉽게 recover할 수 있다. 몇가지 machine learning algorithm들을 MapReduce framework에 map해서  multicore machines를 위한 parallelism을 활용한 연구/논문도 있다 (Chu et al[36])



##### MapReduce & BSP(Bulk-Synchronous Processing)

MapReduce는 BSP paradigm과 비슷하다. 단, 작은 차이점이 있다. MapReduce framework은 map phase에서 worker nodes사이의 communication을 허용하지 않는다. 대신에 map phase와 reduce phase사이에 있는 shuffle phase에서 cross-communication을 허용한다. synchronization barrier를 줄이고 parallelism을 증가시키기 위함이다. 지난 연구 논문들을 참고해보면, 모든 BSP program들은 MapReduce program으로 변환될 수 있다는것이 확인되었고(Goodrich et al[59]), MapReduce application들을 BSP paradigm으로 model해서 BSP의 theoretical correctness와 MapReduce의 효율적인 execution을 통합적으로 장점으로 활용할 수 있다는것이 확인되었다(Pace[116]).

MapReduce는 Google의 framework인데, 이것의 architecture은 open source Hadoop framework에서 recreate되었다. Hadoop은 HDFS를 사용하고, MapReduce는 GFS를 사용한다. 그러나 이들의 전반적인 architecture는 동일하다. Strict한 topology를 가진 MapReduce에서 더 flexible한 구조를 가진 Forests나 generic Directed Acyclic Graphs (DAGs)로 변화하여 advanced variants들이 생겨났다.

#### Apache Spark

Apache Spark는 원래 data processing을 위해 만들어졌다. machine learning 기능이 나중에 탑재된 도구이다.  MapReduce와 Hadoop은 실행의 모든 phase에서 distributed file system에 많이 의존한다. Iterative workload가 동일한 data를 반복적으로 access하는데에 안전성을 보장해주기 위해 storage layer에 intermediate results까지도 저장한다. Machine learning algorithm에서 사용되는 linear algebra의 transformation은 보통 매우 강한 iterative nature를 가지고 있다. Map과 reduce operation paradigm은 iterative 작업의 data flow를 support하기에 적합하지 않다. 그래서 이런 문제에 대응하기 위해 Apache Spark가 만들어졌다.

Apache Spark는 **in-memory distributed data processing** 도구이다. ("in-memory"는 "runs on RAM"을 의미한다.) Open source "unified analytics engine for large-scale data processing"이다. rich한 ecosystem 가지고있다. specifically machine learning을 위해 만들어진 요소는 MLlib이다.

note: parallel vs. distributed 차이점 - 이 두개의 terms can be used interchangeably, BUT! how they access memory is different. (parallel computing shares memory while distributed computing have their own memory)

![parallel_vs_distributed](C:\SJL\스터디_분산ML_system\figures\parallel_vs_distributed.PNG)

Spark는 transformations의 directed acyclic graph를 실행하고(e.g., mapping) actions(e.g., reductions)를 memory에서 fully 실행하는 것이 가능하다. Spark의 구조때문에, complex workloads를 처리하는데에 **MapReduce보다 훨씬 더 빠르다.** 예를 들어서 만약 두개의 map phase들이 연달아서 필요한 경우가있다면, 두개의 MapReduce 작업이 필요해지고 이 둘은 모든 intermediate data를 disk에 써야한다. MapReduce대신 Spark를 사용하게되면, 모든 data를 memory에 유지할 수 있어서 disk로부터 읽는 expensive한 작업을 하지 않아도 된다. 

Spark의 data structure은 Resilient Distributed Dataset(RDD)으로 불린다. (RDD = a fault tolerant immutable collection of distributed datasets stored in main memory) 

RDD는 다음 두가지 operations를 support한다:

- tranformations : create new data set from an existing one. All transformations are lazy (i.e. computed only when an 'action' requires a result)
  - Map
  - ReduceByKey
  - Filter
- actions : return a value after running computation on data set
  - Reduce
  - Collect
  - SaveAsTextFile

예시:

ReduceByKey를 사용해서 distributed RDD of 3 partitions에서 occurrence of keys를 count하는 작업이다.

```python
# Simple ReduceByKey example in python sourced from:
# https://backtobazics.com/big-data/spark/apache-spark-reducebykey-example/
# creating PairRDD x with key value pairs
x = sc.parallelize([("a", 1), ("b", 1), ("a", 1), ("a", 1),
                    ("b", 1), ("b", 1), ("b", 1), ("b", 1)], 3)
 
# Applying reduceByKey operation on x
y = x.reduceByKey(lambda accum, n: accum + n)
print(y.collect())
# [('b', 5), ('a', 3)]
```



RDD dataset은 read-only이고 새로운 instance는 이미 존재하는 RDDs를 transform하거나 disk에 이미 저장된 data로부터 만들어 질 수 있다. RDD의 resilient 부분/역할은 data가 lost 되었을때에 확인해볼 수 있다. 각 RDD는 lineage graph가 주어지는데, 이것은 RDD에 어떤 transformation이 실행되었다는 것을 알려준다. 이 lineage graph는 만약 어떤 data가 lost된다면 Spark가 lineage graph를 통해서 RDD가 따른 길을 trace하고 lost data를 recalculate한다. 이 lineage graph가 cycle을 포함하지않는 Directed Acyclic Graph여야하는것이 매우 중요하다. 왜냐면 Spark가 infinite loop에 빠져버려서 RDD를 다시 생성할 수 없게 되어버리기 때문이다. node failure로 인해 발생한 data loss때문에 수행된 re-computation은 ripple effect를 일으킬 수 있다. 

Spark는 checkpointing을 허용해서 extensive re-computation을 방지한다. Checkpoint는 explicitly request되어야하고 intermediate state을 materialize하면서 RDD lineage graph를 truncate한다. TR-Spark(Microsoft가 만든 Spark version)와 같은 system들은 checkpoint를 생성하는 것이 자동화 되어있어서 interruption of execution이 norm으로 여겨지는 transient한 resources를 사용할 때에도 Spark가 운용될 수 있다.

<img src="C:\SJL\스터디_분산ML_system\figures\Spark_RDD.png" alt="Spark_RDD" style="zoom:50%;" />

Apache Spark는 MLlib를 포함한다. MLlib는 classification, regression, decision trees, clustering 그리고 topic modeling을 위해 여러가지 ML algorithm을 scalable machine learning library이다. MLlib는 ML workflow를 만들기위해서나, feature transformations, hyperparameter tuning을 위해서 여러가지 utilities를 제공한다. MLlib가 Spark의 API를 사용하기때문에 Spark의 scale-out과 failure resilience feature들을 바로 사용할 수 있게된다. MLlib는 Scala linear algebra package인 Breeze와 (Breeze는 최적화를 위해 netlib-java를 활용함) BLAS와 LAPACK와 같이 high performance computing에 사용되는 libraries을 위한 bridge에 의존한다. 

abstracted parallelization - 알아서 분산 processing을 처리한다. (data loading후 어떤 machine에 얼만큼 어떻게 배분할것인인지 등등을 알아서 처리한다)

Apache Spark core build:

![Apache_Spark](C:\SJL\스터디_분산ML_system\figures\ApacheSpark_ecosystem.PNG)



libraries- Spark SQL, Spark streaming, MLlib, GraphX, Spark-NLP, 등이 있음.

runtime - 그냥 PC에서 run할수 있고, cloud computing instance에서도 가능하다. 그리고 Yarn과 Kubernetes 등을 통해서 runtime을 leverage할 수 있다. (원래 처음 Spark가 나왔을때에, Hadoop(Yarn기반) 을 바탕으로 실행되는것에 최적화 되어있었다. ) 지금은 Mesos, Kubernetes도 가능하다.

Mesos를 통해서 (Mesos= a distributed system kernel) Spark는 매 iteration후, intermediate data set을 cache할 수 있다. Since output of each iteration is stored in RDD, only 1 disk read and write operation is required to complete all iterations of SGD.

##### How Spark works

위 그림과 같이 Spark cluster는 a master and multiple worker로 구성되어있다. 

<img src="C:\SJL\스터디_분산ML_system\figures\spark_architecture.PNG" alt="spark_architecture" style="zoom:67%;" />

A master - responsible for negotiating resource requests made by the Spark driver program corresponding to the submitted Spark application. 

Workers - processes hold Spark executors (each of which is a JVM instance) that are responsible for executing Spark tasks.

There are two kinds of operations: transformations and actions. A transformation (e.g., map, filter, join) performs an operation on a RDD and produces a new RDD. An action (e.g., collect, count) triggers a job in Spark. A typical Spark job performs a couple of transformations on a sequence of RDDs and then applies an action to the latest RDD in the lineage of the whole computation. A Spark application runs multiple jobs in sequence or in parallel.

Spark의 driver contains two scheduler components: 

- DAG scheduler : responsible for stage-oriented scheduling
- Task scheduler : responsible for submitting tasks produced by the DAG scheduler to the Spark executors

dependency relations: RDDs set간의 data dependency를 capture하기위해 두 가지 종류의 dependency relation을 설정한다.

- Narrow dependency

  Each partition of the parent RDD is used by at most one partition of the child RDD

  e.g., map, filter, and union transformations

- Shuffle dependency (a.k.a wide dependency)

  Multiple child partitions of RDD may depend on a single parent RDD partition. 

  e.g., groupby and join transformations 

  Defines the boundary of two connected stages

  Data exchange across executors only happens between two adjacent stages and the result of shuffled data from the previous stage constitutes the input of the next stage.

  

##### Lazy evaluation

lazy evaluation = transformation이 바로 실행되지 못하는 것. action이 trigger될때까지 execution이 시작되지 못한다.

Spark는 DAG를 통해서 어떤 operation들이 call되는지 record한다. Spark RDD를 transformation을 통해 build up하는 data로 생각할 수 있다.  transformation이 lazy하기때문에 data에 action을 불러서 언제든 operation을 execute할 수 있다. 그래서 lazy evaluation data는 필요해질때까지 load되지 않는다.

Spark가 lazy evaluation을 employ하는 이유:

- manageability 증가 - Spark program을 smaller operation으로 organize할 수 있음 (reduce the number of passes on data by grouping operations)
- computation을 감소시키고 속도 증가 - necessary values만 compute되기때문에, calculation overhead saved.
- complexities 감소 - operation에 소요되는 time과 space의 complexities 모두 대응할 수 있다. every operation을 execute하지 않기때문에 time이 save되고, action은 data가 required될때만 trigger되기때문에 overhead가 감소된다.
- optimization (최적화) - number of queries 감소

Computation DAG가 다 만들어질때까지 기다리고, 동일한 DAG안에서 transformation을 포함한 execution이 DAG내의 action으로 trigger될때까지 기다린다. (Spark waits until the whole compu)

dependency relation에 따라서 computation이 받는 영향이 달라진다 - narrow dependency는 efficient execution을 가능하게하지만, wide dependency는pipeline을 disrupt하거나 communication intensive shuffle operation을 요구하기때문에 bottleneck을 introduce할 수 있다.

예시:

```python
lines = sc.textFile("data.txt")
lineLengths = lines.map(lambda s: len(s))
totalLength = lineLengths.reduce(lambda a, b: a + b)```
```

위와 같은 code에서는 *lineLengths*가 바로 compute되지 않는다. 'reduce'가 실행되어야만 Spark가 computation을 여러 machine들간에 task로 쪼개어 실행하고 total length를 찾도록 한다.



##### Fault tolerance

Spark는 DAG를 통해서 lineage of operations on RDDs를 track한다.

shuffle/wide dependency를 위해서는, intermediate records from one stage are materialized on the machine holding parent partitions. 이 intermediate data를 사용해서 failure recovery절차를 단순화한다. (만약 task하나가 fail한다면, 해당 stage의 parents가 계속 accessible한동안 task를 retry해볼 수 있다. 만약 stage가 accessible하지 않다면, missing partition은 병렬로 re-compute될 수 있다.)

Driver에서의 scheduler failure에 대한 tolerance는 없지만, schedule의 metadata를 replicate해서 failure에 대응할 수 있다.

Driver에서 task scheduler는 running tasks의 상태를 monitor하고 failed tasks를 retry한다. Cluster내에서 느린 straggler들이 발생한다면 Spark job의 progress를 drag할 수 있다. Cluster이 커질수록 straggler의 악영향이 더 커질 수 있지만, task scheduler가 straggling tasks의 speculative relaunch를 사용해서 straggler의 악영향을 감소시킬 수 있다.



##### Shortcomings

- Spark는 원래 machine learning을 위해 design된것이 아니고, 추후에 MLlib를 통해 machine learning에 맞도록 변경된것이다. Basic machine learning setup에서는 Spark는 driver에 model parameters를 store하고 worker들이 driver와 communicate해서 각 iteration이후 parameter를 update한다. Large scale machine learning deployments를 위해서는 driver node에 model의 parameter들이 모두 fit하지 못할 수 있기때문에 parameter들을 RDD로 maintain해야할 수 있다. RDD로 maintain하기위해서는 큰 overhead를 발생시킬수 있다 (b/c RDD will need to be created in each iteration to hold the updated model parameters) Model update를 위해 machine들 간의 data shuffling이 필요하기때문에 scalability가 제한된다.

- parameter server abstraction이 없기때문에 Spark의 scalability가 제한된다. (especially when facing a machine learning task containing a large volume of model parameters)

- keeping the model as an RDD slows down the performance of Spark significantly -> 그래서 driver에서 model parameter를 update하고 maintain했다. 그럼에도 불구하고 Spark는 나머지 platform들(PMLS, MXNet, TensorFlow) 보다 더 느린 결과를 내었다. (Spark's performance falls below the other platforms for image classification task with single and two hidden layers)

  Spark의 computation speed는 RDD partition개수에 따라서도 많이 변동되었다. (significantly varies depending on the number of RDD partitions)

  computation is modeled as a directed acyclic graph (DAG) - each vertex denotes an RDD and each edge denotes an operation on RDD (On a DAG, an edge E from vertex A to vertex B implies that RDD B is a result of performing operation E on RDD A.)

Medium posting on using Apache Spark for deep learning : https://towardsdatascience.com/deep-learning-with-apache-spark-part-1-6d397c16abd

Medium posting on introducting Spark: https://towardsdatascience.com/your-first-apache-spark-ml-model-d2bb82b599dd

**Important feature of Spark:** 

All **transformations** are lazy (lazy = do not compute results right away) Instead Spark remembers transformation applied to some base dataset. 

transformations are only computed when an action requires a result to be returned to the driver program. By default, each transformed RDD may be recomputed each time you run an action on it. However, you may also *persist* an RDD in memory using the persist (or cache) method, in which case Spark will keep the elements around on the cluster for much faster access the next time you query it. There is also support for persisting RDDs on disk or replicated across multiple nodes.



## Natively Distributed Machine Learning System

### Distributed Ensemble Learning

TensorFlow, MXNet, 그리고 PyTorch와 같은 machine learning framework으로 distributed ensemble 방식을 구현할 수 있다.

많은 generic framework들과 ML libraries는 single machine에서 빠르고 효과적인 성과를 만들어도 distributed training을 하기에는 제한된 support를 가지는 경우가 많다. 이런 framework들과 distribution을 구현하기 위해서는 available data의 subsets으로 separate model을 훈련시키는 것이다. prediction phase에서 이런 instance들이 standard ensemble model aggregation을 통해 통합될 수 있다. 

이 ensemble 방식을 사용하는 전략은 어떤 특정 library에 의존하지 않는다. 이 방식은 이미 존재하는 distribution frameworks(MapReduce와 같은)을 사용해서 orchestrate될 수 있다. 훈련방식은 individual model을 각각 독립적인 machine에서 병렬로 훈련시키며 진행된다. Training이 시작되면 orchestration이나 communication이 필요하지 않다. 즉,  m개의 data subset으로 m개의 machine을 훈련시키기위해서는 m개의 다른 model들이 필요하다. 각 model 은 separate parameter나 또는 separate algorithm을 사용할 수 있다. prediction time에서 모든 훈련된 model들이 새로운 data를 받아 run하고 각자의 output을 aggregate한다. 그리고 그 다음 필요시 다시 distribute될 수 있다.

하나의 큰 단점은 이 방식에서는 훈련 data의 적절한 subdivision이 필수적이고 critical하다는 것이다. 만약에 부적절한 subdivision으로 인해 training data sets에 큰 bias가 존재한다면, 그 instance들은 ensemble의 output에 그대로 큰 bias를 일으킬 수 있다. IID (independently & identically distributed)한 data distribution을 보장하는 것이 매우 중요하다. (만약에 problem의 data가 inherently distributed되어있다면 적절한 data distribution이 이루어졌는지 보장이 필요하다)

### Parallel Synchronous Stochastic Gradient Descent

Synchronized parallelism은 가장 program하기 간단하고 명확한 방식이다. MPI(Message Passing Interface = MPI is a programming model ubiquitously present in any supercomputer to communicate processes executed in different servers)와 같이 현재 존재하는 libraries는 이 목적을 위해 다시 사용될 수 있다. 대부분의 approach들은 AllReduce와 같은 operation에 의존하는데, node들이 tree와 같은 topology로 arrange된다. 먼저 각 node가 local gradient value를 계산한 뒤, 이 gradient값을 children으로 부터 받은 values와 함께 accumulate하고 이것을 parent로 보낸다 (reduce phase). 결국 root node는 global sum을 얻어서 이것을 leaf node들에게 broadcast한다 (broadcast phase). 그러면 각 node는 전달받은 global gradient에 대해 local model을 update한다. 

#### Baidu AllReduce

Baidu AllReduce는 high performance computing technology를 사용해서 SGD model들을 반복적으로 separate mini batches of training data에 훈련시켰다. AllReduce is used to apply each of the workers gradient onto the last common model state after each operation and then propagate the result of each workers' training iteration before continuing to the next.

Baidu에는 "Ring AllReduce"라고 불리는 방식이 있는데 communication을 줄이고 더 최적화한다. Ring 형태로 machines cluster를 형성해서 (각 node가 단 2개의 neighbor를 가지도록)reduction operation을 cascade해서 모든 bandwidth를 최적으로 사용하는 것이 가능하다. 여기서 bottleneck은 neighboring nodes사이의 가장 높은 latency이다.

Baidu는 deep learning networks를 훈련하는데에 linear speedup을 선언하지만, 실제로 demonstrate된 case는 작은 cluster뿐이다. (cluster of 5 nodes, each having multiple GPUs that communicate with each other though the same system) 이 방식은 ring에 있는 node가 miss되지 않기때문에 기본적으로 fault tolerance가 없다. 이점은 효율성을 잃는 대신에 redundancy를 사용해서 counteract/보완?될 수 있다. 이 방식의 scalability는 모든 nodes들이 available한 확률에 따라 지정된다. 이 확률은 Big Data를 달루기위해 large number of commodity machines과 networking을 사용할때에는 낮아질 수 있다. Baidu의 system은 built-in Parameter Server based approach로 TwnsorFlow에 integrate될 수 있다. 



#### Horovod

Horovod는 Uber Engineering이 만든 internal ML-as-a-service platform인 Michelangelo에서 하나의 component로 처음 만들어졌다. Michelangelo는 machine learning models를 더 큰 scale에서 더 쉽게 build & deploy할 수 있도록 하는 서비스 platform이고, 여기에서 Horovod는 open source로 TensorFlow, PyTorch, 그리고 MXNet을 위한 training framework로 사용되었다. ring-allreduce를 단지 code 몇 줄만을 수정해서 Distributed Deep Learning을 빠르고 쉽게 진행 할 수 있도록 만드는것이 목표였다. (Horovod는 Apache 2.0 license아래에 가능하다.)

Horovod는 TensorFlow, Keras, Pytorch 그리고 Apache MXNet을 위한 distributed deep learning training framework이다. Horovod도 **MPI를 사용해서 distributed 방식으로 execute된 process들을 communicate한다.** (MPI는 다른여러 server들에서 실행된 processes들과 communicate하기 위해 supercomputer에 언제 어디서든 존재하는 programming model이다.) 그리고 **Model training속도를 높이기 위해 주로 data parallelism을 사용한다.** (data parallelism: That is to say, all workers train on different data, all workers have the same copy of the model, and Neural network gradients are exchanged.) 

Horovod를 통한 data parallel distributed training 진행방식은 다음과 같이 매우 심플하다:

1. run multiple copies of training script and each copy:
   - reads a chunk of the data
   - runs it through the model
   - computes model updates(gradients)
2. average gradients among those multiple copies
3. update the model
4. repeat from step 1

Horovod는 Baidu의 reduce algorithm을 활용한다 - average gradients and communicate those gradients to all nodes (위의 step2 and 3) that follows the ring-allreduce decentralized scheme. 



##### Ring-Allreduce (visual intuition)

ring allreduce는 two phases로 구성되어있다(first share-reduce phase, and then share-only phase) 더 효율이 높은 방법으로 각nodes단의 communication results를 모두ㅡProcess nodes간의 연결 모양이 ring을 연상하게한다. 

in ring-reduce, the gradient is divided into consecutive blocks at each node, and simultaneously, each block is updated using the previous node and also provides an update to the next node, making a ring pattern. This method is popularized by tools like Horovod.

*<참고 link https://towardsdatascience.com/visual-intuition-on-ring-allreduce-for-distributed-deep-learning-d1f34b4911da>*

https://roman-kazinnik.medium.com/machine-learning-distributed-ring-reduce-vs-all-reduce-cb8e97ade42e

https://brunch.co.kr/@chris-song/96 (explanation in korean with diagram)



다음 그림과 같이 ring-allreduce algorithm을 통해서 worker node들이 gradient들의 average를 구하고 parameter server를 통한 centralized scheme의 필요 없이 이들을 모든 node들에게 disperse한다. 

![Horovod](C:\SJL\스터디_분산ML_system\figures\ring_allreduce.png)

위와 같은 ring-allreduce algorithm에서는 N개의 node들이 각각 두 개의 peers와 2*(N-1)번 communicate한다. 이 communication을 하는 동안, 각 node가 chunks of data buffer를 send & receive한다. 첫 N-1 iterations에서는, 받은 value들이 node의 buffer에 있는 values들에 더해진다. 두번째 N-1 iterations에서는, node의 buffer에 hold된 value들을 received value들이 replace한다. 이 algorithm은 bandwidth-optimal하다. 즉, buffer가 적당하게 크다면 사용가능한 network을 최적의 조건으로 활용할 수 있다.

Baidu와 다른점은 Horovod가 **GPU training에 더 높은 효율성을 활용하기 위해 server내의 GPU사이 data communication을 관리할 수 있는 NVIDIA의 Collective Communications Library (NCCL-2 library)을 사용한다**는 점이다. Horovod는 Baidu의 ring-allreduce implementation을 Nvidia의 NCCL-2로 대신한다. (NCCL-2란? Nvidia's library for collective communication that provides a highly optimized version of ring-allreduce across multiple machines) Single node에서 multiple GPU를 사용할 수 있도록 해서 server내의 GPU들간의 data communication을 manage한다. 



##### Horovod with TensorFlow

Horovod는 Baidu와 매우 비슷한데, AllReduce-based MPI training을 하나의 layer로 TensorFlow에 추가한다. 

TensorFlow model을 data-parallelizing으로 구현하는것이 비교적 심플하다.(only a few lines of code need to be added, wrapping the default Tensorflow training routine in a distributed AllReduce operation) 예를 들어, 128 GPU를 사용해서 Inception v4와 ResNet-101을 통해 benchmarking한 결과, 평균 GPU utilization은 대략 88%이며, 이 수준은 TensorFlow의 Parameter Server approach의 benchmark인 50%보다 높다. 그러나 Horovod는 fault tolerance가 없어서 scalability issue가 발생하는 위험을 가지고있다. 

Horovod를 사용할때의 장점은 model training script에 적용해야하는 변경사항들도 감소되어있어서 distributed training을 보다 쉽게 진행 할 수 있다는 것이다. 그리고 environment 설정이 대부분 자동으로 된다. Azure ML이 curated training environment를 제공해서 다양한 종류의 framework기반으로 training을 진행하는데에 유용하다. (TensorFlow와 Horovod가 preload되어 오기도 한다.)



##### Horovod Timeline

Uber에서 Horovod를 소개할때에 distributed system내 여러 server들의 operation timeline을 확인할 수 있는 "Horovod Timeline"이라고 불리는 high-level profiling tool도 함께 제공했다. 

Horovod Timeline을 사용해서 training job동안 각 time step에서 각 node가 무엇을 하고있는지 확인할 수 있다. 이 도구를 통해 bug를 찾고 performance issue들을 debug할 수 있다.

이 Horovod-focused profiling tool은 Chrome의 "about:tracing" trace event profiling viewer와 compatible하다. Users can enable timelines by setting a single environment variable and can view the profiling results in the browser through ```chrome://tracing```. 

![HorovodTimeline](C:\SJL\스터디_분산ML_system\figures\HorovodTimeline.png)



#### Caffe2

Caffe2도 AllReduce algorithm을 통해서 ML distribution을 구현한다. Caffe2는 Facebook으로인해 유지보수 되고있다. Single host에서 NCCL을 사용해서 GPU간의 collaboration을 관리할 수 있고, Facebook의 Gloo library를 기반으로 custom code를 사용해서 다른 interconnects에서부터 abstract away한다. Facebook은 더 좋은 bandwidth와 parallelism guarantee를 제공하는 Ring AllReduce를 사용한다. recursive halving과 doubling 또한 사용해서 divide-and-conquer 방식을 사용해서 더 나은 latency guarantee를 사용한다. Latency로 인해 limit되는 상황(buffer size가 작고 larger server counts가 있는 경우)에서 peformance를 향상시킨다. 



#### CNTK(Microsoft Cognitive Toolkit)

data parallel distribution의 multiple modes를 제공한다. 대부분 Ring AllReduce 전략을 사용한다. 동일하게 linear scalability와 fault-tolerance간의 tradeoff 특성을 가지고 있다. 이 library는 다음과 같은 두 가지 innovations를 제공한다:

-  1-bit stochastic gradient descent: Training gradient를 single bit per value로 quantize하는 SGD 구현방식이다. 이 방식은 distributed training을 할때에 필요한 communication을 큰 factor로 감소시킬 수있다.
-  block-momentum SGD: 먼저 training set을 m block들과 n split들로 나눈다. n개의 machines 각각 하나의 block에서 하나의 split을 훈련시킨다. 그리고나서 block을 위한 weights를 얻기위해 block내에서 모든 split을 위해 계산된 gradients의 평균을 구한다. 마지막으로 block-level momentum과 learning rate을 적용하는 동안 block updates를 global model에 merge한다.

Microsoft speech LSTM에 benchmarked되면, small number of GPUs를 위해 평균 speedup이 85%수준으로 확인된다. 그러나 scalability는 70% 미만으로 떨어진다. (LSTM 모델을 다른 ordinary DNN 모델과 매우 다르기 때문에 LSTM 모델을 기준으로 확인한 benchmark 수준은 다른 모델에 그대로 비교하기 애매함. )



### Parallel Asynchronous Stochastic Gradient Descent and Parameter Servers

Asynchronous 방식은 구현하고, runtime behavior를 trace하거나 debug하기가 더 복잡한 경우가 많다. 그러나 asynchronism은 frequent synchronization barriers가 없어서 높은 failure rates나 inconsistent performance와 관련된 문제들이 적다는 장점이 있다.

#### DistBelief 

Large-scale distributed ML방식 중에 Google로 인해 예전에 먼저 구현된것중 하나이다. GPU training의 limitation에 대응하기위해 만들어진것이 DistBelief이다. 이것은 data- and model- parallel training on tens of thousands of CPU cores를 지원할 수 있다. (나중에 GPU지원도 추가되었음) 81개의 machine을 사용해서 1.7billion parameter를 가진 huge model을 훈련시키는과정의 속도를 12x높혔다. 

효율적인 model-parallelism을 구하기 위해 DistBelief는 neural networks의 구조를 활용한다. 그리고 각각의 node가 input을 output으로 transform하는 operation을 구현하는 computation graph를 통해 model을 define한다. 각각의 machine이 computation graph의 nodes의 training part를 실행한다. 이 node들은 neural network의 subsets of multiple layers span한다. communication은 한 node의 output이 다른 machine으로 인해 훈련되는 다른 node의 input으로 사용되는 points에서만 필요하다.  

cluster에서 model을 partition하는 것은 투명하고(transparent) 구조적인 변경이 필요하지 않다. 그러나 model 의 architecture로 인해 partitioning의 효율성은 큰 영향을 받고 careful한 design이 필요하다. 예를 들어서 locally connected model은 제한된 cross partition communication으로 인해 model-parallelism에 더 알맞다. 반대로 fully connected model인 경우에는 더 많은 substantial cross partition dependencies가 있기 때문에 DistBelief를 통해서 efficient하게 distribute하기가 어렵다.

Model training을 더 parallelize하기위해서는 data parallelism이 model parallelism위에 추가적으로 적용된다. each set of model replicas가 parameter를 공유하도록 하기위해서 centralized sharded Parameter Server가 사용된다. DistBelif는 두가지 다른 data parallelism 방식을 지원한다. 이 가지 방식 모두 model replicas와 replica failure사이의 processing speed variance에 resilient하다.

- downpour stochastic gradient descent: inherently sequential SGD의 asynchronous한 alternative이다. model의 relica가 가장 최신 model parameter를 Parameter Server로 부터 every n_fetch step마다 fetch해온다. 그리고 model에 맞게 이 parameter들을 update한다. 그 다음, tracked parameter gradients를 Parameter Server에게 everu n_push step들 마다 push한다. 여기서 n_fetch와 n_push 값들은 communication overhead를 줄이기위해 증가될 수 있다. 이런 fetching과 pushing이 background process로 이루어지면서 훈련이 진행된다. 

  downpour SGD는 SGD보다 machine failures에 더 resilient 하다. 몇개의 model replica가 off-line이여도 training이 지속될 수 있도록 허용한다. 그러나 이렇게 parameter가 out of sync되는 경우를 허용하기때문에 optimization process가 덜 predictable하게 된다.

- distributed L-BGFS: model replicas들 간에 training work을 나누는 external coordinator process를 활용한다. 그리고 parameter들과 parameter server shards간의 operations도 활용한다. L-BGFS를 통해 훈련이 진행된다. (L-BGFS= limited-memory BFGS: an optimization algorithm in family of quasi-Newton methods that approximates) Broyden–Fletcher–Goldfarb–Shanno algorithm (BFGS) using a limited amount of computer memory.



#### DIANNE (Distributed Artificial Neural Networks)

필요한 computation을 실행하기 위해 Torch native backend를 사용하는 java-based distributed deep learning framework이다. modular OSGi-based distribution framework를 사용해서 infrastructure의 여러 다른 node들에서 deep learning system의 다른 components들을 실행할 수 있도록 한다. Neural network의 basic building block을 특정 node에 deploy할 수 있어서 model-parallelism이 가능하다. DIANNE방식은 또한 scale될 수 있는 basic learner, evaluator 그리고 parameter server components를 제공하고 downpour SGD를 DistBelief와 비슷하게 구현할 수 있도록 한다.



#### TensorFlow

open source machine learning framework이며 다양한 도구들을 포함해서 아주 풍부한 ecosystem을 가졌고 deep learning 작업에 더 적합하다. Apache Spark와는 다르게 처음부터 deep learning, machine learning을 위해 만들어진 도구이다.

Keras를 high level API로 사용한다.

Python, Java Script, Swift, Java, Go, 등 아주 많은 언어로 API를 제공한다.

tools: TF Probability, TF Agents(for reinforcement learning), TF Ranking, TF Text, TF Federated(for federated learning), 등등

distribution 전략: ML distribution이 보다 쉽게 진행 될 수 있도록만들어져 있다.

Researchers, practitioners, 등이 보다 쉽게사용할 수 있다. 분산 strategies들 간의 변경도 쉽게 이루어 질 수 있도록 하고 코드 수정이 조금만 되어도 빠르게 실행 할 수 있도록 준비되어 있다. "good performance out of the box"를 제공한다. 

DistBelief를 대체하기 위해 Google이 DistBelief의 진화 버전으로 만든것이다. 그래서 DistBelief에서 부터 computation graph와 parameter server를 가져온다. 또한, convolutional neural network을 훈련시키기위한 optimization과 같이 subsequent optimization to parameter server model도 적용되었고, consistency model 과 fault tolerance의 innovation또한 적용 되었다. DistBelief와는 다르게, TensorFlow는 open source software로 개방되었다. 

TensorFlow는 model algorithm과 distribute될 수 있는 dataflow graph와 같은 state를 모두 represent한다. state locality를 고려하는 것과 같이 TensorFlow는 다른 parallelization scheme들을 facilitate한다. dataflow graph의 abstraction level은 tensor의 mathematical operation이다. (n dimensional matrices를 사용하는 operation과 같은)이점은 individual layer의 level에서 abstract하는 DistBelief와는 반대이다. 그래서 TensorFlow에서 neural network layer의 새로운 type을 설정하는데에 custom code를 요구하지 않는다. 기본적인 math operation으로 만들어진 더 큰 model의 subgraph로 represent 될 수 있다. A TensorFlow model은 symbolic dataflow graph로 처음 define될 수 있다. 이 graph가 construct되면, 이것은 최적화되고 available한 hardware에서 실행된다. 이런 execution model은 TensorFlow에게 available한 device의 type에 맞게 최적화하고 실행하도록한다. GPU나 TPU와 함께 실행할때에는 TensorFlow가 이런 devices의 inherent한 asynchronicity와 intolerance 또는 sensitivity를 고려하되 model자체의 변화는 요구하지 않는다. TensorFlow를 통한 performance는 최적화하려는 neural network의 architecture의 영향이 크고, communication overhead가 매우 중요한 역할을 수행한다.

기존의 dataflow system과는 다르게 TensorFlow는 node들이 mutable state를 가지고있거나 update하는 computations를 represent하도록 한다. 

mutable buffer를 가진 *Variable*이 model parameter를 store하고 매 iteration에서 parameter들을 update한다. graphs안의 node들은 operations를 represent하는데, some operations are control flow operations. Values that flow along the directed edges in TensorFlow graph are Tensors(arbitrary dimensionality matrices) 하나의 operation이 one or more tensors를 가지고 result tensor를 만들어낼 수 있다. 또한, special edges는 control dependencies로 불리고, TensorFlow의 dataflow graph에 flowing data 없이도 더해질 수 있다.

In summary, TensorFlow is a dataflow system that offers mutable state and allows cyclic computation graph, and as such enables training a machine learning algorithm with parameter-server model.

TensorFlow는 다음과 같은 3가지 main components를 가지고있다:

##### Client

hold a session where a user can define computational graph to run. client가 session object를 통해서 Tensorflow graph의 evaluation을 request한다면, 이 request는 master service에게 전송된다.

##### Master

client의 request를 받은 master는 one or more workers로 job을 schedule하고 execution of computational graph를 coordinate한다. 

##### Worker

그러면 각 worker는 master로부터 온 request에 응답하고 computational graph에 execution of the kernels를 schedule한다. (dataflow executor in a worker dispatches the kernels to local devices and run the kernels in parallel when possible.) 

(note: kernel의 정의 - implementation of an operation on a particular device)

![tensorflow_software_layers](C:\SJL\스터디_분산ML_system\figures\TensorFlow_software_layers.PNG)

만약 computation에 multiple device들이 involve되어있다면, node placement라는 procedure이 수행된다. TensorFlow는 cost model을 통해 available device(CPU, GPU)에서 an operation을 수행하는데에 필요한 cost를 계산해서 suitable device에 operation을 배정한다. (subject to implicit or explicit device constraints in the graph) TensorFlow는 여러가지 communication protocol을 지원한다. (gRPC over TCP, RDMA over Converged Ethernet)

TensorFlow는 sub-graph execution을 지원한다. graph 또는 subgraph를 실행하는 single round를 "step"이라고 하는데, training application은 두가지 type의 jobs를 포함한다: parameter server job, worker job

TensorFlow에서 (PMLS에서와 같이)data parallel방식이 구현되면, training involved multiple tasks in a worker job training the same model on different mini-batches of data, updating shared parameters hosted in a one or more tasks in a ps(parameter server) job.

![tensorflow_training_diagram](C:\SJL\스터디_분산ML_system\figures\TensorFlow_btwngraph_rep_training.PNG)

위 그림에서 보이는 것과 같이 "between-graph" replicated training에서는 각 worker task를 위해 separate client가 존재한다 (보통 within the same process as the worker task) 각 client가 parameter들의 담긴 비슷한 graph를 build하고(pinned to ps), single copy of  computation intensive part of computational graph(pinned to local task in the worker job)을 build한다. 

예시:

compute gradient during each iteration of SGD algorithm

user는 between-graph replicated training에서 consistency model을 specify할 수 있다 - asynchronous vs. synchronous training

- asynchronous: each replica of graph has independent training loop that executes without coordination

- synchronous: all of replicas read the same values for current parameters, compute gradients in parallel and then apply them to a stateful accumulators which act as barriers for updating variables (stateful뜻? computer or program keeping track of state of interaction)

##### Fault tolerance

user가 save operation을 통한 checkpoint를 통해 fault tolerance mechanism을 설정할 수 있다. save operation은 tensors를 checkpoint file에 write하고, restore은 tensors를 checkpointing file에서부터 read한다. let users be able to make a balance between reliability and checkpointing overhead

##### Programming interface

TensorFlow software는 3 functional layer로 나누어지는데, user client layer는 여러 language(Python, C++, etc)의 client APIs(libraries)를 제공한다. (All programming APIs in TensorFlow are encapsulated in this layer)

TensorFlow는 low level math operations API도 제공할뿐만이 아니라 high level operations과 optimization algorithms for facilitating ML/DL을 제공한다.

PMLS나 Spark와 비교해서 TensorFlow는 더 많은 API와 primitives를 제공한다. (users can either deploy their ML/DL algorithms with build-in modules or build their algorithms from scratch by low-level APIs)



#### MXNet

MXNet은 TensorFlow와 매우 비슷한 전략을 활용한다. model이 dataflow graph로 표현되고 이 model은 parameter server를 사용해서 coordinate되고 abstracted away된 hardware에서 실행된다. MXNet는 n-dimensional arrays의 operation과 같이 imperative definition of dataflow graphs를 지원해서 특정 network을 보다 쉽게 구현할  수 있도록 해준다.

MXNet의 Parameter Server는 KVStore로 불리고, 기존의 key-value store상에서 구현된다. KVStore는 device에서부터 store까지 key-value pairs를 push하는 것을 지원한다. 그리고 current value of a key를 store에서부터 pull해오는 것을 지원한다. KVStore은 여러 다른 consistency models를 enforce할 수 있다. two-tier system인데, multiple threads와 GPU가 full cluster로 push되기 전에 local machine에서 merge된다. KVStore abstraction이 이론적으로는 stale synchronicity를 구현할 수 있도록 해준다.

MXNet is a dataflow system that allows cyclic computation graphs with mutable states, and supports training with parameter server model.

아래 그림과 같이, main components of MXNet. The runtime dependency engine analyzes the dependencies in computation processes and parallelizes the computations that are not dependent. On top of runtime dependency engine, MXNet has a middle layer for graph and memory optimization.

![MXNet_composition](C:\SJL\스터디_분산ML_system\figures\MXNet_composition.PNG)

##### Fault tolerance

checkpointing을 통한 basic fault tolerance 제공 (model parameters are saved in the checkpoint file)

##### Programming interface

declarative programming to represent computations in directed graphs



#### DMTK (Distributed Machine Learning Toolkit)

Microsoft가 만든 방식이고, Parameter Server인 Multiverso를 포함하고있다. (can be used together with CNTK to enable Asynchronous SGD instead of the default Allreduce-based distribution in CNTK)



#### Ray

***need to check further if Ray belongs here***

https://bair.berkeley.edu/blog/2018/01/09/ray/

**Relation to deep learning frameworks:** Ray is fully compatible with deep learning frameworks like TensorFlow, PyTorch, and MXNet, and it is natural to use one or more deep learning frameworks along with Ray in many applications (for example, our reinforcement learning libraries use TensorFlow and PyTorch heavily).

![What is Ray](C:\SJL\스터디_분산ML_system\figures\what_is_ray.png)

**Relation to other distributed systems:** Many popular distributed systems are used today, but most of them were not built with AI applications in mind and lack the required performance for supporting and the APIs for expressing AI applications. The following features are missing (in various combinations) from today’s distributed systems:

- Support for millisecond level tasks and millions of tasks per second
- Nested parallelism (parallelizing tasks within tasks, e.g., parallel simulations inside of a hyperparameter search) (see the figure below)
- Arbitrary task dependencies determined dynamically at runtime (e.g., to avoid waiting for slow workers)
- Tasks operating on shared mutable state (e.g., neural net weights or a simulator)
- Support for heterogeneous resources (CPUs, GPUs, etc)

<img src="C:\SJL\스터디_분산ML_system\figures\ray.PNG" alt="ray" style="zoom:67%;" />

https://docs.ray.io/en/ray-0.3.0/tutorial.html

- Ray executes tasks asynchronously to achieve parallelism

- How Ray uses object IDs to represent immutable remote objects.

Ray is a Python-based distributed execution engine. The same code can be run on a single machine to achieve efficient multiprocessing, and it can be used on a cluster for large computations.

When using Ray, several processes are involved.

- Multiple **worker** processes execute tasks and store results in object stores. Each worker is a separate process.
- One **object store** per node stores immutable objects in shared memory and allows workers to efficiently share objects on the same node with minimal copying and deserialization.
- One **local scheduler** per node assigns tasks to workers on the same node.
- A **global scheduler** receives tasks from local schedulers and assigns them to other local schedulers.
- A **driver** is the Python process that the user controls. For example, if the user is running a script or using a Python shell, then the driver is the Python process that runs the script or the shell. A driver is similar to a worker in that it can submit tasks to its local scheduler and get objects from the object store, but it is different in that the local scheduler will not assign tasks to the driver to be executed.
- A **Redis server** maintains much of the system’s state. For example, it keeps track of which objects live on which machines and of the task specifications (but not data). It can also be queried directly for debugging purposes.

OPEN MPI, Python multiprocessing, ZeroMQ가 spectrum의 한쪽끝이고, 반대쪽에는 TensorFlow, Apache Spark가 있다면, Ray는 이 spectrum의 중간에 위치한다. (그림 및 설명 at https://towardsdatascience.com/modern-parallel-and-distributed-python-a-quick-tutorial-on-ray-99f8d70369b8)

[Ray](https://github.com/ray-project/ray) occupies a unique middle ground. Instead of introducing new concepts. Ray takes the existing concepts of **functions** and **classes** and translates them to the distributed setting as **tasks** and **actors**. This API choice allows serial applications to be parallelized without major modifications.



### Parallel Stale-Synchronous Stochastic Gradient Descent

#### Petuum/PMLS

big data와 big models(hundreds of billions of parameters) 의 machine learning을 위한 generic platform을 제공해준다. Large dataset과 models에서 good scalability를 확보하기위해 ML의 error tolerance, dynamic structural dependencies, non-uniform convergence를 활용한다. fault tolerance와 recovery에 focus하는 Spark와는 거의 반대이다. minor한 만큼의 staleness는 convergence에 minor한 영향을 끼칠것이기때문에 Petuum은 stale synchronicity를 사용해서 machine learning의 inherent tolerance를 활용한다.  Dynamic scheduling policies를  적용해서 parallelization error와 synchronization cost를 최소화 시킬 수 있도록 dynamic structural dependencies를 활용한다.

Petuum은 Parameter Server paradigm을 사용해서 훈련되는 모델의 parameter들을 keep track한다. Parameter Server는 staleness guarantee를 유지하는 책임을 맡는다. Model developer가 parallelized model update의 순서를 제어할 수 있도록 scheduler를 expose한다. 

Petuum을 사용해서 model을 개발할때에는 push라는 method를 사용해야하는데, 이것은 각각의 parallelized model training operation을 담당한다. 역할 - pull the model state from the parameter server, run a training iteration, push a gradient to the parameter server. data-parallel model이 추가적인 operation을 필요로하지 않도록, 기본적으로 Petuum은 scheduling aspect와 parameter merging logic을 자동으로 관리한다. 만약 model-parallelism이 필요하다면, schedule method (telling each of the parallel workers what parameter they need to train)그리고 pull method(defining aggregation logic for each of the generated parameter gradients)가 구현되어야 한다.

Petuum은 HDFS과 YARN을 사용하는 system에서도 동작할 수 있도록 abstraction layer를 제공한다.(YARN: Hadoop job scheduler, HDFS: Hadoop file system) 그래서 이미 존재하는 cluster들에서 동작할때에 보다 쉬운 compatibility를 확보해준다.

**Petuum architecture:**

![](C:\SJL\스터디_분산ML_system\figures\petuum_architecture.PNG)

basic programming interface로 C++와 Java가 있다. (YARN과 Hadoop Distributed를 지원해서 Hadoop clusters로 바로 실행가능하도록 한다.)

Petuum은 iterative-convergent ML program의 unique properties (error tolerance, dependency structure, non-uniform convergence, compact updates)를 활용해서 ML algorithm의 convergence rate와 per-iteration time을 개선한다. 그래서 close-to-ideal P-fold speedup w/ P개의 machines의 개선을 목표한다.

다음과 같이 두개의 underlying systems가 있다:

1. **Bosen:** bounded-asynchronous distributed key-value store for <u>data parallel</u> ML programming이다. SSP consistency model을 사용한다. ML algorithm의 correctness를 sacrifice하지 않고도 MapReduce나 bulk synchronous execution을 outperform하는 asynchronous-like 방식을 구현한다.

2. **Strads:** dynamic scheduler for <u>model parallel</u> ML programming이다. ML update operations의 fine-grained scheduling을 수행한다. unsafe parallel operations (that can lead to non-convergence of ML programs)를 피하면서 ML program에서 가장 computation이 필요한 부분부터 prioritize한다.

현재 classic ML algorithm을 지원하는데, logistic regression, K-means, random forest, 그리고 supervised topic model(MedLDA)와 같은 최신 algorithm, deep learning, distance metric learning, and sparse cosing을 Petuum을 통해 distributed paradigm으로 구현할 수 있다.

Petuum의 deep learning system인 "Poseidon"은 single-machine인 Caffe project을 가지고 Caffe안에서 memory access routine을 Bosen의 distributed key-value store's distributed shared memory programming interfaces로 바꾸어서 distributed GPU system으로 바꾼다. Poseidon platform의 가장 큰 장점은 familiarity와 usability이다. Caffe user는 새로운 tool을 배울 필요 없이 바로 사용이 가능하다.

PMLS는 원래 "Petuum"으로 불렸다.

iteration을 기반으로 다음과 같은 general 방식으로 진행:

- first, perform computation on dataset and current model state to output intermediate result

- second, update the model state based on the result

worker process/thread가 up-to-date model parameter를 requesting하고 partition of data에 computation을 수행하는 역할을 맡는다.

parameter-server thread는 model parameter들을 update하고 store하며, worker들의 request에 응답하는 역할을 맡는다.

![PMLS_architecture](C:\SJL\스터디_분산ML_system\figures\PMLS_architecture.PNG)

위 그림 설명:

##### Server

parameter server은 distributed table로 구현된다. 모든 model parameter들이 이 tables로 저장되고 server threads로 maintain된다. 하나의 PMLS application에는 한개 이상의 table을 register할 수 있다. 

각 table은 multiple rows로 구성되는데, each cell in a row는 column ID로 identify되며 보통 한개의 parameter를 store한다. 이 table들의 row들은 multiple servers on different machines에 store될 수 있다.

##### Worker

workers는 각 iteration에서 partition된 dataset에 user가 설정한 computation computation을 수행하고, computation을 위해 up-to-date parameter를 request하는 역할을 수행한다. 하나의 worker는 multiple working threads를 포함할 수 있다. worker들 사이의 communication은 없지만, 그대신 worker들은 server와 communicate한다. "worker"와 "server"는 물리적으로 분리되어있지 않다. (server threads co-locate with the worker processes/threads in PMLS)

##### Error tolerance

efficiency와 consistency사이의 trade-off를 위해서, ML algorithm들의 error tolerant property를 활용한다. (most ML algorithms can tolerate bounded error in their iterative optimization process)

error tolerant property를 활용하기위해 SSP(Stale Synchronous Parallel) model형식을 사용한다 (user가 staleness threshold를 설정함)  SSP model에서는 worker threads가 slow thread를 기다리지 않아도 된다. Fast threads는 stale model parameter를 사용해서 computation을 실행할 수 있다. Stale model parameter를 사용하면 error를 발생시키지만, 이는 bounded error이기때문에 SSP model로 인해 straggler가 전체의 속도를 저하시키는 영향을 감소시킬 수 있다. (error가 어떻게 bounded되나? PMLS ensures the fastest working thread can not be s iteration ahead of the slowest working thread, where the staleness threshold s is configured by a user.)

##### Fault tolerance

parameter server안의 model parameter들을 checkpoint해서 fault tolerance를 가능하게 한다. failure로 부터 resume하기위해 전체 system이 마지막 checkpoint에서부터 다시 시작한다. PMLS는 매번의 iteration에서 checkpoint하지않고 user가 설정한 period마다 checkpoint한다.

##### Programming interface

C++언어로 구현되며, user API를 PMLS의 system API로부터 decouple하지 않는다. 즉, users can use any public methods in a PMLS's core class.

To write custom application: 

first, define tables to store model parameters (each table contain multiple rows of specific type) 하나의 table로 ML algorithm의 parameter들을 store할수도 있다. 

By its client APIs, user can access to model parameters in its cache and server threads

Parameter update를 위해, user가 update해야할 entry of a row를 specify하거나 또는 a whole row of table에다가 batch update를 perform할수도 있다. Parameter server는 *Clock()* 메소드를 invoke해서 buffered parameter updates를 server에게 release한다.

##### Shortcomings

PMLS는 C++ 언어를 사용해서 low-level에서 parameter server를 구현해야하기 때문에 실행 속도는 좋지만, user가 low-level API를 구동/제어 할줄 안아야겠다. (While PMLS has very little overhead. User of PMLS need to know how to handle computations using relatively low-level APIs.)



### Parallel Hybrid-synchronous SGD

synchronous 와 asynchronous방식 모두 각자의 significant drawbacks를 가지고있다. 그래서 few frameworks가 "middle ground"를 찾아서 each model of parallelism의 best properties만 통합해서 drawbacks를 줄이는 방법을 연구하고있다.

#### MXNet-MPI

MXNet-MPI는 asynchronous (Parameter Server)방식과 synchronous(MPI) 방식의 best acpect만 통합한 distributed ML 방식이다. MXNet의 기본적인 architecture과 동일하지만, 하나의 worker가 parameter server와 communicate하지 않고 대신에 여러 worker들이 group으로 함께 cluster해서 MPI에 synchronous SGD를 AllReduce와 함께 적용한다. 이 방식은 easy linear scalability of synchronous MPI approach와 asynchronous Parameter Server approach의 fault tolerance의 장점을 가지고있다.
