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




# The Distributed Machine Learning Ecosystem

cluster를 통해 large volume의 data를 process하는 문제는 machine learning분야 만이 아니라 distributed system과 database research 분야에서도 오랫동안 연구되어 왔다. 그 결과 Apache Spark와 같은 general purpose distributed platform이 distributed system의 현실적인 구현 방법으로 활용되었고, MLlib와 같은 최적화된 library가 제공되고 있다. general purpose와는 반대의 방향에는 purpose-built machine learning libraries가 있는데, 기존에는 single machine에서 동작하도록 design되었으나, 점점 더 distributed setting에서 실행될 수 있도록 개선되고 있다. 예를 들어, Keras는 Google의 TensorFlow와 Microsoft의 CNTK에서 실행 될 수 있도록 backends를 받았다. Nvidia 또한 그들의 machine learning stack을 더 발전시켜서 Collective Communication Library(NCCL)을 구현했는데, 기존에 동일 machine에서 multiple GPU를 지원하는 것이였지만, version 2 부터는 multiple nodes에서 실행 될 수 있도록 발전 시켰다. 아얘 처음부터 distributed machine learning을 위해 설계되고 만들어진 specific algorithm과 operational model이 distributed ML ecosystem의 중심에 있다. e.g., Distributed Ensemble Learning, Parallel Synchronous Stochastic Gradient Descent (SGD), 또는 Parameter Servers. 원래 대부분의 system들은 user 또는 on-premise로 운용(operate)되도록 의도되어왔으나, 점점 더 많고 다양한 machine learning services가 cloud delivery model을 통해 공급되고 있다. 이들은 established distributed machine learning system을 중심에 두고 surrounding platform으로 인해 개선되고있으며 해당 기술/technology가 data scientist나 결정권자들에게 더 쉽게 사용할 수 있도록 만들어가고 있다.

## General Purpose Distributed Computing Frameworks

distributed system은 하나의 값비싼 large server보다는 number of commodity servers(each with relatively small storage capacity and computing power)를 활용하는 방식에 의존한다. 이 방식은 Google이 가장 처음 도입했던바와 같이 충분히 fault tolerance를 software에 build하는 방식이 확보되어 있는 한, 비싼 specialized hardware을 사용하는 것보다 훨씬 더 affordable하다는 것으로 확인된다. 그리고 scale-out model에서는 각 node의 I/O subsystem이 모두 포함되기 때문에, 소수의 more powerful machine을 사용할때와 비교해보면 결국 더 높은 aggregate I/O bandwidth를 제공한다. 이 특성은 data ingestion이 매우 중요한 data-intensive application에서 매우 큰 도움이 된다. 

- storage: Google File System(GFS), block-based model

  현존하는 framework들의 storage layer는 GFS나 또는 comparable implementation들에 기반되어있다. GFS는 Google이 소유하고 사용하는 시스템으로 모든 Big Data storage 요구사항을 처리하기위해 사용된다. GFS는 cluster로 upload된 data를 chunks로 나누어서 chunk servers로 distribute한다. chunk들은 machine failure발생시 사용할 수 없게되는 상황을 대비해서 미리 보호하기위해 replicate된다. User는 master를 통해서 chunk server들에 있는 data를  access할 수 있다. master는 name node 역할을 수행하는데 file의 모든 chunk의 위치를 알려준다. GFS architecture은 Hadoop이 adopt했고, 지금은 Apache Foundation이 maintain하고 있다. Hadoop File System (HDFS)은 GFS design과 거의 동일하고 이름(nomenclature)만 다르며, storage layer 역할을 수행한다.

- compute: (storage와는 다르게) compute을 위한 frameworks는 다양하게 존재한다. 여러 framework들은 자원들(resources)을 compute하는 작업들을 여러 features 및 trade-offs와 함께 scheduling 또는 distributing하기 위해 서로 경쟁한다.

  - MapReduce:  Distributed setting에서 data를 처리(process)하기 위해 Google이 개발한 framework이다. Functional programming concept으로부터 아이디어를 빌려오고, multiple phase로 구성된 architecture를 가지고 있다. 

    먼저 "map phase"동안, 모든 data는 tuple(a.k.a "key-value pairs")로 split된다. 이것은 functional programming에서 second-order function을 하나의 set과 mapping하는 것과 comparable하다. "map phase"에서는 두 가지 다른 value들 사이에서 function을 mapping할때에 data dependencies가 없기때문에, 이 phase가 fully parallel하게 실행될 수 있다. 그 다음, "shuffle phase"에서는 tuples가 node들 사이에서 exchange되고 전달된다(passed on). 보통 aggregation은 data dependencies를 가지고 있고, 정확도를 위해 동일한 key에서 기인한 모든 tuple들은 동일한 node로 처리되는것이 보장되어야하기때문에 필수적이다. 그 다음 순서인 "reduce phase"에서는 각  key당 single output value를 생성하기위해 tuples에 aggregation을 실행시킨다. 이것은 functional programming의 fold operation과 비슷하다. fold operation은 single result value를 생성하는 second-order function을 사용하여 collection을 하나 roll up 한다. 그러나 fold의 구성을 보면, 각 fold step이 바로 전 step에 의존하기 때문에 fold는 parallelized되지 못한다. "Reduce phase"에서 parallelism을 가능하게하기 위해서는 data를 shuffling하고 key를 reducing 하는것이 필요하다. 

    MapReduce의 가장 큰 장점은 이 framework을 통해서  data dependencies없이 그리고 전부 parallel(병렬)방식으로 동일 phase의 작업(tasks)을 수행 하는 동시에 아주 많은 machine들에게 data가 distribute될 수 있다는 것이다. 동일 machine들은 GFS (또는 비슷한 타 storage cluster)의 node가 될수 있어서 data를 program으로 이동시키는 것대신에 program을 data로 옮기고 data locality와 performance를 개선하는 방법이있다. Program은 wire선상에서 보통 several orders of magnitude 더 작기때문에 전달하기에 매우 효율적이다. 

    그리고 scale-out 방식의 관점으로 보면, MapReduce는 실패한 작업들(tasks)을 healthy node들에게 rescheduling하는 방식과 heartbeat message를 사용해서 worker node들의 health를 monitoring하면서 software에 fault-tolerance를 구현한다. 보통 하나의 작업(task)의 granularity는 input data set의 single block의 크기와 동일하기때문에 node failure은 전체 application의 작은 일부분만 영향을 끼친다. 전체 시스템은 손쉽게 recover할 수 있다. 몇가지 machine learning algorithm들을 MapReduce framework에 map해서  multicore machines를 위한 parallelism을 활용한 연구/논문도 있다 (Chu et al[36])

    **MapReduce & BSP(Bulk-Synchronous Processing)**

    MapReduce는 BSP paradigm과 비슷하다. 단, 작은 차이점이 있다. MapReduce framework은 map phase에서 worker nodes사이의 communication을 허용하지 않는다. 대신에 map phase와 reduce phase사이에 있는 shuffle phase에서 cross-communication을 허용한다. synchronization barrier를 줄이고 parallelism을 증가시키기 위함이다. 지난 연구 논문들을 참고해보면, 모든 BSP program들은 MapReduce program으로 convert될 수 있다는것이 확인되었고(Goodrich et al[59]), 모든 MapReduce application들은 BSP paradigm으로 model되어야 BSP의 theoretical correctness와 MapReduce의 효율적인 execution을 통합적으로 장점으로 활용할 수 있다는것이 확인되었다(Pace[116]).

    MapReduce는 Google의 framework인데, 이것의 architecture은 open source Hadoop framework에서 recreate되었다. Hadoop은 HDFS를 leverage하고, MapReduce는 GFS를 사용한다. 그러나 이들의 전반적인 architecture는 동일하다. Strict한 topology를 가진 MapReduce에서 더 flexible한 구조를 가진 Forests나 generic Directed Acyclic Graphs (DAGs)로 변화하여 advanced variants들이 생겨났다.
