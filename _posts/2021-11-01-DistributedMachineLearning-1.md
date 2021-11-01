---
layout: post           									# (require) default post layout
title: "Distributed Machine Learning I"            # (require) a string title
date: 2021-11-01       									# (require) a post date
categories: [machineLearning]   # (custom) some categories, but makesure these categories already exists inside path of `category/`
tags: [DistributedML]           	# (custom) tags only for meta `property="article:tag"`


---



Distributed Machine Learning

# Communication

Big Data Analytics 분야에서도 Machine learning 에서도 기술이 발전하고 고객들이 더욱 resource consumption과 return of investment에 주목하게 되면서, system aspect가 더욱 더 중요한 요소가 되어가고 있다. 

ML algorithm들과 system이 점점 더 co-designed된 방향으로 발전해나아가고 있다. 예를들어서 system resources를 더 잘 사용할 수 있도록 algorithm들을 adapt하거나 또는 특정 algorithm classes를 더 잘 지원하는 새로운 system을 만들기도 한다.

Distributed machine learning system은 주어진 hardware resource로 더욱 autonomous한 ability를 확보해서 computation과 distribution을 최적화해 나아가고 있다. 주요 machine learning libraries가 machine learning 기술을 전반적으로 더 쉽게 접근할 수 있도록 유도한것처럼 앞으로의 지속적인 발전이 distributed machine learning을 활용하는데의 진입장벽을 더 낮추어 줄것이다. 

## Nodes synchronization

spectrum: fast/correct model <---> faster/fresher updates (from top of list to bottom of list below)

BSP (Bulk Synchronous Parallel): ensures consistency by synchronizing btwn each computation and communication phase. guaranteed to output correct solution, but is slow (e.g., MapReduce)

SSP (Stale Synchronous Parallel): relaxes synchronization overhead by allowing faster workers to move ahead for a certain number of iterations. SSP still guarantees strong model convergence but staleness can get too high. (when staleness becomes too high, convergence rate can quickly deteriorate) (e.g., Conits 논무 [167])

ASP (Approximate Synchronous Parallel): SSP와는 반대로 parameter가 얼마나 inaccurate될 수 있는지를 제한하는 방식이다. 이 방식에서는 만약 aggregated update가 중요한 수준이 아니라면(is insignificant), synchronization을 무한으로 연기할 수도 있다. 단지, 어떤 parameter를 선택해야 올바르게 update가 insignificant한지 아닌지를 판단할 수 있는지가 어렵다.

BAP/TAP (Barrierless Asynchronous Parallel/Total Asynchronous Parallel): 이 방식에서는 기다림 없이 worker machine들이 바로 서로 병렬로 communicate한다. 아주 빠르다는 것이 장점이지만, model convergence가 느리게 확보될수있다는 risk가 있다. Model이 아얘 incorrect하게 develop될 수도 있다. BSP나 SSP와는 다르게 error가 delay와 함께 커질 수 있다.  

### communication strategies

- continuous communication -  in order to prevent burst of communication over the network
- neural network가 layers로 구성되어있고 top layer가 contain 하고 있는 parameter는 total computation에 작은 portion만 차지한다는 사실을 활용해서 WFBP(Wait-free Backpropagation)을 제시한다. (WFBP exploits the neural network structure by already sending out the parameter updates of the top layers while still computing the updates for the lower layers, hence hiding most of the communication latency.)
- WFBP가 communication overhead를 줄여주지는 못하기때문에, hybrid communication을 적용한다. (e.g., Parameter Servers + Sufficient Factor Broadcasting = choose the best communication method depending on the sparsity of the parameter tensor)



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
