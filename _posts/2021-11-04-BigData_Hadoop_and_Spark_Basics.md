---
layout: post           									# (require) default post layout
title: "Big Data Analytics II"            # (require) a string title
date: 2021-11-04       									# (require) a post date
categories: [BigDataProcessing]   # (custom) some categories, but makesure these categories already exists inside path of `category/`
tags: [BigDataAnalytics]           	# (custom) tags only for meta `property="article:tag"`

---



Big Data Analytics

## Big Data, Hadoop, and Spark Basics

Big Data - Velocity, Variety, Volume, Veracity --> Value

Big Data requires parallel processing on account of massive volumes of data that are too large to fit on any one computer.

### Parallel Processing and Scalability

why use parallel processing - 기존에 single machine내의 storage와 compute에서 small data를 process했던 것과 다르게 big data는 multiple machine이 필요해진다.

#### linear processing vs. parallel processing

linear processing = 기존 처리 방식으로 problem을 set of sequential instructions로 나누어서 차례대로 처리한다. (a problem statement broken into a set of instructions that are executed sequentially till all instructions are completed successfully) 만약 중간에 error가 발생한다면 error가 해결되고나서 sequence전체가 처음부터 다시 실행되어야한다. 만약 problem이 크고 complex하다면 시간이 많이 소모되고 효율이 떨어질 위험이 커진다. so linear processing is mostly for simple computing tasks.

parallel processing = problem을 set of executable instructions로 쪼갠다. 나누어진 instructions는 각각 equal processing power를 가진 multiple execution nodes로 distribute된다. 그리고 병렬로 instruction들이 실행된다. 

instruction들이 separate execution node들에서 실행되기때문에, error가 발생한다면 해당 부분의 error를 해결하고 그대로 진행하면된다. node들이 다른 nodes로 부터 locally independent하기 때문에 가능하다.

![linear_vs_parallel](https://raw.githubusercontent.com/miscaminos/miscaminos.github.io/master/static/img/_posts/linear_vs_parallel_processing.PNG)

#### parallel processing 장점:

- 작업시간 감소 - parallel processing approach can process large datasets in a fraction of time (compared to linear processing)
- 분산을 통한 필요 자원(메모리, 계산) 감소 - less memory and compute requirements needed as set of instructions are distributed to smaller execution nodes

- 유연성 & infrastructure cost 감소  - flexibility - execution nodes를 필요에 따라 추가하거나 제거할 수 있다. 

#### Data scaling 

overflow of data를 manage, store, process하기위한 techniques

다루어야하는 data가 커질 수 록, machine의 available compute capacity와 storage memory도 커져야한다. Vertical scaling(scaling-up)은 single machine에 추가적인 capacity를 확보한다. 

하나의 node내에서 capacity 증가 diagram:

![vertical_Scaling](https://raw.githubusercontent.com/miscaminos/miscaminos.github.io/master/static/img/_posts/vertical_scaling.PNG)

Single node의 capacity를 증가시키는 전략보다 더 좋은 방법은 horizontal scaling이다. 

추가적으로 additional nodes를 더한다. 이렇게 형생된 Individual nodes를 합해서 computing cluster이라고 부른다. 이 computing cluster는 "embarrassingly parallel" calculation을 처리할 수 있다. these are kind of workloads that can easily be divided and run independent of one another. 

![parallel_scaling](https://raw.githubusercontent.com/miscaminos/miscaminos.github.io/master/static/img/_posts/horizontal_scaling.PNG)

만약 one process가 fail한다면, 다른 process들에게는 영향을 끼치지않고 이어서 진행될 수 있다.

![not_s_easy](https://raw.githubusercontent.com/miscaminos/miscaminos.github.io/master/static/img/_posts/not_so_easy_parallel.PNG)

horizontal scaling 으로 parallel 방식을 구현하다보면 "not so easy" 문제를 직면할 때도 있다. multiple computation들이 함께 coordinate해야한다. 하나의 process가 각자의 process를 완료하려면 다른 proces들의 처리 내용/상태를 먼저 인지해야하는 경우가 있다. 이런 경우 network선상으로 서로에게 message를 주고받거나 cluster내의 모든 process들이 access할 수 있는 file system에 message를 쓴다. 

#### fault tolerance

system 또는 cluster에서 node(machine, computer, process 등)하나의 고장이나 문제가 발생함으로 인해 전체에게 주는 영향을 최소화 하기위해서 fault tolerance가 존재한다.

fault tolerance는 ability of a system to continue operating without interruption despite an error in a node or group of nodes. cluster내의 one or more computer에 예상밖의 문제가 발생했을때 전체가 영향을 받지않는다.

fault tolerance는 Hadoop primary data storage system이나 이와 비슷한 storage systems(S3, object storage, 등) 에서 역할을 수행한다. 

fault tolerance의 활용 예시:

다음 그림과 같이 P1,P2,P3가 들어있는 맨 왼쪽의 1st node에 문제가 생기면, 그림의 맨 오른쪽 node를 새로 생성한 뒤, 다른 node에 copy해두었던 processes들(P1,P2,P3)을 새로운 node에 1st node와 동일하게 확보해서 전체 흐름을 이어갈 수 있다. (Clearly, this is an extraordinarily complex maintenance process, but the Hadoop filesystem is a robust and time-tested framework. It can be reliable to five 9's (99.999%))

 ![fault_tol](https://raw.githubusercontent.com/miscaminos/miscaminos.github.io/master/static/img/_posts/fault_tolerance.PNG)

### Big Data Tools and Ecosystem

Big Data tools 6 main categories:

- data technologies
- analytics and visualization
- business and intelligence
- cloud providers
- NoSQL databases
- programming tools

for visualization to make data understandable through graphs, charts and maps, use Tableau, Palantir, SAS, Pentaho and Teradata (to name a few examples of analytics and visualization tools)

to analyze data further based on visualized results from visualization tools, BI tools are used. Business intelligence tools transform raw data into meaningful actionable and information that is suitable for business analysis. examples of BI tools: Cognos, Oracle, PowerBI, Business Objects, and Hyperion



### Open source and Big Data

Hadoop plays a major role in 다음과 같은 open source Big Data projects:

- Hadoop MapReduce (not used as much as before)
- Hadoop File System(HDFS)
- Yet Another Resource Negotiator (YARN)

kubernetes도 더 많이 사용되고 역할이 중요해지고있음



### Big Data Use cases

manufacturing (optimization finding the right solution)

Enterprises using Big Data to gain insights from data collected by IoT devices

cases requiring parallel processing on account of massive volumes of data (too large to fit on any one computer)



### Apache  Hadoop

How does Hadoop work?

Hadoop has individual components for storing and processing data (term Hadoop is often used to refer to both the core components of Hadoop & ecosystem of related projects)

#### Hadoop's core components:

- Hadoop Common (essential part of the Apache Hadoop Framework) - refers to the collection of common utilities and libraries that support other Hadoop modules

- HDFS (storage component) - handles and stores large data, scales a single Hadoop cluster into as much as thousand clusters

- MapReduce (processing unit) - processes Big Data by splitting the data into smaller units and process them simultaneously, MapReduce가 가장 먼저 HDFS에 query data를 store하기위해 사용되었고 지금은 Hive, Pig와 같은 system들이 alternative로 존재한다.

- YARN(Yet Another Resource Negotiator) - component that prepares RAM and CPU for Hadoop to run data in batch processing, stream processing, interactive processing, and graph processing

#### Hadoop의 shortcomings:

- simple task를 하기위해 가장 좋은 방법은 아니다. it's not good for processing transactions due to its lack of random access
- when work cannot be parallelized or when there are dependencies in the data, Hadoop is not good. (data에 dependencies가 있다는 것은 예를 들면, record no.2를 처리하기전에 record no.1을 반드시 먼저 처리해야하는 경우를 의미한다.)
- low latency data access를 위해서는 좋지 않다 (low latency는 real time 특성을 provide하는데에 small(unnoticeable to humans) delays (between an input being processed and corresponding output)를 허용한다. low latency는 trading, online gaming, Voice Over IP를 구현하는 경우에 매우 중요하기때에 이런 application에는 Hadoop이 적절하지 않다.
- processing lots of small files
- intensive calculation with little data

위와 같은 단점들을 보완하는 방안으로 Hive, Pig가 개발되었다.

Hive: SQL과 비슷한 query를 제공해서 user들이 strong statistical functions을 사용할 수 있도록 해준다.

Pig: multi-query approach to cut down the number of times data is scanned

 

### Apache Spark

Apache Spark is open source in-memory application framework for distributed data processing and iterative analysis on massive data volumes

in-memory application이라는 것은 - all operations happen within the memory or RAM

Spark is written in Scala (general purpose programming language that supports both object-oriented and functional programming) and Spark runs on JVM(Java Virtual Machine)

distributed computing = group of computers or processors working together behind the scenes

**note: Distributed computing vs. parallel computing:**

이 두개의 terms can be used interchangeably, BUT! how they access memory is different. 

![parallel_vs_distributed](https://raw.githubusercontent.com/miscaminos/miscaminos.github.io/master/static/img/_posts/parallel_vs_distributed.PNG)

parallel computing shares memory while distributed computing have their own memory



#### Apache Hadoop vs. Apache Spark

Apache Hadoop은 MapReduce로 data processing은 수행하고, Apache Spark는 RDD(resilient distributed datasets)로 data processing을 수행한다.

Hadoop은 HDFS(Hadoop distributed file system)을 가지고있다. data files를 multiple machine에 저장할 수 있다. (able to store data files across multiple machines) 추가적으로 servers & machines를 더해서 증가하는 data volume을 대응할 수 있기때문에 file system이 scalable하다. 

반면 Spark는 distributed file storage system을 제공하지 않는다. Spark를 사용하기위해 Hadoop이 필요한것은 아니지만, Spark가 Haddop위에 사용되면 Hadoop의 HDFS에 있는 distributed datasets를 활용할 수 있기때문에 함께 사용된다. Spark는 Hadoop위에 computation을 목적으로 사용되는 것이다. 

Performance의 차이점:

Hadoop is great for batch processing, but inefficient for iterative processing, so they created Spark to fix this. Spark programs iteratively run about 100 times faster than Hadoop in-memory, and 10 times faster on disk. Spark’s in-memory processing is responsible for Spark’s speed. Hadoop MapReduce, instead, writes data to a disk that is read on the next iteration. Since data is reloaded from the disk after every iteration, it is significantly slower than Spark



#### why use Apache Spark

1. scalability

   Spark scales very well with data - massive datasets를 활용할때에 매우 중요하다.

   distributed system 은 inherently scalable하기때문에 multiple machines를 통해 horizontally scale될 수 있다. 즉, user는 추가 machine을 더해서 증가한 workload를 handle할 수 있다. Single system을 반복적으로 update하지 않아도 된다. scalability의 cap이 없다고 보면 된다.

2. fault tolerance and redundancy

   cluster내에서 하나 또는 여러개의 machines가 offline되어도 cluster는 계속해서 run할 수 있다.

    Spark는 distributed learning의 benefit을 모두 가지고있다. (Spark supports a computing framework for large-scale data processing and analysis. also provides parallel distributed data processing capabilities, scalability, and fault-tolerance on commodity hardware)

3. programmable flexibility

   enables programming flexibility - easy to use Python, Scala, Java APIs

##### MapReduce와의 차이점?

MapReduce는 iteration마다 disk or HDFS에 reads & writes가 필요하다. 이렇게 read & write을 자주 하는것이 시간과 cost가 많이 소요된다. 

Apache Spark은 read/write을 반복적으로 많이하지않고 대부분의 data를 in-memory에 keep한다. (avoiding expensive disk I/O, thus reducing overall time by orders of magnitude)

Apache Spark for data engineering & machine learning:

![libraries](https://raw.githubusercontent.com/miscaminos/miscaminos.github.io/master/static/img/_posts/ApacheSpark.PNG)

Apache Spark은 functional programming의 lambda functions을 사용한다. 이를 통해서 big data를 처리하는 workload를 worker node들에게 distribute하고 parallelized computation을 구현한다.

#### Functional programming Basics이란?

functional programming(FP) 는 mathematical function format을 따르는 programming 방식이다. 

"how-to" 보다는 final output인 "the what"에 focus하고 expression을 사용한다. 

Scala, Python, R and Java provide rudiment support 

Apache Spark는 Scala로 주로 쓰여진다. function을 first-class citizen으로 treat한다. Functions in Scala are passed as arguments to other functions returned by other functions and used as variables.

간단한 예를 들어보면 다음과 같이 element에 1을 증가시키는 function을 구현한다:

![FP_Example](https://raw.githubusercontent.com/miscaminos/miscaminos.github.io/master/static/img/_posts/FP_simple_example.PNG)

functional programming capabilities를 parallelization로 적용해서 다음과 같이 program이나 code의 수정 없이 you can scale the algorithm to any size by adding more compute and resources. 전체 computation을 그림과 같이 세개의 chunks로 나누어서 진행함으로서 function runs three times in parallel. (the result is the same as if the function exists on only a single node) 

![FP_parallel](https://raw.githubusercontent.com/miscaminos/miscaminos.github.io/master/static/img/_posts/FP_parallel.PNG)

Functional programming applies mathematical concept called lambda calculus.(lambda - every calculation can be expressed as anonymous function which is applied to a data set)

이런 병렬 계산 방식이 inherent하기때문에 어떤 size의 data이든 Spark cluster에 추가적인 resource만 더하면 가능하다. (all functional spark programs are inherently parallel)

#### parallel programming using RDD

RDD: Resilient Distributed Datasetes는 Spark의 primary data abstraction이다. Fault-tolerant collection of elements이다. RDD는 partitioned across the nodes of the cluster, capable of accepting parallel operations, 그리고 immutable하다.(cannot be changed once created)

모든 Spark application은 consists of **driver program** (runs user's main functions and runs multiple parallel operations on a cluster. 

RDD는 text, sequence files, Avro, Parquet and Hadoop input format tile types를 지원한다. 또한, local, Cassandra, H Base, HDFS, Amazon S3 그리고 다른 relational & noSQL databases의 file format을 지원한다. 

![RDD_supportedfiles](https://raw.githubusercontent.com/miscaminos/miscaminos.github.io/master/static/img/_posts/RDD_supported_files.PNG)

##### methods to create RDD:

1. create an RDD using an external or local file from a Hadoop supported file system (e.g., HDFS, Cassandra, HBase or Amazon S3)

   ![RDD_Create](https://raw.githubusercontent.com/miscaminos/miscaminos.github.io/master/static/img/_posts/RDD_create_Spark.PNG)

   Dataset is broken into partitions and partitions are each stored in a worker's memory.

2. apply parallelize function to existing collection in the driver program (this driver program can be any of supported high level APIs like Python, Java, Scala)

   important parameter for parallel collection:

   - number of partitions specified to cut the dataset (Spark runs one task for each partition of the cluster) 보통 cluster에서 CPU 하나당 2~4 partition을 원한다. 

   - Spark는 자동으로 number of partition을 set하려고하지만, 수동으로도 partition을 설정할 수 있다.(by passing the number as a second parameter to the parallelize function)

   ![RDD_parallel](https://raw.githubusercontent.com/miscaminos/miscaminos.github.io/master/static/img/_posts/RDD_create_snippet.PNG)

3. apply transformation on an existing RDD to create a new RDD

   parallel programming parses tasks into discrete parts that are solved concurrently using multiple processors. The processors access a shared pool  memory which in place has control & coordination mechanisms,

RDD와 parallel programming: RDD를 어떻게 만들었냐에 따라서 RDD can inherently be operated on in parallel.

RDD가 어떻게 parallel programming을 enable하는지 예시:

![RDD_parallel](https://raw.githubusercontent.com/miscaminos/miscaminos.github.io/master/static/img/_posts/RDD_parallel.PNG)

RDD는 immutability와 caching을 통해서 Spark에 resilience 특성을 제공한다. 

- immutability - RDDs are always recoverable as data is immutable
- persisting & caching - caching data set in memory cross operations (cache : alway fault tolerant and recoverable ) persistence allows future actions to be much faster (often by more than 10 times)

#### Scale-out/ Data Parallelism in Apache Spark

##### components:

Apache Spark consists of 3 main components:

![components](https://raw.githubusercontent.com/miscaminos/miscaminos.github.io/master/static/img/_posts/Apache_spark_components.PNG)

- Data Storage : loads data from data storage into memory. Any Hadoop compatible data source 가능
- Compute Interface : high-level programming APIs (Spark는 Scala, Python, Java 사용)
- Cluster Management framework : distributed computing aspect를 제어한다. Scaling big data를 위해서는 cluster management가 essential하다. 

##### Spark (a.k.a Spark Core) : 

Spark core는 base engine이다. 

Spark core는 fault-tolerant하고 performs large scale parallel and distributed data processing, manages memory, schedules tasks, houses APIs that define RDDs, contains a distributed collection of elements that are parallelized across the cluster.

##### scaling big data in Spark

![scaling_spark](https://raw.githubusercontent.com/miscaminos/miscaminos.github.io/master/static/img/_posts/Spark_scaling_bigdata.PNG)

Spark application은 driver program과 executor program으로 크게 나뉜다. 

**Executor program**은 worker nodes에서 실행된다. Spark는 만약 충분한 memory와 core가 확보가능하다면, additional executor processes on a worker를 시작할 수 있다. 

또한, Executor는 multithreaded calculation을 위해 multiple cores를 가질 수 있다. Spark가 executors에게 RDD를 distribute한다. 

Communication occurs among the driver and the executors. driver contains the Spark jobs that the application needs to run and split the jobs into tasks submitted to the executors.

**Driver program**은 work를 allocate하는 역할을 수행한다.

만약 Apache Spark가 하나의 기업이라고 가정하면, driver program이 기업의 executive management의 역할을 맡아서 obtaining capital, allocating work에 필요한 결정을 내리고, executor program은 junior employees 역할을 맡아서 주어진 resources를 가지고 주어진 work/job 을 수행한다.

**worker node**는 기업의 직원이 상주하는 physical office space로 생각하면된다. big data processing을 incrementally scale하기 위해서 additional worker nodes를 추가하면된다. 

#### Dataframes and SparkSQL

Spark SQL = Spark module for structured data processing

Spark SQL은 "dataframe"이라는 programming abstraction을 제공한다. Dataframes act as distributed SQL query engine

Dataframes are conceptually equivalent to a table in a relational database or a data frame in R/Python, but with richer optimizations

dataframes - highly scalable, support wide array of data formats and storage systems, state-of-the-art optimization and code generation through the Spark SQL Catalyst optimizer, seamless integration with all data tooling and infrastructure vis Spark
