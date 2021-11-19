---
layout: post           									# (require) default post layout
title: "Distributed Machine Learning IV"            # (require) a string title
date: 2021-11-14       									# (require) a post date
categories: [DistributedML]   # (custom) some categories, but makesure these categories already exists inside path of `category/`
tags: [DistributedML]           	# (custom) tags only for meta `property="article:tag"`

---

Distributed Machine Learning

# Demonstration

Distributed 환경에서 application을 구현할때에 어떤 방식으로 data communication과 processing이 진행되는지에 따라 다음과 같이 spectrum을 그려볼 수 있다.

![](C:\SJL\스터디_분산ML_system\figures\tools_distributed_setting.jpeg)

Spectrum의 한쪽 끝에는 OpenMPI, Python multiprocessing, ZeroMQ와 같이 messages를 주고 받기 위해 low-level primitives를 제공하는 tool이 있다. 이 tool들을 가장 basic하고 powerful하지만, 이들을 통해서 single-threaded application을 large scale로 구현하려면 아얘 다시 application을 작성해야하는 번거로움이 있다. 

이 spectrum의 반대편에는 ML model 훈련을 위한 TensorFlow, data processing과 SQL을 위한 Spark, stream processing을 위한 Flink와 같은 domain-specific tool들이 있다. 이들은 neural networks, datasets, streams와 같은 high-level abstraction을 제공하여 large scale 구현이 가능하도록 한다. 

Ray는 이 spectrum의 중간정도에서위치하며 양쪽의 tool들과는 다르게 scalability를 더욱 간단하게 제공해준다. Ray는 기존 application을 구성하는 functions & classes를 distributed setting에서의 tasks & actors로 변환시켜서 새로운 concept/abstraction 을 만들지않고 concept을 그대로 유지한다. Non-distributed 환경에서 distributed로의 transition을 가장 쉽게 구현할 수 있도록 하는 도구이기도 하다.

(*source: Tools to process more than one task simultaneously range over a spectrum with one end that has tools like OpenMPI, Python multiprocessing, ZeroMQ and the other end that has domain-specific tools like TensorFlow(for model training), Spark (for data processing and SQL), and Flink(for stream processing) - https://towardsdatascience.com/modern-parallel-and-distributed-python-a-quick-tutorial-on-ray-99f8d70369b8*)

## Ray

open source framework that supports distributed machine learning training system - helps to enhance computational efficiency

link : https://docs.ray.io/en/latest/index.html 

https://www.kdnuggets.com/2021/03/getting-started-distributed-machine-learning-pytorch-ray.html

### RayTune

using RayTune(A Python library for hyper parameter tuning at large (or any) scale) with Pytorch: https://medium.com/pytorch/getting-started-with-distributed-machine-learning-with-pytorch-and-ray-fd83c98fdead 

### RaySGD for Image classification

using RaySGD(A Python library built on top of distributed PyTorch for easy & flexible application deployment) with Pytorch: https://medium.com/distributed-computing-with-ray/faster-and-cheaper-pytorch-with-raysgd-a5a44d4fd220 (scale-out PyTorch training across a cluster on AWS/ github code available: https://github.com/ray-project/ray/tree/master/python/ray/util/sgd/torch/examples) 

*--> CIFAR images로 ResNet18 모델 훈련진행. AWS의 p3.8xlarge instance type으로 총16개의 V100 GPUs를 통해 30분 훈련시, 대략 1만원 비용 소모됨.*

*실행 코드 @Github,* 

*data 확보 및 model training  코드:  **raysgd_torch_signatures.py*** 

*Ray를 통한 distribution 구현 코드: **example-sgd.yaml***

### Multiprocessing & Ray 

#### Python module multiprocessing

[Python multiprocessing](https://docs.python.org/3/library/multiprocessing.html) offers one solution to this, providing a set of convenient APIs that enable Python programs to take advantage of multiple cores on a single machine. (However, while this may help an application scale 10x or maybe even 50x, it’s still limited to the parallelism of a single machine and going beyond that would require rethinking and rewriting the application.)

Python의 multiprocessing package는 threading module과 비슷하게 processes를 spawn한다. Effectively side-stepping the Global Interpreter Lock(GIL) by using subprocesses instead of threads, 이 library는 local and remote concurrency를 제공한다.

The [`multiprocessing`](https://docs.python.org/2/library/multiprocessing.html#module-multiprocessing) module also introduces APIs which do not have analogs in the [`threading`](https://docs.python.org/2/library/threading.html#module-threading) module. A prime example of this is the `Pool` object which offers a convenient means of parallelizing the execution of a function across multiple input values, distributing the input data across processes (data parallelism). The following example demonstrates the common practice of defining such functions in a module so that child processes can successfully import that module. 

*Pool* class is a better way to deploy Multi-Processing because it distributes the tasks to available processors using the First In First Out schedule. It is almost similar to the map-reduce architecture- in essence, it maps the input to different processors and collects the output from all processors as a list. The processes in execution are stored in memory and other non-executing processes are stored out of memory.

Multiprocessing에는 Pool, Process, Lock, Value, Array와 같은 class들이 있다. 

https://docs.python.org/2/library/multiprocessing.html

예시:

multiprocessing library를 활용하여 다음 그림과 결과와 같이 동시에 'sleepy_man' function이 실행된다.

```Python
import multiprocessing
import time

def sleepy_man():
    print('Starting to sleep')
    time.sleep(1)
    print('Done sleeping')

tic = time.time()
p1 =  multiprocessing.Process(target= sleepy_man)
p2 =  multiprocessing.Process(target= sleepy_man)
p1.start()
p2.start()
toc = time.time()

print('Done in {:.4f} seconds'.format(toc-tic))
```

결과:

```
Done in 0.0023 seconds
Starting to sleep
Starting to sleep
Done sleeping
Done sleeping
```

<img src="C:\SJL\스터디_분산ML_system\figures\multiprocessing1.png" alt="multiprocessing1" style="zoom:50%;" />



```Python
import multiprocessing
import time

def sleepy_man():
    print('Starting to sleep')
    time.sleep(1)
    print('Done sleeping')

tic = time.time()
p1 =  multiprocessing.Process(target= sleepy_man)
p2 =  multiprocessing.Process(target= sleepy_man)
p1.start()
p2.start()
p1.join()
p2.join()
toc = time.time()

print('Done in {:.4f} seconds'.format(toc-tic))
```

결과:

```
Starting to sleep
Starting to sleep
Done sleeping
Done sleeping
Done in 1.0090 seconds
```

<img src="C:\SJL\스터디_분산ML_system\figures\multiprocessing2.png" alt="multiprocessing" style="zoom:50%;" />

다음 예시들을 보면 multiprocessing API들을 통해 " Perfect number"를 찾는 computation의 속도가 점점 더 향상되는것을 확인할 수 있다.

for-loop 사용시,

```python
import time

def is_perfect(n):
    sum_factors = 0
    for i in range(1, n):
        if (n % i == 0):
            sum_factors = sum_factors + i
    if (sum_factors == n):
        print('{} is a Perfect number'.format(n))

tic = time.time()
for n in range(1,100000):
    is_perfect(n)
toc = time.time()

print('Done in {:.4f} seconds'.format(toc-tic))
```

```
6 is a Perfect number
28 is a Perfect number
496 is a Perfect number
8128 is a Perfect number
Done in 258.8744 seconds
```



Process 사용시,

```python
import time
import multiprocessing

def is_perfect(n):
    sum_factors = 0
    for i in range(1, n):
        if(n % i == 0):
            sum_factors = sum_factors + i
    if (sum_factors == n):
        print('{} is a Perfect number'.format(n))

tic = time.time()

processes = []
for i in range(1,100000):
    p = multiprocessing.Process(target=is_perfect, args=(i,))
    processes.append(p)
    p.start()

for process in processes:
    process.join()

toc = time.time()
print('Done in {:.4f} seconds'.format(toc-tic))
```

```
6 is a Perfect number
28 is a Perfect number
496 is a Perfect number
8128 is a Perfect number
Done in 143.5928 seconds
```



Pool 사용시,

```python
import time
import multiprocessing

def is_perfect(n):
    sum_factors = 0
    for i in range(1, n):
        if(n % i == 0):
            sum_factors = sum_factors + i
    if (sum_factors == n):
        print('{} is a Perfect number'.format(n))

tic = time.time()
pool = multiprocessing.Pool()
pool.map(is_perfect, range(1,100000))
pool.close()
toc = time.time()

print('Done in {:.4f} seconds'.format(toc-tic))
```

```
6 is a Perfect number
28 is a Perfect number
496 is a Perfect number
8128 is a Perfect number
Done in 74.2217 seconds
```

Pool class사용시, 기존 for-loop을 사용할때 보다 71%수준의 computation time reduction이 가능하다.

Python multiprocessing을 사용하는 demonstration은 Ray와 함께 진행했다. (나중에 Ray 부분에서 나올 예정)

#### Multiprocessing for Deep Learning problem

이 보다 훨씬 더 복잡한 ResNet50와 같은 모델을 구동하는데에도 multiprocessing이 사용될 수 있다. 

muchhhh more complex example at Kaggle for Google Landmark Recognition competition 2019: https://www.kaggle.com/artyomp/resnet50-baseline/script



그러나 multiprocessing module는 다음과 같이 modern application에 필수적인 요소들을 충족하지 못한다:

- Running the same code on more than one machine. (?)
- Building [microservices](https://en.wikipedia.org/wiki/Microservices) and [actors](https://en.wikipedia.org/wiki/Actor_model) that have state and can communicate.
- Gracefully handling [machine failures](https://en.wikipedia.org/wiki/Fault_tolerance).
- Efficiently handling [large objects and numerical data](https://ray-project.github.io/2017/10/15/fast-python-serialization-with-ray-and-arrow.html).

이점들을 cover하기위해 Ray framework이 사용될 수 있다.

#### Multiprocessing API  with Ray

serial Python, Python의 multiprocessing library의 pool class, Ray framework의 scalability 비교 그래프:

Monte Carlo Pi estimation을 구현한 결과를 sample수 range에 따라 그려본 결과이다.

![comparison_Python_Ray](C:\SJL\스터디_분산ML_system\figures\comparison_serial_multiprocessing_RayMultiprocessing.png)

***need to use AWS instances***

parallel on a single AWS m4.4xlarge instance using multiprocessing.Pool,

distributed on a 10-node cluster of AWS m4.4xlarge instances using Ray.

[how-to] k8s cluster만들기 on AWS: https://www.golinuxcloud.com/setup-kubernetes-cluster-on-aws-ec2/



EC2: https://introduction-to-inferentia.workshop.aws/

EC2 cluster: https://ecs-cats-dogs.workshop.aws/en/01_intro.html



sources: https://medium.com/distributed-computing-with-ray/how-to-scale-python-multiprocessing-to-a-cluster-with-one-line-of-code-d19f242f60ff

https://towardsdatascience.com/modern-parallel-and-distributed-python-a-quick-tutorial-on-ray-99f8d70369b8



## TensorFlow Distributed

TensorFlow-Custom-Distributed -training-GPU (Plant Pathology 2021):https://www.kaggle.com/mohammadasimbluemoon/tensorflow-custom-distributed-training-gpu

CosFace (Google Landmark Retrieval 2020) with distributed TF: https://www.kaggle.com/akensert/glret-cosface-with-distributed-tf/notebook

https://github.com/cvdfoundation/google-landmark



##Reference
Zhang Kuo, et al. A Comparison of Distributed Machine Learning Platforms. Conference Paper. 2017 
