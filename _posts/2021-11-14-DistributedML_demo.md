---
layout: post           									# (require) default post layout
title: "Distributed Machine Learning IV"            # (require) a string title
date: 2021-11-14       									# (require) a post date
categories: [DistributedML]   # (custom) some categories, but makesure these categories already exists inside path of `category/`
tags: [DistributedML]           	# (custom) tags only for meta `property="article:tag"`

---

Distributed Machine Learning



## Demonstrations

### Distributed ML platforms 비교 (Spark, PMLS, TF, MXNet)

**Spark**

Dataflow system을 대표하는 platform. 

Spark performs good for simple logistic regression(더 복잡한 machine learning tasks 수행 시, performance가 많이 떨어짐)

Hadoop vs. Spark (distributed logistic regression수행결과를 바탕으로 비교)

![Hadoop_vs_Spark_LR_performance](https://raw.githubusercontent.com/miscaminos/miscaminos.github.io/master/static/img/_posts/Hadoop_vs_Spark_Performance비교.jpg)

첫 iteration에서는 소요시간이 Hadoop과 spark가 비슷하지만, Spark RDDs는 memory안에서 serveral map operation을 수행하기때문에 interim data set을 disk에 쓸필요가 없다. 그래서 LR performance의 경우, 거의 10x 더 빠르게 진행된다. (그리고 Spark를 사용하면 Map과 Reduce function을 따로 작성할 필요가 없다?) 

**PMLS**

Parameter-server system을 대표하는 platform.

Parameter server model은 training machine learning과 deep learning 작업에 fast iteration과 good performance를 보여주었다. 

**TensorFlow**

More advanced system으로 TensorFlow가 있다.  

TensorFlow에서는 directed graph를 통해서 computation이 abstract & represent된다.

**MXNet**

TensorFlow와 비슷한점이 많다. 

MXNet is a dataflow system that allows cyclic computation graphs with mutable states, and supports training with parameter server model.



#### Evaluation

platform comparison experiment done by SUNY Buffalo - Logistic Regression과 MNIST image 분류 문제를 4개의 platform(Spark, PMLSTensorFlow, MXNet)에서 performance를 측정함 

commonly 사용 환경 - All of our experiments are conducted in Amazon EC2 cloud computing platform using m4.xlarge instances. Each instance contains 4 vCPU powered by Intel Xeon E5-2676 v3 processor and 16GiB RAM. The dedicated EBS Bandwidth of m4.xlarge instance is 750Mbps.

platform들이 release한 code를 사용하고, same setting, hyperparameters(e.g., learning rate, optimizer, activation function, number of units in the layer, etc)를 사용함 



#### Logistic regression 문제를 통한 비교 결과

two class logistic regression algorithm실행함.

data(synthetically made): data set에는 10,000개의 data samples and each sample에는 10,000개의 features를 가짐 (total size: 750MB)

PMLS의 경우에는 SSP model의 batch size =1, SSP=3를 설정함.

Spark의 경우에는 batch data processing에 적합하기때문에, full batch gradient descent (batch size=10,000)로 model을 train 함. (model parameter들을 Spark의 RDD가 아닌 driver에 store함)

TensorFlow와 MXNet의 경우에는, synchronous stochastic gradient descent training을 w/ varying batch sizes와 실행함. (batch sizes: 100, 500) TensorFlow는 between-graph replicated synchronous training으로 실행되었음. 

이 실험에서는 cluster of each system은 3 worker nodes + 1 extra node(driver 또는 parameter server역할을 맡은)가 포함되었음.

그 결과는 다음과 같다:  

System speed (samples per second) 값을 비교 시, 느린 -> 빠른 순으로 나열하면

- TF(batch size=100): 403, TF(batch size=500): 443  
- Spark: 5,883
- MXNet(batch size=100): 19,277, MXNet(batch size=500): 19,283
- PMLS(ssp=3): 21,132  

System speed로는 MXNet과 PMLS가 가장 빠르고, TensorFlow가 가장 느린것으로 확인됨. 

속도차이의 이유/배경:

1. PMLS는 system 자체가 Spark나 TensorFlow보다 더 가볍고, PMLS는 low level 언어인 C++ programming으로 high performance를 구현할 수 있는 것등, PMLS가 빠른 배경/이유가 있다. (반면, Spark의 경우 high level 언어인 Scala로 JVM 상에서 구현되기때문)

2. PMLS는 TensorFlow에 비해 less abstraction을 가지고있다. (TensorFlow는 꽤 많은 abstraction을 가지고있음) abstraction은 system의 complexity를 증가시키고 runtime overhead를 발생시킬 수 있다.



#### MNIST 이미지 분류 문제를 통한 비교 결과

(PMLS는 제외됨. 논문의 실험진행시 no suitable example code was released from PMLS)

trianing speed를 확인한것 외에도, utilization of CPU, network, memory of both worker and ps node를 Ganglia라는 tool을 사용해서 측정함. (Spark에서 ps는 driver를 의미함)

##### diff models w/ same cluster size:

size of EC2 cluster = (3 worker nodes + 1 ps node) for each system 으로 fix해서, 3개의 training model로 training을 진행 함 - softmax, single-layer neural network(SNN), multi-layer neural network(2개의 hidden layers 가진)

동일한 조건(setting, hyperparameter)를 사용했지만 단, Spark에는 full batch training을 진행함. (Spark MLlib의 default setting이 full batch임) ,TF와 MXNet에는 synchronous와 asynchronous training이 적용됨.

그 결과는 다음과 같다:  

<img src="https://raw.githubusercontent.com/miscaminos/miscaminos.github.io/master/static/img/_posts/platform_comparison_result1.PNG" alt="result" style="zoom: 67%;" />

multi-layer neural network을 model을 훈련시킬때에 (TensorFlow Asyn가 가장 높고, 이것을 제외하면) MXNet이 TensorFlow보다 속도가 높은 편이다. TF와 MXNet에 비해서 Spark는 model size가 증가할 수록, 속도가 significant하게 감소된다. 또한 아래 CPU utilization 수준을 보면, Spark가 다른 platform system들 보다 larger model training에 더 많은 CPU 사용이 필요한것이 확인됨. (indicates potentil CPU-bottleneck for Spark) 반면, network per worker은 나머지 두 platform보다 더 낮다(less network per worker)



<img src="https://raw.githubusercontent.com/miscaminos/miscaminos.github.io/master/static/img/_posts/performance_Distributed.PNG" alt="CPU_utilization" style="zoom:67%;" />

Figure 8 shows the memory utilization. Spark’s memory utilization is relatively higher than Tensorflow and MXNet. This is because Spark needs to cache the whole training data as RDD during the training process. By contrast, for Tensorflow and MXNet the memory utilization of storing data is smaller, as they only need to keep a mini-batch of training data in memory.

<img src="https://raw.githubusercontent.com/miscaminos/miscaminos.github.io/master/static/img/_posts/graph.PNG" style="zoom:67%;" />

##### same model w/ different cluster size:

cluster size를 scale하면서 system performance가 변화하는 것을 확인함. (SNN synchronous training with varying number of workers: 1,3,5 workers) ps node는 1개만 유지함.

아래 그림을 보면, cluster size가 증가할 수록 system speed와 per worker speed가 변화하는 것을 보여준다. TF와 MXNet에는 synchronization으로 인한 cost가 Spark보다 더 높다. (TF와 MXNet에서 worker의 수가 증가할수록 떨어지는 속도의 폭이 더 큼)

<img src="https://raw.githubusercontent.com/miscaminos/miscaminos.github.io/master/static/img/_posts/scalability_comparison.PNG" alt="scalability" style="zoom:67%;" />

아래 그림을 보면, cluster size가 커질 수록, parameter server node의 CPU가 더 많은 network I/O system calls를 serve해야해서 ps node의 CPU utilization이 증가한다.

We can see the CPU utilization of the worker is highly correlated with the training speed of worker. The CPU utilization of ps node increases as we increase the number of workers in the cluster. This is because the ps CPU needs to serve more network I/O system calls as the size of the cluster increases.

<img src="https://raw.githubusercontent.com/miscaminos/miscaminos.github.io/master/static/img/_posts/scalability_CPU_utilization.PNG" style="zoom:67%;" />

Network utilization per worker의 경우, Spark가 가장 적은 network utilization을 보임. Spark는 full batch training을 employ하고 low frequency of transferring data between worker and ps를 가지고있기때문이다. 그대신 Spark는 memory utilization이 타 platform system보다 두배 이상 높다. MXNet이 가장 적은 memory를 사용한다.

<img src="https://raw.githubusercontent.com/miscaminos/miscaminos.github.io/master/static/img/_posts/network_utilization_cluster_size.PNG" style="zoom: 67%;" />

<img src="https://raw.githubusercontent.com/miscaminos/miscaminos.github.io/master/static/img/_posts/memory_utilization_cluster_size.PNG" style="zoom: 67%;" />

More advanced dataflow systems are developed to allow cyclic execution graphs with mutable states in order to support the parameter-server model.

advanced data flow system

(the advanced dataflow systems developed for machine learning, TensorFlow and MXNet, failed to perform well in terms of speed. This is due to the overhead caused by the high levels abstractions used in these platforms. On the other hand, these abstractions enable these systems to work on multiple platforms and leverage not only CPU but also GPU and other computational devices. While these systems have been shown to scale to hundreds of machines, our experiments were performed with up to 6 workers, so they do not evaluate these platforms at large-scale. In our experiments, we found that asynchronous training of the workers with respect to the parameter-server achieved higher speeds than synchronous training)

usability front에서는 advanced dataflow system이 여러 장점을 제공한다. several benefits:

By adopting symbolic execution graphs, they abstract away from the distributed execution at the nodes level and also enable optimizations by graph rewriting/partitioning when staging the computation on the underlying distributed nodes. They provide, to some extent, flexibility of customizing the parameter-server implementation (with different optimization algorithms, consistency schemes, and parallelization strategies) at the application layer. While support for data-parallel training with parameter-server abstraction is provided, it is still very cumberome to program model-parallel training using these platforms.



### Multiprocessing (Python package)

threading module과 비슷하게 processes를 spawn할 수 있도록, local and remote concurrency를 제공한다. (effectively side-stepping the Global Interpreter Lock(GIL) by using subprocesses instead of threads) 

The [`multiprocessing`](https://docs.python.org/2/library/multiprocessing.html#module-multiprocessing) module also introduces APIs which do not have analogs in the [`threading`](https://docs.python.org/2/library/threading.html#module-threading) module. A prime example of this is the `Pool` object which offers a convenient means of parallelizing the execution of a function across multiple input values, distributing the input data across processes (data parallelism). The following example demonstrates the common practice of defining such functions in a module so that child processes can successfully import that module. 



This basic example of data parallelism using `Pool`,

```python
from multiprocessing import Pool

def f(x):
    return x*x

if __name__ == '__main__':
    p = Pool(5)
    print(p.map(f, [1, 2, 3]))
```

will print to standard output [1,4,9]. Pool이외에도 Process, Lock, Value, Array와 같은 class들이 있다.

https://docs.python.org/2/library/multiprocessing.html

Tools to process more than one task simultaneously range over a spectrum with one end that has tools like OpenMPI, Python multiprocessing, ZeroMQ and the other end that has domain-specific tools like TensorFlow(for model training), Spark (for data processing and SQL), and Flink(for stream processing) - https://towardsdatascience.com/modern-parallel-and-distributed-python-a-quick-tutorial-on-ray-99f8d70369b8

another example at Kaggle: https://www.kaggle.com/artyomp/resnet50-baseline/script



### Horovod (w/ TensorFlow) --> DL

performance metrics

1. predictive accuracy of the mode

   performance metric to compare multiple models

2. computational speed of the process (e.g., speedup)

   depends on platform on which model is deployed

   speedup = ratio of solution time for the sequential (or 1 GPU) algorithms vs. its parallel counterpart (w/ many GPUs) 

3. throughput (e.g., number of images per unit time processed)

   depends on the network type

4. scalability

   ability of a system to handle a growing amount of work efficiently

   depends on the cluster configuration

#### case study: image classification(ResNet50/ResNet152)

1. setup MUNGE/SLURM to allocate computing resources

   SLURM: open-source, fault-tolerant, highly-scalable cluster management and job scheduling system for large and small Linux clusters

   주로 sbatch/salloc commands를 사용한다.

2. write up the code for ResNet50, RestNet152 models training using CIFAR-10 dataset

   obtain the results of model performance (5 epochs) as 기준 baseline for scalability demonstration

3. Multiple GPUs in one Server

   TensorFlow의 distribution strategies중 하나를 통해 distributed training 실행해보기

   *<참고 link https://towardsdatascience.com/train-a-neural-network-on-multi-gpu-with-tensorflow-42fa5f51b8af>*

   "MirroredStrategy"라는 TensorFlow API 활용 for synchronous distributed training on multiple GPUs on one server

   SLURM script:

   ```
   #!/bin/bash
   #SBATCH --job-name=”ResNet50"
   #SBATCH --D .
   #SBATCH --output=ResNet50_%j.output
   #SBATCH --error=ResNet50_%j.err
   #SBATCH --nodes=1
   #SBATCH --ntasks=1
   #SBATCH --cpus-per-task=160
   #SBATCH --gres=gpu:4
   #SBATCH --time=00:20:00
   
   module purge; module load gcc/8.3.0 cuda/10.2 cudnn/7.6.4 nccl/2.4.8 tensorrt/6.0.1 openmpi/4.0.1 atlas/3.10.3 scalapack/2.0.2 fftw/3.3.8 szip/2.1.1 ffmpeg/4.2.1 opencv/4.1.1 python/3.7.4_ML
   
   python ResNet50.py -- epochs 5 -- batch_size 256 -- n_gpus 1
   python ResNet50.py -- epochs 5 -- batch_size 512 -- n_gpus 2
   python ResNet50.py -- epochs 5 -- batch_size 1024 -- n_gpus 4
   ```

4. Multiple GPUs in multiple Servers

   Horovod package & TensorFlow 함께 distributed training 실행해보기

   *<참고 link https://towardsdatascience.com/distributed-deep-learning-with-horovod-2d1eea004cb2>*

   **Horovod 설치/setup**

   Horovod는 python package이며, pip로 설치할 수 있다. 보통 it assumes installation of MPI(for worker discovery and reduction coordination) and Nvidia's NCCL-2 libraries(to support inter-GPU communication) 만약 MPI가 install되지 않았다면, Horovod가 포함하는 Gloo를 사용하면된다. Gloo는 open-source collective communications library (by Facebook)이다. Runtime에서 ```--gloo```  argument를 ```horovodrun```에게 pass해서 Gloo를 사용하도록 지정하면된다.

   **Horovod 사용방법**

   ```hvd``` object has to be initialized and wrap the optimizer

   GPU is bound to this process using its local rank, and we broadcast variables from rank 0 to all other processes during initialization

   Horovod Python program은 ```mpirun``` 명령어를 통해 launch된다. it takes as parameters the hostname of each server as well as the number of GPUs to be used on each server

   1. Import Horovod

   2. Horovod must be initialized before starting using

   3. Pin each GPU to a process. (employee local rank. i.e., first process in the server will be pinned to the first GPU, second process to the second GPU, and so on)

   4. Scale learning rate by number of workers

      Effective batch size in synchronous distributed training is scaled by the number of workers. An increase in learning rate compensates for the increased batch size. 

   5. Wrap optimizer in ```hvd.DistributedOptimizer```. (The distributed optimizer delegates gradient computation to the original optimizer, averages gradients using *allreduce* or *allgather*, and then applies those averaged gradients.)

   6. Specify `experimental_run_tf_function=False` to ensure TensorFlow uses Horovod’s distributed optimizer to compute gradients.

   7. Add `hvd.callbacks.BroadcastGlobalVariablesCallback(0)` to broadcast initial variable states from rank 0 to all other processes. 훈련시, 이전의 checkpoint에서 restore하거나 random weights로 훈련을 시작할때에 모든 worker들이 consistent한 initialization을 하도록 보장하기위해 반드시 필요한 단계이다.

   8. 만약 checkpoint로 저장한다면. If you need to save checkpoints, do it only on worker 0 to prevent other workers from corrupting them. Or if you want to run evaluation or to print information to the standard output, it is recommended to do it on worker 0. This can be accomplished with `hvd.rank() = 0`.

      SLURM script:

        ```
        #!/bin/bash
        #SBATCH --job-name horovod1
        #SBATCH -D .
        #SBATCH --output hvd_1_%j.output
        #SBATCH --error hvd_1_%j.err
        #SBATCH --nodes=1
        #SBATCH --gres='gpu:4'
        #SBATCH --ntasks-per-node=4
        #SBATCH --cpus-per-task 40
        #SBATCH --time 00:25:00

        module purge; module load gcc/8.3.0 cuda/10.2 cudnn/7.6.4 nccl/2.4.8 tensorrt/6.0.1 openmpi/4.0.1 atlas/3.10.3 scalapack/2.0.2 fftw/3.3.8 szip/2.1.1 ffmpeg/4.2.1 opencv/4.1.1 python/3.7.4_ML

        horovodrun -np $SLURM_NTASKS -H localhost:$SLURM_NTASKS --gloo \
        python3.7 tf2_keras_cifar_hvd.py --epochs 10 --batch_size 512
        ```

##Reference
Zhang Kuo, et al. A Comparison of Distributed Machine Learning Platforms. Conference Paper. 2017 
