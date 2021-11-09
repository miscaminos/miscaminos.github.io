## Principles & 전략 of ML system design

Large scale에서 수행 속도를 높히기 전에 먼저 ML algorithms이 가진 unique properties의 이해가 필요하다. 

**주요 4 unique properties of ML algorithm: **

- error tolerance

  ML algorithms are robust against minor errors in intermediate calculations.

  만약  limited number of updates가 틀리게 계산되어 접목되었어도 ML program은 optimal set of model parameters로 converge되도록 mathematically guarantee되어있다.

- dependency structure

  model의 parameter들이 update되는 순서가 ML program의 progress 또는 correctness에 영향을 끼친다.

- non-uniform convergence

  모든 model parameter들이 동일한 number of iteration을 통해 optimal 값으로 converge되지않는다. converge하기 위해 특정 (non-zero)parameter들의 computation이 prioritize되어야한다. 

- compact updates

  updates are significantly smaller than the size of the matrix parameters. "matrix-parametrized" model (model parameters가 matrix형태), individual updates를 few small vector(low-rank update)로 reduce될 수 있다. 이렇게 compactness를 통해 storage, computation 그리고 communication cost를 줄일 수 있다. 이점을 가만해서 distributed system을 design하면 order of magnitude speedup을 이끌어낼 수 있다.

ML programs are parallelized by subdividing the updates over either the data or the model —referred to respectively as data parallelism and model parallelism:

​	**data parallel :** IID assumption을 기반으로 포괄적으로 대부분의 ML programs에 적용 할 수 있다

​	**model parallel :** data parallel과는 다르게, update function takes scheduling 및 selection function. 예를들어 in order to prevent different workers from trying to update the same parameters



### Principles

ML algorithm의 unique properties와 data parallelism & model parallelism의 complementary strategies가 함께 couple되어서 이상적인 mathematical view보다 더 좋은 성능을 낼 수 있는 design을 설계할 수 있다. (beyond the ideal mathematical view that is suggested by general iterative-convergent update equation)

Parallel ML을 위한 system interface가 다음과 같은 역할을 수행하기 위해 필요하다:

- facilitate the organized, scientific study of ML considerations
- organize these considerations into a series of high-level principles for developing new distributed ML systems. ML program's equation이 the distributed system에게 "무엇을 compute"할지를 알려주면, system must be able to answer the following questions:

#### how to distribute the computation

- dependency structure: mutually dependent feature들로 인한 intricate dependencies를 handle하기위해서 (e.g., LDA topic modeling) simultaneous data and model parallel strategy가 필요하다.

- scheduling in ML programs: dependency structure를 최대한 violate하지 않기위해서 updates를 schedule하고 task가 없거나 load balance를 적절하게 설정하지 못해서 사용가능한 worker를 idle하게 두는것을 방지한다. 

  - dependency check를 통해 어느 수준의 degree만큼 A_j와 A_k parameter들이 interdependent한지(data correlation between j-th and k-th feature dimensions)를 확인하다. 작은 threshold를 지정하고 data correlation이 threshold를 넘지않는 경우에만 interdependent subset data로 따로 구분하여 parallel worker machine들에 assign될 수 있도록 한다. worker들이 각 subset에 있는 parameter들을 순서대로(sequentially) update하기 때문에 dependency structure을 violate하는 것을 방지할 수 있다. 그러나 모든 data pair의 correlation을 계산하는것은 매우 비현실적이기때문에 SAP(Structure aware parallelization) 방식을 고려해야한다. SAP에서는 prioritization을 먼저 진행해서 선택된 것들에 대한 correlation 계산만 진행할 수 있도록 제한을 두어 더 현실적인 방법이다.

- compute prioritization in ML program: ML algorithm의 non-uniform parameter convergence 특성을 활용하여 slower-to-converge parameter를 먼저 prioritize해서 progress per iteration of ML algorithm을 개선하고 결국 model convergence를 확보할때까지 걸리는 iteration수를 감소시킬 수 있다. (select parameters Aj with probability proportional to their squared rate of change,  (Aj(t – 1) – Aj(t – 2))2 + ε, where ε is a small constant that ensures  that stationary parameters still have a small chance to be selected.)

- balancing workloads in ML program: machines들간의 synchronization(exchange of parameter updates)을 위해 모두 stop해야하는 구간이 발생한다. 각각의 machine의 balanced workload를 확보하고 특히, 현실의 cluster에서 발생할 수 있는 machine의 속도를 저하 문제들(changing data center temperature, machine failures, background jobs, other users, etc)에 대응이 필요하다. 

  - "slow-worker agnosticism": system takes direct advantage of the iterative-convergent nature of ML algorithms and allows faster workers to repeat their updates while waiting for slow workers to catch up
  - use bounded-asynchronous execution (synchronous MapReduce-style execution과는 반대)를 활용한다.

- SAP(structure aware parallelization): scheduling + prioritization + load balancing 이 세 가지를 통합하여 SAP라는 single programmable abstraction을 만든다. 

  - SAP를 통해:

    1. prioritize parameters to speed up convergence
    2. perform dependency checks on the prioritized parameters and schedule them into independent subset
    3. load-balance the independent subsets across the worker machines

    SAP는 심플한 MapReduce와 같은 programming interface를 제시해서 다음 세 가지 functions를 구현한다: 

    ​		schedule(): small number of parameters를 prioritize하고 이들이 dependency check에 		expose되게 한다,

    ​		push(): update function 역할 실행, 

    ​		pull(): aggregation function 역할 실행

    수학적으로 prove할 수 있는데 다음과 같은 theorem들로 SAP의 성능이 확인된다:

    1. SAP is close to ideal execution - convergence correctness도 확보하고, convergence per iteration도 개선한다.
    2. SAP slow-worker agnosticism improves convergence progress per iteration - SAP slow-worker agnositicism을 통해 ML program의 convergence가 accelerate된다.

#### how to bridge computation with inter-machine communication

- BSP (Bulk Synchronous Parallel) 와 asynchronous bridging models 각각 다른 challenge들이 있다. BSP을 적용하면 iteration throughput에 ideal P-fold increase를 확보하기 어렵고, asynchronous방식을 적용하면  sequential ML program에서와 같이 ideal progress per iteration을 유지하기 어려워진다. 

- 하나 promising solution으로 SSP(stale synchronous parallel)방식이 있다. SSP는 bounded asynchronous 방식으로 지정된 제한(threshold)까지만 asynchronous 방식으로 진행되는 방법이다. (SSP는 BSP가 더 포괄적이게 개선된 버젼으로 생각하면 된다)

- ##### SSP(Stale Synchronous Parallel)

  programming interface는 BSP와 비슷하다. 동작 방식은 다음과 같다:

  P개의 parallel worker들이 update과 aggregation과 같은 ML computation을 iterative하게 수행하는데, 각 iteration의 끝에서 SSP worker가 본인의 일은 끝났다고 signal을 보인다. 이때 BSP라면 sychronization barrier가 enact되어서 inter communication이 수행되겠지만, SSP에서는 이 barrier를 enact하는것 대신에 worker들을 각자 상황에 따라서 stop하거나 proceed하도록 허락한다. SSP will stop a worker if it is more than s iterations ahead of any other worker, where s=staleness threshold.

  특정 횟수의 iteration 동안에는 더 빠른 worker들이 먼저 진행나아갈 수 있도록 허용하여 synchronization overhead를 완화한다. 이 특정 횟수를 넘기면, worker들 모두 쉬어야한다. Worker들이 data의 cached version으로 operate하고 각 작업 사이클(task cycle)의 끝에서 변경점들을 commit하기 때문에, 다른 worker들이 오래된(stale) data로 operate하게 된다. 

  (Every worker machine keeps an  iteration counter t, and a local view of the model parameters A. SSP worker machines “commit” their updates Δ, and then invoke  a “clock()” function that: 

  ① signals that their iteration has ended,  

  ② increments their iteration counter t, and 

  ③ informs the SSP  system to start communicating Δ to other machines, so they can update their local views of A. 

  This clock() is analogous to BSP’s  synchronization barrier, but is different in that updates from one  worker do not need to be immediately communicated to other  workers—as a consequence, workers may proceed even if they  have only received a partial subset of the updates. This means that the local views of A can become stale, if some updates have not been received yet.)

  ![SSP](C:\SJL\스터디_분산ML_system\figures\SSP.PNG)

  SSP를 구현할때에 다음과 같은 bounded staleness condition들이 존재한다:

  - bounded clock difference
  - timestamped updates
  - model state guarantees
  - read-my-writes

  

  **장점**- SSP는 any pair of workers사이의 maximum staleness를 threshold s로 제한하기때문에, data parallel 또는 model parallel방식에서 모두 strong theoretical model convergence가 guarantee된다.

  **단점**- staleness가 너무 높아지면(slow down하는 machine의 비중이 너무 커지는 경우 발생), convergence rate이 빠르게 deteriorate한다. 

  SSP 방식으로 운영되는 ML program(data parallel & model parallel 방식 모두)은 다음과 같은 두 개의 complementary theorem을 기반으로 near-ideal convergence progress per iteration을 확보할 수 있다.

  ###### SSP data parallel convergence theorem:

  BSP와 동일하게 correctness가 보장된다. 그러나 현실에서 적용할때에는 staleness와 asynchrony를 최소로 유지하는것이 매우 중요하다. complex production environment에서는 other tasks나 user등으로 인해 machine이 temporarily slow down할 수 있고 maximum staleness와 staleness variance가 너무 커져버리는 문제가 발생할 수 있다.

  ###### SSP model parallel asymptotic consistency:

  global view of parameter A가 결국 converge될것이고, stale local worker view of parameter또한 global view A로 converge될것이다라는 것을 말한다. 그리고 이렇게 converge된 값이 optimal solution이 될것이다라는 것을 이야기한다.

#### how to communicate between machines

Distributed processing에서 communication은 performance와 scalability를 설정하는 중요한 요소이다. Updates를 어떤 순서로 communicate해야하는지 설정되어야한다. SSP의 경우에는 update가 "s iteration보다만 더 늦게 도착하지 않으면 되는" 특성이 있기때문에 이를 설정하는 spectrum(or design space)이 넓다.

Machine들간에 data를 spread하고 exchange되는 data량을 절약하기위해 다음과 같은 몇가지 communication 관리 전략을 활용해야한다:

- Continuous communication 

  SSP를 적용할때에도 각 iteration이 끝날때(SSP clock() 명령이 호출될때)에 모든 inter-machine communication이 발생하기때문에, 그 사이동안에는 network이 idle하게 낭비되고 burst of communication이 발생하게된다. (giga~terabytes of magnitude required) burst of communication은 synchronization delay를 발생시킨다. In order to prevent burst of communication over the network (network상에서 communication burst를 방지하기 위해, 지속적인 communication이 활용된다.) 

  SSP가 구현될때에 rate limiter를 통해서 continuous communication을 수행할 수 있다. outgoing communication이 queue up되는데, next in line을 send out 하기전에 이전 communication이 완료되기까지 기다린다. Continuous communication은 ML algorithm이 data parallel방식이든 model parallel방식이든 SSP의 bounded staleness condition을 유지하기때문에 worst-case convergence progress per iteration guarantee를 보장한다.

  또한 managed communication자체가 synchronization delay를 감소시키고 iteration throughput을 개선하고 progress per iteration도 개선하기때문에 overall convergence에 small (two or three-fold) speedup을 제공한다.

- WFBP(Wait-free Backpropagation)

  neural network가 layers로 구성되어있고 top layer가 contain 하고 있는 parameter는 total computation에 작은 portion만 차지한다는 사실을 활용한다. 

  "After performing  back-propagation on the top layers, the system will communicate  their parameters while performing back-propagation on the bottom  layers. This spreads the computation and communication out in an  optimal fashion, in essence “overlapping 90% computation with 90%  communication.”"

  (WFBP exploits the neural network structure by already sending out the parameter updates of the top layers while still computing the updates for the lower layers, hence hiding most of the communication latency.)

  neural networks가 layers로 구성된다. layers의 훈련과정은 back-propagation gradient descent algorithm을 사용하며 high sequential 하다. Top layers가 개부분의 parameter를 포함하지만 전체 computation에는 작은 부분만 기여하고있어서 WFBP(Wait-free Backpropatation)이 proposed됬다. WFBP는 lower layer를 위한 update를 계산하는 동안에 미리 top layer를 위한 update를 전송하면서 neural network structure를 exploit하여 거의 대부분의 communication latency를 숨길 수 있다. 

- hybrid communication

  WFBP가 communication overhead를 줄여주지는 못하기때문에, hybrid communication을 적용한다. (e.g., Parameter Servers + Sufficient Factor Broadcasting = choose the best communication method depending on the sparsity of the parameter tensor)

  WFBP는 communication overhead를 감소시키지 않기때문에, HybComm(hybrid communication)이 propose되었다. PS(Parameter Server)를 SFB(Sufficient Factor Broadcasting)과 함께 통합하는 것인데, parameter tensor의 sparsity에 따라서 가장 좋은 communication method를 선택한다.

- Update prioritization

  Convergence에 가장 많이 contribute하는 update function (또는 part of it)에 focus하는 기준으로 available bandwidth를 prioritize하는것이다. 이 방식은 SAP와 관련되어있다. SAP가 더 중요한 parameter들로 computation을 prioritize하는 동안, update prioritization은 changes to these important parameters가 빠르게 다른 worker machine들에게 propagate되도록 한다. (changes to these important parameters가 바로 반영이 되도록) 다음 두 가지 전략을 사용한다:

  - absolute magnitude prioritization
  - relative magnitude prioritization

  empirically, SSP + continuous communication위에다가 진행하는 prioritization strategies는 25%의 speedup을 확보했다.

- parameter storage and communication topologies

  network에서 model parameter를 어떻게 place할지(parameter storage) + parameter updates를 communicate할 network routes를 제어하는 방법인데, 이 두가지 요소는 어떤 communication topologies를 사용할지에 큰 영향을 끼친다.

  Model parameter들을 store하는 두 가지 paradigm이 존재한다:

  ![](C:\SJL\스터디_분산ML_system\figures\centralized_decentralized_parameter_storage.PNG)

  - centralized storage:

    ![master_slave_topology](C:\SJL\스터디_분산ML_system\figures\master-slave_nework_topology.PNG)

  - decentralized storage:

    ![P2P_network_topology](C:\SJL\스터디_분산ML_system\figures\P2P_network_topology.PNG)

    P2P network에서는 all worker들이 각자 communicate하기때문에 O(P^2) message가 전송되어 master-slave topology에서의 O(P)보다 훨씬 더 많은 message가 전송되어 보이지만, P2P network에서 update가 compact 또는 compressible structure이여서 low rank matrix로 compress한후 represent하는 방식으로 communicate된다면 master-slave보다도 더 작은 message로 communication이 가능해진다. (master-slave may not admit compression (because the messages consist of the actual parameters, not the compressible updates))

  - Halton sequence:

    ![Halton_topology](C:\SJL\스터디_분산ML_system\figures\Halton_network_topology.PNG)

    Workers can reach any other worker by routing  messages through intermediate nodes. For example, the routing  path 1→2→5→6 is one way to send a message from worker 1 to  worker 6. The intermediate nodes can combine messages meant for  the same destination, thus reducing the number of messages per  iteration (and further reducing network load). However, one drawback to the Halton sequence topology is that routing increases the  time taken for messages to reach their destination, which raises the  average staleness of parameters under the SSP bridging model.(For  example, the message from worker 1 to worker 6 would be three  iterations stale.)

    Halton은 매우 큰 cluster network을 위해 적절한 network topology이다.

#### what to communicate

- Early aggregation

  Use of aggregate (before transmission over the network) - F(aggregate function)의  additive structure를 활용해서 update message 의 사이즈를 감소시킨다. 특히, centralized parameter storage paradigm에서 early aggregation is preferred for communication of full parameters from servers to workers.

- SFB (Sufficient Factor Broadcasting)

  MPM (matrix-parametrized models) - matrix-structured parameters를 가진 ML programs (e.g., multiclass logistic regression(MLR), neural networks(NN), distance metric learning(DML), sparse coding)

  MPM에서 각 update는 low-rank matrix이며 small vectors로 factor될 수 있다. 여기서 small vectors가 sufficient factors이고, network상에서 transmit하기에 더 cheap하다. MPM들의 sufficient factor property를 활용하는 방법이 SFB이다. 

  Full broadcast를 하기는 보통 communication volume때문에 금지되기때문에, SFB (Sufficient Factor Broadcasting)이 propose되어서 communication overhead를 감소시킨다. SFB의 parameter matrix는 'sufficient factor'로 분해(decompose)되는데, 즉 2개의 vector만으로도 update matrix를 reconstruct하는 것이다. SFB는 이렇게 딱 적당한 factors만 broadcast하고 worker들이 updates를 reconstruct하게 한다. 그리고 다른 model들은 communication 수준을 less frequent한 synchronization points로 줄이고 각각의 individual model들이 temporarily diverge하도록 허락한다. 

  SFB 활용 전략을 적용하는 방법:

  Instead of transmitting Δ = i bi ci T (total size KD) between workers,  we can instead transmit the individual vectors bi and ci (total size  S(K + D), where S is the number of data samples processed in the current iteration), and reconstruct the update Δ at the destination machine. 

  SFB 전략은 decentralized storage paradigm에 적절하다. (where only updates are transmitted between workers) SSP bridging model에서 더욱 강력하며 특히 Halton sequence와 같은 topologies로 update의 staleness를 증가시키고 그 대신 lower bandwidth usage를 사용할 때에 유용하다. 

  Centralized parameter storage paradigm에서도 활용될 수 있는데, worker에서부터 server까지 전송하는 transmission에 적용될 수 있다. (from server to worker 방향으로는 full matrix를 전송하기때문에 sufficient factor property를 활용할 수 없다.)

  더 높은 update communication cost와 더 낮은 staleness는 ML program니 network communication을 위해 기다리는 시간을 증가시킨다. 어떤 staleness 수준에서도 SFB는 훨씬 더 적은 network waiting을 요구한다. (b/c SFBs are much smaller than full matrices in FMS(Full matrix synchronization))

  Computation time for SFB is slightly longer than for FMS because:

  ① update matrices  Δ must be reconstructed on each SFB worker, and 

  ② SFB requires a  few more iterations for convergence than FMS, due to slightly higher average parameter staleness compared with FMS. Overall, SFB’s  reduction in network waiting time far surpasses the added computation time, and hence SFB outperforms FMS.

  in some situations, SFB와 FMS를 함께 사용한다. 즉, naturally call for a hybrid of SFB and full updates transmission) 예를 들면, deep learning using convolutional neural networks인데, top layer는 보통 fully connected & using matrix parameters containing millions of elements이고, bottom layer는 convolutional & involving tiny matrices with at most a few hundred elements. 이 경우에는 top layer에는: SFB를 top layer의 updates에 apply하고, bottom layer에는: aggregate(sum) bottom layers' updates before transmission

  **Gossip Learning??**

  Gossip Learning은 다음과 같은 아이디어를 기반으로 만들어졌다.  Model이 mobile하고 peer-to-peer network에서 독립적인 random walk를 수행한다. 이런 아이디어는 data- 그리고 model-parallel processing framework을 형성하기때문에, 각각의 model이 다르게 생성 및 성장되어서 ensembling technique를 통해 통합되어야한다. Gossip learning에서는 current model과 previous visitor들의 limited cache를 통합하면서 nodes에 지속적으로 이런 현상이 발생한다. 



### Challenges & 보완방안

#### Performance

Adding additional resources를 통해 training time을 줄이는 대신 total aggregate processing time과 energy consumption이 증가하면서 전체적인 효율은 떨어진다. TensorFlow를 통해 distributed 방식으로 GPU를 사용하는 경우에 efficiency가 75%보다 낮은 수준이 확인되는 수준이다.

Distributed system을 통해 benchmark으로 linear speedups이 확인되는 synchronous SGD-based frameworks의 경우 performance의 비효율성 문제가 덜한 편이지만, 이  benchmark test에는 몇백개의 machine들이 사용된다. 앞으로 더 현실적인 문제와 상황에 맞는 workload optimization이나 system architecture에 대한 연구가 필요하다.

#### Fault tolerance

Parameter server 방식보다 synchronous AllReduce-based approach이 더 scale잘 되지만, fault-tolerance가 매우 부족하다는 단점이 있다. 그래서 sing machine의 failure가 entire training process를 block해버리는 문제에 매우 취약하다. 

High Performance Computing기반의 pattern인 MPI나 NCCL은 fault-tolerance특성이 아얘 없다. Checkpoint를 활용하는 등, 이 단점을 보완하는 부분이 있긴하지만, 적용될 수 있는 production-ready solution은 아직 없다. 

각 node의 failure probability를 감소시킬 수 있는 방법들이 있긴하지만 hardware에 아주 specific한 adjustment가 필요해서 매우 비싸고 비현실적인 방안이다.

Asynchronous implementation은 fault-tolerance의 부재로 인한 영향을 받지않는다. Asynchronous는 전체 performance가 느려짐에도 불구하고 straggling(slow-running) and failing nodes를 tolerate하기위해 만들어진 방식이기때문에, 사용자는 fault-tolerance를 선택할지 아니면 slow performance를 인내할지, 두가지 constraints중에 하나는 어쩔 수 없이 받아들여야하는 상황이다. 

앞으로 fault-tolerant한 AllReduce방식과 같이 performance와 fault-tolerance를 모두 가진 방법을 찾아야한다.

#### Privacy

Privacy-sensitive data를 다룰때에는 distributed ensemble model을 사용한다. (allows perfect separation of training data subsets, with drawbacks that a method needs to be found that properly balances each trained model's output for an unbiased result)

Federated learning이 privacy-sensitive data를 다룰때 data를 local & confidential하게 유지해준다. 

#### Portability

Neural network의 훈련을 위해 다양한 libraries, framework가 존재하지만, 한번 훈련이 진행되고나서는 계속 그 framework 사용을 유지하게 된다. 

각 framework이 custom format을 사용해서 results를 저장한다. (Tensor$ow [2] uses a SavedModel directory, which includes a protocol bu#er de!ning the whole computation graph. Ca#e [78] also uses a binary protocol bu#er for storing saved models, but with a custom schema. Theano [18] uses pickle to serialize models represented by Python objects, and PyTorch [118] has a built-in save method that serializes to a custom ASCII or binary format)

그리고 machine learning이 요구하는 heavy computation을 수행하는 customized hardware(ASIC as TPU)가 더 많아지면서 diversification이 커지고있다. This diversification makes it more difficult to make sure that your trained model can run on any of theses hardware platforms.

framework들간의 이동이 더 편리해질 수 있도록 개발되고 있는 것은 framework independent specifications to define machine learning models and computation graphs. ONNX(Open Neural Network Exchange) format defines a protocol buffer schema that defines an extensible computation graph model, as well as definitions for standard operators and data types.
