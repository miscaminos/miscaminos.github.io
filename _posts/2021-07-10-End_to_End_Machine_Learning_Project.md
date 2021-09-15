---
layout: post           									# (require) default post layout
title: "Machine-Learning-Project-End-to-End"            # (require) a string title
date: 2021-07-10       									# (require) a post date
categories: [machineLearning]   # (custom) some categories, but makesure these categories already exists inside path of `category/`
tags: [MLProject]           	# (custom) tags only for meta `property="article:tag"`
---



Hands on Machine Learning(chapter02)

# Machine Learning Project End to End

 Machine Learning project 주요 steps:

1. 큰 그림을 본다 .
2. 데이터를 구한다.
3. 데이터로부터 통찰을 얻기위해 탐색하고 시각화한다.
4. 머신러닝 알고리즘을 사용하기위해 데이터를 준비/처리한다.
5. 모델을 선택하고 훈련시킨다.
6. 모델을 상세하게 튜닝한다.
7. 솔류션을 제시한다.
8. 시스템을 론칭하고 모니터링하고 유지 보수한다.

<br>

캘리포니아 주택 가격 예측 문제를 tackle한다고 가정하고 step들을 순서대로 진행하면서 어떤 질문들을 고민하고 task들을 수행하는지 알아보았다.

<br>

<br>

## 1. 큰 그림 보기

### 문제 정의

문제 정의를 위해 문제의 배경을 파악한다.

1) "비즈니스의 목적이 정확히 무엇인가?" - 모델을 만드는것이 최종 목적은 아닐것이다. 회사는 이 모델을 사용해서 어떻게 이익을 얻으려는 걸까?

문제를 어떻게 구성할 지, 어떤 알고리즘을 선택할지, 모델 평가에 어떤 성능 지표를 사용할지, 모델 tuning을 위해 얼마나 노력과 시간을 투자할지 결정하는데에 중요한 starting point이다.

2) "현재 솔루션은 어떻게 구성되어있나?" - 문제 해결 방법에 대한 정보도 얻고, 참고 성능으로도 사용할 수 있다.

현재 솔루션이 어떻게 구성되어있는지는 현재 파이프라인이 어떻게 구성되어있는지 살펴보면 알 수 있다. 이 파이프라인 시스템에서 나는 어느 component를 담당하게 되는지 그 앞/뒤 component에 어떤 신호들이 투입되는지 파악해야 한다.

<hr>

**note:**

**파이프라인** : 데이터 처리 component들이 연속되어 있는것을 데이터 pipeline이라고 한다. Component들의 사이에 인터페이스로 데이터 저장소를 두고 각 component가 추출하고 처리한 대량의 데이터를 저장한다. component들은 보통 비동기적으로 동작하여서 하나의 component가 process된 데이터를 저장소에 저장하면, 일정시간 후, 파이프라인의 다음 component가 데이터를 추출해서 자신의 출력 결과를 생성해낸다. 

 이렇게 각각의 component는 독립적으로 운영될수 있고, 각 component를 담당하는 팀은 각자의 component에 집중할 수 있다.

한 component가 다운되어도 하위 component는 가장 최신 출력을 사용해서 task를 지속하면 되지만, 적절한 모니터링이 진행되어야 고장난 component를 빠르게 인식하고 전체 시스템의 성능이 떨어지지 않도록 유지할 수 있다. 

<hr>

구체적으로 문제를 정의하기위해 어떤 모델로 어떤 시스템을 만들지 설계한다. 이번 문제와 같이 집값을 예측하는 문제는 (median housing value가) label된 훈련 샘플을 사용하기때문에 지도학습이 수행되어야 한다.

주어진 데이터 세트를 보면 label값을 예측하기위해 여러개의 특성이 사용되기때문에 다변량 회귀 (multivariate regression) 문제이다. 

<hr>

**note:**

**회귀의 종류:**

다변량 회귀  vs. 단변량 회귀 (multivariate regression vs. univariate regression):

- regression model을 구성하는 종속 variable이 1개일때: 단변량(univariate)

- 종속 variable이 2개 이상일때: 다변량(multivariate)

이름 그대로 regression model의 종속 variable이 하나인지, 여러개인지를 구분하는것이다.

종속 variable이 여러개인 다변량분석의 경우 종속 - 독립 간의 인과관계를 분석하는 회귀만이 아니라, 종속 variable들 간의 상관관계나 개체들의 분류를 위한 다른 방식의 분석도 진행할 수 있다.  

그래서 다변량 분석은 좀 더 넓은 범위로 다음과 같은 분석을 의미하기도한다:

​	(1) 변수들간의 인과관계 (causal relationship)을 분석 : 회귀분석, 분산분석

​	(2) 변수들간의 상관관계 (correlation)를 이용하여 데이터의 차원을 축소
​	 : 주성분분석, 요인분석

​	(3) 개체들의 유사성에 의해 개체를 분류 : 판별분석, 군집분석



여기서 잠깐, 햇갈릴수도 있는 비슷한 이름의 다중 회귀 vs. 단순 회귀 (simple regression vs. multiple regression) term들은 종속 변수가 아닌 독립변수가 하나 또는 여러개인 경우를 구분하기위해 사용된다. 

- 독립 변수가 1개일때: 단순 회귀(simple regression model)

- 독립 변수가 2개 이상일때: 다중 회귀(multiple regression model)



**표기법**

공식에서 x는 특성, h(x)는 예측값,  y는 target값

소문자: bold는 vector, 일반은 scalar

대문자: bold는 matrix

note: vectors & matrices - 

vector로 하나의 샘플의 특성들을 나열할 수 있다.

matrix안에 여러 샘플을 나열할 수 있다. 대부분 각 행이 하나의 샘플의 특성을 갖게한다. (그러나 열을 따라 샘플이 놓여질때도 있음)

<hr>

**평가지표**

다변량 회귀문제라는것이 파악되면, 문제의 솔루션을 어떤 성능 지표로 측정할지도 파악해야한다. 보통 회귀문제의 성능 지표는 RMSE(Root Mean Square Error)평균 제곱 오차이다.

종종 MAE(Mean Absolute Error)평균 절대 오차 (aka Mean Absolute Deviation 평균 절대 편차)도 사용되는데, RMSE가 MAE보다 조금 더 이상치에 민감하다. (예측값의 vector와 target값의 vector 사이의 거리를 잴때에 norm 지수가 클수록 큰값의 원소에 치우치고 작은 값은 무시되기때문에)

다음단계로 데이터를 가져와서 탐색 및 처리하기전에, 지금까지 만든 가정(assumptions)를 쭉 나열해서 검사해봐야한다. 이런 과정에서 예측의 결과로 원하는 것이 정확한 집 값인지, 아니면 단순히 집값의 category (저렴, 보통, 고가, 등)인지를 파악하는 등, 어떤 형태의 학습이 적합한지를 점검한다.

#### 주요 모델평가지표
회귀 모델 평가지표:

- mse
  
- rmse
  
- mae
  
- mpe
  
- rmpe
  
- R square
  
- Adjusted R square
  
- AIC, BIC

분류모델 평가지표:  
- Accuracy

- Precision

- Recall

- F1 score

- Fall-out

- Log loss

- ROC curve

  <br>

  <br>

## 2. 데이터 가져오기

실습에서는 .tgz 형태의 파일로 데이터가 주어졌다. 

tarfile 라이브러리를 import해서 tgz파일을 열고 데이터내용을 추출해서 pandas를 통해 csv파일로 저장할 수 있다. csv파일로 저장된 데이터는 pandas의 dataframe으로 조회하여 pandas의 다양한 도구들을 사용해서 데이터를 탐색하고 처리할 수 있다.

info(), describe(), 등 함수를 통해 dataframe의 요약정보를 파악하고

matplotlib.pyplot을 import해서 hist()를 통해 각 특성의 histogram을 그리고  분포를 탐색할수있다. 각 특성의 histogram을 보면,

-특성 값의 scale을 파악하고, 만약 scale이 서로 많이 다르다면, 특성 scaling을  어떻게 진행하는게 좋을지 고민해본다.

-최대값, 최소값 한정이 존재하여 예측할 수 있는 값에 한계를 두지는 않는지 확인한다.

-한계를 넘은 구역에서 예측값을 도출해야하면 한계 밖의 구역에 대한 정확한 label 데이터를 구하거나 또는 훈련세트에서 한계 이상의 구역을 제거한다.

-histogram의 꼬리가 두껍다면 (왼쪽 또는 오른쪽으로 분포의 중심이 쏠려있다면) 일부 머신러닝 알고리즘에서 패턴을 찾기 어렵기때문에, 좀 더 종모양(bell-shaped)의 분포가 될 수 있도록 변형시킨다.

#### 테스트 세트 확보

데이터를 상세하게 탐색하기 전에, 먼저 테스트 세트를 따로 분리해 놓아야한다. Data snooping 편향을 방지하기 위함이다. 문제를 해결할 모델의 성능을 평가할때에는 들여다보지 않은 데이터를 기반으로 모델의 일반화 성능을 평가해보아야한다. 그래야 편향의 영향을 받지않은 모델을 구할 수 있고, 결국 시스템을 론칭했을때에 더 나은 성능을 가질 수 있다.

보통 데이터의 20%를 따로 뽑아두지만, 무작위로 20%를 뽑으면 sampling 편향이 생길 수 있는 위험이 존재한다. 그래서 계층적 샘플링 (stratified sampling)을 통해 테스트 세트가 전체 모수를 대표하도록 각 계층에서 올바은 수의 샘플을 추출한다. 이를 위해 sci-kit learn의 StratifiedShuffleSplit 클래스를 import해서 사용한다. 

만약 데이터가 지속적으로 갱신되고, 훈련에 사용되었던 데이터가 테스트 세트에 포함되어버리는 것을 방지하기위해서 일반적으로 crc32를 사용한다. 샘플의 식별자를 사용해서 테스트 세트로 보낼지 말지를 정하는 방법이다. 만약 갖고있는 데이터셋에 식별자 컬럼이 없다면, index를 식별자로 사용할 수 있다. 

인덱스를 고유 식별자로 사용할때에는 새 데이터는 반드시 데이터셋의 끝에 추가되어야하고, 어떤 행도 삭제되어서는 안되는 restriction이 생기게된다. 그래서 고유 식별자 (샘플마다 다른 값을 가지고, 변경이 없는 안정적인 값을 가진 특성을 기반으로 고유 식별자를 만들 수 있음) 를 대신 사용 할 수도 있다.

<br>

<br>

## 3. 데이터 탐색

California 주의 주택 위치를 latitude, longitude 값으로 알고있기때문에, scatter plot을 그려서 데이터를 시각화할 수 있다. 더 현실성이 있도록 California state 지도를 확보하여 지도 위해 scatter plot을 그리도록 한다.

주택의 위치가 많이 겹치는 구역이 발생하면 밀집도를 구분하기 어렵기때문에, 이때에는 alpha값을 사용하여 밀집된 지역을 구분해서 시각화 할 수 있다. 

더 개선된 방법으로는 scatter plot의 point의 색깔을 median house value값을 표현하도록 하고, scatter point의 circular 크기를 population 밀집도를 표현하도록 할 수 있다. 

**상관관계**

특성간의 standard correlation coefficient를 corr() 함수를 이용해서 확인할 수 있다. 상관관계의 범위는 (-1,1). 상관관계가 0일때에는 두 특성이 완적 독립적이거나 아니면 비선형관계를 갖고있다.

median_house_value는 median_income 특성과 강한 양의 상관 관계를  갖고있음이 확인된다. 

**특성조합**

특성간의 연산으로 새로운 특성을 도출해 낼 수 있다. 새롭게 생성한 특성이 강한 상관 관계를 가지기도 한다. 데이터에 대한 통찰을 얻기위해 다양한 시도를 해볼 수 있다. 

<br>

<br>

## 4. ML algorithm을 위한 데이터 준비/처리

**데이터 정제**

sci-kit learn의 Imputer로 누락값을 처리할 수 있다. 누락값을 처리하는 방법은 일반적으로 3가지가있다

- null값이 있는 해당 구역 삭제

- null값을 가진 특성 전체를 삭제

- null값을 어떤 값으로 대체한다. (0, mean, median, etc)

  

**텍스트로 된 범주형 특성 처리**

문자로 구성된 categorical 특성을 처리하기위해 숫자로된 범주형 값을 one-hot vector로 바꿔주는 OneHotEncoder를 사용한다.

텍스트 category를 숫자 category로, 숫자 category를 one-hot vector로 바꿔주는 이 두가지 변환을 CategoricalEncoder를 사용하여 한번에 처리할 수 있다.

<hr>

**note:**

sci-kit learn의 설계 철학

1. 일관성 - 모든 객체가 일관되고 단순한 인터페이스를 공유한다

   - 추정기(estimator): 데이터셋을 기반으로 일련의 모델 파라미터들을 추정하는 객체 (e.g., imputer) 추정 작업을 fit()을 통해 수행하고, 하나의 매개변수로 하나의 데이터셋만 전달한다.
   - 변환기(transformer): 데이터 셋을 변환하는 추정기를 변환기라고함. 데이터셋을 매개변수로 전달받은 transform() 함수가 수행한다. fit() 후, transform()을 호출하는 것과 동일한 방법으로 fit_transform()을 호출한다.
   - 예측기(predictor): 일부 추정기는 주어진 데이터셋에 대해 예측을 할수 있다. (e.g., LinearRegression 모델) 예측기의 predict() 메서드로 새로운 데이터셋을 받아서 이에 맞는 예측값을 반환한다.

2. 검사 기능 - 모든 추정기의 하이퍼파라미터는 public instance 변수로 직접 접근할 수 있다. (e.g., imputer.strategy) 그리고 모든 추정기의 학습된 모델 파라미터도 underbar를 붙혀서 public instance 변수로 활용된다. (e.g., imputer.statistics_)

3. 클래스 남용 방지 - 데이터셋을 별도의 클래스가 아니라 numpy array나 scipy sparse matrix로 표현

4. 조합성 - 기존의 구성요소를 최대한 재사용한다.

5. 합리적인 기본값 -  기본적으로 시스템이 빠르게 형성될 수 있도록 default로 값들이 설정되어있다.

   *참고* : sci-kit learn에서의 변수 표기 방식 

   .statistics_  <--이렇게 underbar가 있는 속성은 sci-kit learn의 SimpleImputer class가 학습한 결과

   .strategy  <--이렇게 underbar없는 속성은 사용자가 지정한 매계변수/ hyperparameter


<hr>

**사용자 정의 변환기**

sci-kit learn에서 제공하는 변환기외에 사용자가 필요에 의해 정의할 수 있다. 특정 특성을 조합하거나, 특별한 정제 작업이 필요할때에 나만의 변환기를 정의해서 사용할 수 있다.

내가 정의한 변환기가 sci-kit learn의 기능들과 매끄럽게 연동될 수 있도록하려면, fit(), transform(), fit_transform() 함수들을 구현한 Python class를 만들면 된다. (sci-kit learn은 상속이 아닌 duck typing을 지원하기때문) 

내가 정의하는 Python class에 TransformerMixin과 BaseEstimator를 상속하면 특성 변환과 hyperparameter tuning에 필요한 함수들을 얻게된다. 

이렇게 작성한 변환기에 새롭게 조합된 특성을 추가하는것이 머신러닝 알고리즘에 도움이 될지 안될지는 hyperparameter로 쉽게 확인해볼 수 있다. 이렇게 데이터 준비 단계에 대해 hyperparameter를 추가할 수 있는데, 이런 준비단계를 자동화 하면, 더 많은 조합을 자동으로 시도해 볼 수 있고, 최상의 조합을 찾을 가능성을 높일 수 있다. 

*참고:* ColumnTransformer : sci-kit learn.compose의 클래스. 열마다 각다른 transformer를 적용할 수 있다

**특성 scaling**

머신러닝 알고리즘에 입력되는 숫자들이 scale이 많이 다르면 원하는대로 잘 작동하지 못하는 경우가 발생한다. 

모든 특성의 범위가 같도록 만들기위해 min-max scaling과 표준화(standardization)가 많이 사용된다. 

MinMaxScaling은 정규화(normalization)이라고도 부른다. 모든 특성의 값들이 0~1 범위에 들도록 값을 이동하고 scale을 조정한다.  데이터에서 Min값을 뺀 후, Max-Min 차이값으로 나누는 것이다. 

MinMaxScaling도 sci-kit learn의 MinMaxScaler 변환기를 통해 쉽게 적용할 수 있다. 만약 0~1 범위를 원하지 않는다면, feature_range로 매개변수로 범위를 변경할 수 있다.

Standardization 표준화 방법은 데이터에서 평균을 뺀 후, 표준편차로 나누어서 결과 분포의 분산이 1이 되도록 한다. 상한/하한 범위안으로 데이터가 제한되지 못해서 어떤 알고리즘에서는 문제가 될 수 있다. (e.g., 0~1 범위를 벗어나는 경우 신경망에는 적합하지 못함) 그러나 표준화는 이상치의 영향을 덜 받는다. sci-kit learn에서는 표준화를 수행하는 StandardScaler 변환기를 제공한다.

연속형 변수 scaling방법들을 정리해보면 다음과 같다:

– 변수 간 단위가 크게 차이가 날 때 사용 

- StandardScaler – 평균, 표준편차 사용

- RobustScaler – 중앙값, IQR 사용, outlier 영향 최소화

- MinMaxScaler– 최대/최소값이 1,0이 되도록 scaling

– 변수 간 상관관계가 큰 경우에는 변수 선택이나 PCA 등의 차원축소



**변환 pipeline**

데이터 준비과정에서 많은 변환 단계가 순차적으로 수행되어야한다. 이런 경우 sci-kit learn의 Pipeline 클래스를 활용해서 연속된 변환을 순서대로 처리할 수 있다. 

Pipeline은 연속된 단계를 나타내는 이름/추정기 쌍의 목록을 입력받는다. 마지막 단계에는 변환기와 추정기를 모두 사용할 수 있고, 그외에는 모두 변환기여야한다. 

변환기로 반환받는 수치형 컬럼은 numpy array형태이기때문에 pandas의 dataframe형태로 바꾸는 작업을 따로 수행해야한다. 사용자 정의 변환기를 하나 더 생성해서 (e.g., DataFrameSelector라는 이름으로 Python class를 하나 생성한다. BaseEstimator와 TransformerMixin을 상속받아서 이 class안에 fit(), transform() 메소드를 정의한다.)

DataFrameSelector로 필요한 특성을 선택해서 dataframe을 numpy array로 바꾸는 식으로 데이터를 변환시킨다. 수치형의 특성만 가진 dataframe과  categorical 특성을 가진 dataframe을 각각 반환 받는다.

sci-kit learn의 FeatureUnion 클래스를 import해서 이 두 pipeline을 결합시킨다. 머신러닝 알고리즘에 주입할 데이터를 자동으로 정제하고 준비하기 위해 변환 파이프라인이 작성되었다.

<br>

<br>

## 5. 모델 선택 및 훈련

 sci-kit learn의 LinearRegression을 import해서 선형회귀모델을 생성해서 훈련시킬 수 있다. 훈련 데이터로 훈련된 모델에 몇개의 샘플을 넣고 예측값을 확인하거나,  예측값과 타겟값을 비교해볼 수 있다. 

그리고 모델의 오차를 확인하기위해 sci-kit learn의 mean_sequre_error 함수를 사용한다. 전체 훈련 세트에 대한 regression 모델의 RMSE를 측정할 수 있다. 

다른 모델로  DecisionTreeRegressor를 훈련시켜볼 수 있다. 보통 decision tree의 평가 결과는 높게 나온다. 훈련데이터에 과대 적합되었을 가능성이 매우 높다. 과대적합 여부를 검증하고 더 나은 성능의 모델을 확보하기위해 교차검증을 사용한 평가를 진행한다. 

**cross-validation**

k-fold cross-validation가 많이 사용된다. 먼저 훈련세트를 fold라 불리는 10개의 subset으로 무작위로 분할 한다. 그 다음, decision tree model을 10번 훈련하고 평가하는데, 매번 다른 fold를 선택해서 평가하고 나머지 9개 fold는 훈련에 사용하는 것이다. 이렇게 10번의 평가를 진행하고, 10개의 평가 결과 값을 확인할 수 있다. 

실습에서는 이렇게 교차검증이 진행된 결과, decision tree 모델이 선형모델보다 좋지않은 성능을 가지고있다는것이 확인되었다.

더 좋은 성능이 예상되는 다른 모델을 시도해보자면, RandomForestRegressor을 시도해볼 수 있다. 이 모델은 특성을 무작위로 선택해서 많은 decision tree모델을 만들고 그 예측을 평균내는 방식이다. 여러 다른 모델을 모아서 하나의 모델을 만드는 방식인 ensemble 학습중 하나로서, 머신러닝 알고리즘의 성능으 극대화 시킬 수 있는 장점을 가지고있다. 

보통 이렇게 다양한 모델을 시도해보아서 가능성이 있는 2~5개 정도의 모델을 선정하는 것이 목적이다. 

Cross validation의 종류:
- k-fold – 전체 데이터를 k등분 후 k-1개는 train set, 
1개는 validation set 총 k번 만큼 교차검증

- holdout – 일반적인 train, validation, test set 분리

- leave-one-out – 전체 n개의 샘플 데이터셋을 n-1, 1개로 나누어 n번만큼 교차검증

- leave-p-out – 전체 n개의 샘플 데이터셋을 n-p, p개로 나누어 nCp번만큼 교차검증

  <br>

  <br>

## 6. 모델 세부 tuning

가능성이 있는 모델들의 hyperparameter를 조정해서 가장 높은 성능을 가능케하는 조합을 찾아내야한다. Hyperparameter를 tuning하면서 모델을 평가하는 반복적인 작업을 sci-kit learn의 GridSearchCV를 통해 빠르게 진행 할 수 있다. 

탐색하고자하는 hyperparameter와 시도해볼 값을 지정하면된다. 그러면 가능한 모든 hyperparameter 조합에 대해 cross validation을 사용해서 평가해준다. 

GridSearchCV는 비교적 적은 수의 조합을 탐색할때에 적절하다. 만약 hypyerparameter 탐색공간이 커지면 RandomizedSearchCV를 사용하는 것이 더 적절하다. RandomizedSearchCV는 가능한 모든 조합을 시도하는 대신 각 반복마다 hyperparameter에 임의의 수를 대입해서 지정한 횟수만큼 평가한다.

최상의 모델을 분석하면 문제에대한 좋은 통찰을 얻을 수 있다.  GridSearchCV의 best_estimator_ & feature_importances_ 를 호출해서 모델의 정확한 예측을 만들기위해 각 특성의 상대적인 중요도를 확인할 수 있다. 

시스템이 특정한 오차를 만들었다면 왜 그런 문제가 생겼는지 이해하고 문제를 해결하는 방법이 무엇인지 찾아야한다 - 특성 추가 또는 불필요한 특성 제거, 또는 이상치 제외, 등등

**테스트 세트로 시스템 평가**

모델 tuning으로 만족할 만한 모델이 확보가 되면, 테스트 세트에서 최종 모델을 평가한다. 테스트 세트로 평가를 한 결과는 보통 hyperparameter tuning시 확인했던 성능 값보다 조금 낮은 성능이 확인되는것이 일반적이다.



## 7. 솔루션 제시

학습한 것, 한 일과 하지 않은 일, 수립한 가정(assumptions), 시스템 제한 사항등을 포함하여 솔루션을 제시한다. 적절한 도표를 사용해서 데이터분석 결과 시각화하여 솔류션을 report로 정리한다.



## 8. 론칭, 모니터링, 시스템 유지 보수

제품 시스템에 적용하기위한 준비가 진행되어야한다. 입력 데이터 소스를 시스템에 연결하고, 테스트 코드를 작성한다. 입력 데이터의 품질도 평가해야한다.

일정 간격으로 시스템의 실시간 성능을 체킇고, 성능이 저하되었을때 알람을 받을 수 있도록 설정한다. 갑작스런 오작동이나, 성능이 감소되는 상황을 빠르게 파악해서 개선작업을 수행한다. 새로운 데이터를 사용해서 주기적으로 훈련시키지 않으면, 데이터가 오래됨에 따라 보통 모델도 함께 낙후된다. 

가능하면 정기적으로 모델을 훈련시키는 과정을 자동화하는것이 좋다. 자동화 되지않으면 모델을 갱신해야할때에 시스템에 영향을 끼칠 수 있다. 



#### notes: open source data repositories / list of repositories

-UCI machine learning 저장소 (http://archive.ics.uci.edu/ml/)

-Kaggle (http://www.kaggle.com/datasets)

-Amazon AWS data set (http://aws.amazon.com/datasets)

-한국데이터거래소 (https://kdx.kr/main)

-공공데이터포털 (https://www.data.go.kr/)

-(http://dataportals.org/)

-(http://opendatamonitor.edu/)

-(http://quandl.com)

-Wikipedia machine learning data sets lists (https://goo.gl/SJHN2K)

-Quora.com 질문 (http://goo.gl/zDR78y)

-Data set Subreddit (http://www.reddit.com/r/datasets)



<hr>

#### reference

1. Hands on Machine Learning with Scikit-Learn, Keras, and TensorFlow second edition by Aurelien Geron, Haesun Park
