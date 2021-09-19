---
layout: post           									# (require) default post layout
title: "Classification"            # (require) a string title
date: 2021-07-12       									# (require) a post date
categories: [machineLearning]   # (custom) some categories, but makesure these categories already exists inside path of `category/`
tags: [MLProject]           	# (custom) tags only for meta `property="article:tag"`
---

Hands on Machine Learning(chapter03)

# Classification

Handwritten numbers data set from MNIST를 사용해서 숫자 classification을 수행해본다.

```Python
from sklearn.datasets import fetch_openml
mnist = fetch_openml('mnist_784', version=1, as_frame=False) 
```
as_frame parameter를 True로 지정하면 dataframe 형태로 data를 fetch해올 수 있다.

```Python
X, y = mnist["data"], mnist["target"]
X.shape
y.shape
```
이렇게 X의 shape을 확인해보면, (70000, 784) 임을 확인할 수 있다.
7만개의 sample이 있고, 각 sample이 784(=28x28)가지의 특성(여기에서는 이미지의 pixel)을 가지고 있는것이다.
y의 shape은 (70000, )임으로 7만개의 sample의 target값을 가지고 있다.


## 1. binary classification (이진분류)

실습에서는 '5-감지기'를 만들어서 이진분류 모델을 사용한다. (기존 MNIST data set에서 5에 해당하는 target값만 따로 추출해서 target vector를 생성하면 됨)

```Python
# y_train_5에 train target중 5인 경우만 따로 뽑아서 이진분류를 수행해본다.
y_train_5 = (y_train == 5) #target 값 = True/False
y_test_5 = (y_test == 5)
```

분류모델을 하나 생성하기위해 sci-kit learn의 SGD (확률적 경사 하강법)을 사용한다. SGD는 큰 data set을 효율적으로 처리하는 장점을 가지고있다. SGD가 한번에 하나씩 훈련 샘플을 독립적으로 처리하기때문이다. (그래서 SGD가 온라인 학습에 잘 맞음)

참고: sci-kit learn 웹사이트에서 SGD Classifier의 parameter들을 설명해준다. SGDClassifier를 사용할때에 참고하면 좋다. <https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.SGDClassifier.html>

SGD Classifier 클래스 인스턴스 생성시 주요 parameter는:

- max_iter: (default=1000) max number of passes over training data (epoch와 비슷함) epoch 횟수란? The maximum number of passes over the training data (aka epochs). It only impacts the behavior in the fit method, and not the partial_fit method.
- tol: (default=1e-3) The stopping criterion. If it is not None, training will stop when (loss > best_loss - tol) for n_iter_no_change consecutive epochs
- random_state: (default=None) 종종 designate random state with number 42 사용.

다음과 같이 SGD Classifier 모델을 훈련시키고 훈련된 모델을 통해 예측값을 확인할 수 있다.
(note: SGD training 방식으로 linear classifier(SVM, logistic regression,등) 수행할 수 있다.
기본적으로 SVM(Soft Vector Machine)을 경사하강법(SGD)으로 풀어낸다.)
```Python
from sklearn.linear_model import SGDClassifier

sgd_clf = SGDClassifier(max_iter=1000, tol=1e-3, random_state=42)
sgd_clf.fit(X_train, y_train_5)

some_digit = X[0] #임의로 첫번째 sample을 지정
sgd_clf.predict([some_digit])
```

교차 검증을 통해 SGD Classifier 모델을 검증할 수 있다.

sci-kit learn에서 제공하는 교차검증(cross_val_score() 함수)보다 더 많은 조건으로 제어하기위해서는 교차검증을 직접 구현할 수 있다. 

```Python
from sklearn.model_selection import StratifiedKFold
from sklearn.base import clone

skfolds = StratifiedKFold(n_splits=3, random_state=42)

for train_index, test_index in skfolds.split(X_train, y_train):
    clone_clf = clone(sgd_clf)
    X_train_folds = X_train[train_index]
    y_train_folds = y_train[train_index]
    X_test_folds = X_train[test_index]
    y_test_folds = y_test[test_index]
    
    clone_clf.fit(X_train_folds, y_train_folds)
    y_pred = clone_clf.predict(X_test_fold)
    n_correct = sum(y_pred == y_test_fold)
    print(n_correct / len(y_pred))    
```

위와같은 수동작업 대신 코드 한줄로 cross-val 수행가능하다.
```Python
cross_val_score(sgd_clf, X_train, y_train_5, cv=skfolds)
```
fold가 3개인 k-겹 교차검증을 통해 accuracy를 확인해보면 90% 이상의 높은 수준이 확인된다. 사실 양성 클래스가(숫자가 '5'인 sample의 양이) 전체 dataset의 10% 정도 밖에되지않기 때문에, 이와 같이 불균형한 dataset의 분류를 평가할때에는 accuracy(정확도)보다는 confusion matrix(오차행렬)을 사용한다.

```Python
from sklearn.model_selection import cross_val_predict

y_train_pred = cross_val_predict(sgd_clf, X_train, y_train_5, cv=3)

from sklearn.metrics import confusion_matrix 

# 다음과 같이 임의로 완벽한 matrix를 생성해서 오차행렬 확인
y_train_perfect_predictions = y_train_5
confusion_matrix(y_train_5, y_train_perfect_predictions)
```
output:
array([[54579,     0],
       [    0,  5421]])
       
confusion_matrix의 row는 data의 실제 클래스를 구분하고, col은 data를 예측한 값을 구분한다.

이번 case와 같이 binary class인 경우에는 2 by 2 matrix가 ouput으로 확인되는것이다.

 1st row = 음성 클래스 ('5'가 아닌)
 
 2nd row = 양성 클래스 ('5')
 
 1st col = 음성 클래스를 예측한 값
 
 2nd col = 양성 클래스를 예측한 값 
 

위 코드의 결과를 보면 임의로 분류 결과가 5인 경우 완벽한 분류예측기의 오차행렬은 true positive와 true negative만 >0이고, 나머지 false값듯은 0이라는것을 확인할 수 있다.

```Python
# 실제 예측한 결과의 오차행렬을 확인
confusion_matrix(y_train_5, y_train_pred)
```
output:
array([[53892,   687],
       [ 1891,  3530]])
       
실제 예측 결과를 보면 687개의 sample을 5가 아닌데 5로 잘못 예측한 했고 (false positive),
1891개의 sample을 5인데 5가 아니라고 잘못 예측(false negative)했다는것을 확인할 수 있다.

```
from sklearn.metrics import precision_score, recall_score
from sklearn.metrics import f1_score

# 정밀도
precision_score(y_train_5, y_train_pred)
cm = confusion_matrix(y_train_5, y_train_pred)

# 재현율
recall_score(y_train_5, y_train_pred)

# precision과 recall 지표들을 f1 score 하나의 값으로 표현가능
f1_score(y_train_5, y_train_pred)
```

**정밀도 = (True Positive)/(True Positive + False Positive)**

정밀도는 양성 예측의 정확도를 표현해준다 (즉, 양성으로 예측한것중에 몇개가 정말 양성이였나?)

**재현율 = (True Positive)/(True Positive + False Negative)**

재현율은 분류기가 정확하게 감지한 양성 sample의 비율로 민감도(sensitivity) 또는 진짜 양성 비율(True Positive Rate, TPR) 즉, 진짜로 양성일것들 중에 몇개가 정말 양성인가?를 말해준다.

정밀도와 재현율은 하나의 숫자로 표현하기위해 F1 score라는 지표를 사용한다.
특히, 두개의 분류기를 비교하는 상황에서는 각 분류기의 f1 score를 사용한다.
f1 score은 정밀도와 재현율의 조화 평균 (harmonic mean)이다. 

정밀도와 재현율이 비슷한 분류기에서는 f1 score가 높다. 하지만 경우에 따라 항상 바람직한것은 아니다.

예를 들어 한번의 오답이 안전에 매우 critical한 모델이라면, 재현율은 높으나 오답이 몇개 나오는것보다, 정답이 많이 제외 되더라도(낮은 재현율) 정답만 노출시키는(높은 정밀도) 분류기가 선호될것이다. 

반면, 감시 카메라로 좀도둑을 찾아내는 분류문제의 경우, 훈련된 모델의 재현율이 99%라면 정확도가 30%만 되더라도 사용해볼 만 할것이다. 잘못된 호출이 종종 울려서 번거롭겠지만, 좀도둑을 잡을 수는 있을것이다. 

정밀도와 재현율은 서로 trade off 관계라서 하나가 높으면 다른 하나가 낮아지는 것은 어쩔수 없는것이다.

SGD Classifier가 어떻게 (무엇을 기준으로) 분류를 하는지를 decision_function 함수를 통해 확인해볼 수 있다. decision function은 모델이 각 sample에 주는 점수를 알려준다. threshold (임계값)보다 점수가 더 크면 양성 클래스에 할당하고, 더 작으면 음성 클래스에 할당한다.

```Python
y_scores = sgd_clf.decision_function([some_digit])

threshold = 0 #임계값 지정
y_some_digit_pred = (y_scores > threshold)
y_some_digit_pred
# output: True

threshold = 8000 #임계값 지정
y_some_digit_pred = (y_scores > threshold)
y_some_digit_pred
# output: False
```

위 두 threshold 값에 따라서 반대의 분류 결과를 보인다는것을 확인할 수 있다.

### 2. ROC 곡선
ROC: 거짓 양성률 FPR vs. 참 양성률 TPR

FPR = FP/(FP+TN)
```
from sklearn.metrics import roc_curve

fpr, tpr, thresholds = roc_curve(y_train_5, y_scores)

def plot_roc_curve(fpr, tpr, label=None):
    plt.plot(fpr, tpr, linewidth=2, label=label)
    plt.plot([0, 1], [0, 1], 'k--') # 대각 점선
    plt.axis([0, 1, 0, 1])                                   
    plt.xlabel('False Positive Rate (Fall-Out)', fontsize=16) 
    plt.ylabel('True Positive Rate (Recall)', fontsize=16)    
    plt.grid(True)                                           

plt.figure(figsize=(8, 6))                                    
plot_roc_curve(fpr, tpr)
fpr_90 = fpr[np.argmax(tpr >= recall_90_precision)]          
plt.plot([fpr_90, fpr_90], [0., recall_90_precision], "r:")   
plt.plot([0.0, fpr_90], [recall_90_precision, recall_90_precision], "r:")  
plt.plot([fpr_90], [recall_90_precision], "ro")               
save_fig("roc_curve_plot")                                    
plt.show()
```
output:

![classification_ROC](https://raw.githubusercontent.com/miscaminos/miscaminos.github.io/master/static/img/_posts/classification_ROC.png)

```
from sklearn.metrics import roc_auc_score

roc_auc_score(y_train_5, y_scores)
```







