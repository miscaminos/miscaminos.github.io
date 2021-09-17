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



## 1. binary classification (이진분류)

실습에서는 '5-감지기'를 만들어서 이진분류 모델을 사용한다. (기존 MNIST data set에서 5에 해당하는 target값만 따로 추출해서 target vector를 생성하면 됨)

분류모델을 하나 생성하기위해 sci-kit learn의 SGD (확률적 경사 하강법)을 사용한다. SGD는 큰 data set을 효율적으로 처리하는 장점을 가지고있다. SGD가 한번에 하나씩 훈련 샘플을 독립적으로 처리하기때문이다. (그래서 SGD가 온라인 학습에 잘 맞음)

참고: sci-kit learn 웹사이트에서 SGD Classifier의 parameter들을 설명해준다. SGDClassifier를 사용할때에 참고하면 좋다. <https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.SGDClassifier.html>

SGD Classifier 클래스 인스턴스 생성시 주요 parameter는:

- max_iter: max number of passes over training data (epoch와 비슷함)
- epoch 횟수 - The maximum number of passes over the training data (aka epochs). It only impacts the behavior in the fit method, and not the partial_fit method.
- tol: The stopping criterion. If it is not None, training will stop when (loss > best_loss - tol) for n_iter_no_change consecutive epochs
- random_state: designate random state with number 42

SGD Classifier 모델 훈련
SGD training 방식으로 linear classifier(SVM, logistic regression,등) 수행
기본적으로 SVM(Soft Vector Machine)을 경사하강법(SGD)으로 풀어낸다
```
from sklearn.linear_model import SGDClassifier

sgd_clf = SGDClassifier(max_iter=1000, tol=1e-3, random_state=42)
sgd_clf.fit(X_train, y_train_5) #max_iter 기본값은 1000, tol 기본값은 1e-3
```


SGD Classifier 모델 검증

교차검증

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

위와같은 수동작업 대신 코드 한줄로 cross-val 수행가능
```
cross_val_score(sgd_clf, X_train, y_train_5, cv=skfolds)
```

accuracy외에 classification 문제의 결과를 평가하기위해 오차행렬 사용
```
from sklearn.model_selection import cross_val_predict

y_train_pred = cross_val_predict(sgd_clf, X_train, y_train_5, cv=3)

from sklearn.metrics import confusion_matrix #오차행렬=confusion_matrix

confusion_matrix(y_train_5, y_train_pred)
y_train_perfect_predictions = y_train_5  # 완변한척 하자
confusion_matrix(y_train_5, y_train_perfect_predictions)
```
임의로 완벽한 matrix를 생성해서 confusion matrix의 구성 구조 할 수 있다.

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

y_scores = sgd_clf.decision_function([some_digit])

threshold = 0 #임계값 지정
y_some_digit_pred = (y_scores > threshold)

```

## 2. ROC 곡선
ROC: 거짓 양성률 FPR vs. 참 양성률 TPR

FPR = FP/(FP+TN)
```
from sklearn.metrics import roc_curve

fpr, tpr, thresholds = roc_curve(y_train_5, y_scores)

def plot_roc_curve(fpr, tpr, label=None):
    plt.plot(fpr, tpr, linewidth=2, label=label)
    plt.plot([0, 1], [0, 1], 'k--') # 대각 점선
    plt.axis([0, 1, 0, 1])                                    # Not shown in the book
    plt.xlabel('False Positive Rate (Fall-Out)', fontsize=16) # Not shown
    plt.ylabel('True Positive Rate (Recall)', fontsize=16)    # Not shown
    plt.grid(True)                                            # Not shown

plt.figure(figsize=(8, 6))                                    # Not shown
plot_roc_curve(fpr, tpr)
fpr_90 = fpr[np.argmax(tpr >= recall_90_precision)]           # Not shown
plt.plot([fpr_90, fpr_90], [0., recall_90_precision], "r:")   # Not shown
plt.plot([0.0, fpr_90], [recall_90_precision, recall_90_precision], "r:")  # Not shown
plt.plot([fpr_90], [recall_90_precision], "ro")               # Not shown
save_fig("roc_curve_plot")                                    # Not shown
plt.show()
```

```
from sklearn.metrics import roc_auc_score

roc_auc_score(y_train_5, y_scores)
```







