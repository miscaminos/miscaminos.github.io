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
- tol: 
- random_state:

SGD Classifier 모델 훈련



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
    
```



