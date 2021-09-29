Hands on Machine Learning(chapter05)

# Support Vector Machine SVM

 

### 정규방정식을 사용한 선형 회귀

#### 선형 회귀 모델의 예측(vector형태)

![$\hat{\boldsymbol{y}} = \boldsymbol{h_{\theta}}({\boldsymbol{X}}) = \theta^T\cdot\boldsymbol{X}$](https://render.githubusercontent.com/render/math?math=%5Chat%7B%5Cboldsymbol%7By%7D%7D%20%3D%20%5Cboldsymbol%7Bh_%7B%5Ctheta%7D%7D%28%7B%5Cboldsymbol%7BX%7D%7D%29%20%3D%20%5Ctheta%5ET%5Ccdot%5Cboldsymbol%7BX%7D&mode=inline)

![$\hat{\boldsymbol{y}}$](https://render.githubusercontent.com/render/math?math=%5Chat%7B%5Cboldsymbol%7By%7D%7D&mode=inline)은 예측값이다.

![$\boldsymbol{X}$](https://render.githubusercontent.com/render/math?math=%5Cboldsymbol%7BX%7D&mode=inline)는 샘플의 특성 vector이다.

![$\boldsymbol{\theta}$](https://render.githubusercontent.com/render/math?math=%5Cboldsymbol%7B%5Ctheta%7D&mode=inline)는 편향값과 특성 가중치값들을 담고있는 모델의 파라미터 vector이다.

![$\boldsymbol{h_{\theta}}$](https://render.githubusercontent.com/render/math?math=%5Cboldsymbol%7Bh_%7B%5Ctheta%7D%7D&mode=inline)는 모델 파라미터 ![$\boldsymbol{\theta}$](https://render.githubusercontent.com/render/math?math=%5Cboldsymbol%7B%5Ctheta%7D&mode=inline)를 사용한 가설 (hypothesis)함수 이다.

```Python
import numpy as np

X = 2 * np.random.rand(100, 1)
y = 4 + 3 * X + np.random.randn(100, 1)
var = np.random.rand(100, 1)


plt.plot(X, y, "b.")
plt.xlabel("$x_1$", fontsize=18)
plt.ylabel("$y$", rotation=0, fontsize=18)
plt.axis([0, 2, 0, 15])
save_fig("generated_data_plot")
plt.show()
```

**정규 방정식**

![$\hat{\boldsymbol{\theta}} = (\mathbf{X}^T \mathbf{X})^{-1} \mathbf{X}^T \mathbf{y}$](https://render.githubusercontent.com/render/math?math=%5Chat%7B%5Cboldsymbol%7B%5Ctheta%7D%7D%20%3D%20%28%5Cmathbf%7BX%7D%5ET%20%5Cmathbf%7BX%7D%29%5E%7B-1%7D%20%5Cmathbf%7BX%7D%5ET%20%5Cmathbf%7By%7D&mode=inline)

```Python
X_b = np.c_[np.ones((100, 1)), X]  
theta_best = np.linalg.inv(X_b.T.dot(X_b)).dot(X_b.T).dot(y)
theta_best
```

```
array([[3.89960932],
       [3.11452058]])
```


다음과 같이 예측값을 구하는 선형모델을 생성할 수 있다.

![$\hat{y} = \mathbf{X} \boldsymbol{\hat{\theta}}$](https://render.githubusercontent.com/render/math?math=%5Chat%7By%7D%20%3D%20%5Cmathbf%7BX%7D%20%5Cboldsymbol%7B%5Chat%7B%5Ctheta%7D%7D&mode=inline)

```Python
X_new = np.array([[0], [2]])
X_new_b = np.c_[np.ones((2, 1)), X_new]  # 모든 샘플에 x0 = 1을 추가합니다.
y_predict = X_new_b.dot(theta_best)
y_predict
```

```
array([[ 3.89960932],
       [10.12865049]])
```

```Python
plt.plot(X_new, y_predict, "r-")
plt.plot(X, y, "b.")
plt.axis([0, 2, 0, 15])
plt.show()
```

```Python
# sci-kit learn의 LinearRegression을 통해 훈련을 할 수 있다.

from sklearn.linear_model import LinearRegression

lin_reg = LinearRegression()
lin_reg.fit(X, y)
lin_reg.intercept_, lin_reg.coef_
```

```
(array([3.89960932]), array([[3.11452058]]))
```

```Python
lin_reg.predict(X_new)
```

```Python
array([[ 3.89960932],
       [10.12865049]])
```

```Python
# LinearRegression 클래스는 scipy.linalg.lstsq() 함수("least squares"의 약자)를 사용한다.
# 싸이파이 lstsq() 함수를 사용하려면 scipy.linalg.lstsq(X_b, y)와 같이 씁니다.
theta_best_svd, residuals, rank, s = np.linalg.lstsq(X_b, y, rcond=1e-6)
theta_best_svd
```

```python
array([[3.89960932],
       [3.11452058]])
```


이 함수는 ![$\mathbf{X}^+\mathbf{y}$](https://render.githubusercontent.com/render/math?math=%5Cmathbf%7BX%7D%5E%2B%5Cmathbf%7By%7D&mode=inline)을 계산합니다. ![$\mathbf{X}^{+}$](https://render.githubusercontent.com/render/math?math=%5Cmathbf%7BX%7D%5E%7B%2B%7D&mode=inline)는 ![$\mathbf{X}$](https://render.githubusercontent.com/render/math?math=%5Cmathbf%7BX%7D&mode=inline)의 *유사역행렬* (pseudoinverse)입니다(Moore–Penrose 유사역행렬입니다). `np.linalg.pinv()`을 사용해서 유사역행렬을 직접 계산할 수 있습니다:

![$\boldsymbol{\hat{\theta}} = \mathbf{X}^{-1}\hat{y}$](https://render.githubusercontent.com/render/math?math=%5Cboldsymbol%7B%5Chat%7B%5Ctheta%7D%7D%20%3D%20%5Cmathbf%7BX%7D%5E%7B-1%7D%5Chat%7By%7D&mode=inline)

```Python
np.linalg.pinv(X_b).dot(y)
```

```Python
array([[3.89960932],
       [3.11452058]])
```

## 경사 하강법

경사 하강법은 여러 종류의 문제에서 최적의 solution을 찾는 방법중 하나 이다. 다양한 종류의 경사 하강법들이 있는데, 공통 목적은 비용 함수를 최소화 하기위해 반복해서 parameter를 조정해가는 것이다.

### 배치 경사 하강법을 사용한 선형 회귀

**비용 함수의 그레이디언트 벡터**

![$ \dfrac{\partial}{\partial \boldsymbol{\theta}} \text{MSE}(\boldsymbol{\theta})  = \dfrac{2}{m} \mathbf{X}^T (\mathbf{X} \boldsymbol{\theta} - \mathbf{y}) $](https://render.githubusercontent.com/render/math?math=%5Cdfrac%7B%5Cpartial%7D%7B%5Cpartial%20%5Cboldsymbol%7B%5Ctheta%7D%7D%20%5Ctext%7BMSE%7D%28%5Cboldsymbol%7B%5Ctheta%7D%29%0A%20%3D%20%5Cdfrac%7B2%7D%7Bm%7D%20%5Cmathbf%7BX%7D%5ET%20%28%5Cmathbf%7BX%7D%20%5Cboldsymbol%7B%5Ctheta%7D%20-%20%5Cmathbf%7By%7D%29&mode=inline)

**경사 하강법의 스텝**

![$ \boldsymbol{\theta}^{(\text{next step})} = \boldsymbol{\theta} - \eta \dfrac{\partial}{\partial \boldsymbol{\theta}} \text{MSE}(\boldsymbol{\theta}) $](https://render.githubusercontent.com/render/math?math=%5Cboldsymbol%7B%5Ctheta%7D%5E%7B%28%5Ctext%7Bnext%20step%7D%29%7D%20%3D%20%5Cboldsymbol%7B%5Ctheta%7D%20-%20%5Ceta%20%5Cdfrac%7B%5Cpartial%7D%7B%5Cpartial%20%5Cboldsymbol%7B%5Ctheta%7D%7D%20%5Ctext%7BMSE%7D%28%5Cboldsymbol%7B%5Ctheta%7D%29&mode=inline)

배치경사 하강법은 매 스텝에서 전체 훈련 세트 X에 대해 계산한다. 그래서 매우 큰 훈련 세트에서는 아주 느리다.


