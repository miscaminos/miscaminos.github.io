---
layout: post           # (require) default post layout
title: "EDA-Exploratory-Data-Analysis"          # (require) a string title
date: 2021-07-06       # (require) a post date
categories: [dataAnalysis]   # (custom) some categories, but makesure these categories already exists inside path of `category/`
tags: [EDA]           		# (custom) tags only for meta `property="article:tag"`
---

# EDA(Exploratory Data Analysis)

<br>

## 탐색적 데이터 분석

<br>

EDA는 data의 주요 특징을 알아내기위해 data set을 분석하는 방식이다. 특정 기법들로 이루어지기보다는 데이터 분석이 진행되는 방향을 제시하거나 기반이 되는 approach/philosophy이다. 주로 시각적인 방법으로 결과를 나타내는데, statistical graphics 또는 다른 data visualization을 사용해서 분석 결과를 나타낸다.

"...EDA is not a mere collection of techniques; EDA is a philosophy as to how we dissect a data set; what we look for; how we look; and how we interpret. It is true that EDA heavily uses the collection of techniques that we call "statistical graphics"... "


<br>

NIST(National Institute of Standards and Technology)에서는 EDA가 다음과 같이 7개의 목적을 가지고있다고 설명한다.

1) maximize insight into a data set

2) uncover underlying structure

3) extract important variables

4) detect outliers and anomalies

5) test underlying assumptions

6) develop parsimonious models 

7) determine optimal factor settings


<br>
<br>

### 1. 목적

EDA의 목적은 데이터를 이해하는것.

가장 심플한 방법은 질문을 잘 만들고 이를 탐색하는 과정을 거쳐서 데이터를 표현하는 적절한 시각화 산출물과 다음 과정을 위한 데이터 또는 실험해볼 아이디어를 생성하는 것이다.

데이터에 대해 아는것이 없는 상태에서 가장 빠르게 좋은 질문을 만드는 방법은 많은 양의 질문을 생성하고 탐색하는 것이다.


<br>
<br>

### 2. 방식

가장 기본적인 방법으로 데이터의 속성과 기복적인 통계값들을 탐색해볼 수 있다.

1) 데이터 탐색 (데이터의 항목 개수, 속성 목록, 누락값, 각 속석의 데이터타입, 데이터 가공 과정에서 나오는 오류/누락 확인)

2) 이상치분석(전체적인 추세와 특이사항 : 무작위로 표본을 추출해서 관찰, 요약 통계 지표 활용)
-데이터의 중심(평균/중앙값/최빈값)
-데이터의 분산도(범위/분산)

또한, 개념적으로 다음 두가지 질문을 탐색해볼 수 있다.

1) 데이터에 포함된 변수에 내재된 변동성(variation) 유형이 어떻게 되는지?

2) 변수들간에 공변동(covariation)은 어떻게 되는지? (변수 2개 또는 그 이상의 변수들이 관련되어 함께 변동하는 경향이 어떤지)

<table>
    <tr><td></td><td> 연속형  </td><td>  범주형  </td></tr>
    <tr><td>  연속형  </td><td>  산점도  </td><td>  상자그림  </td></tr>
    <tr><td>  범주형  </td><td>  상자그림  </td><td>  타일(heatmap)  </td></tr>
</table>

*범주형 변수 : 명목형, 순서형 데이터    
*연속형 변수 : 이산형, 연속형 데이터


EDA 방식의 기법들은 대부분 graphical 기법 또는 quantitative 기법들이다. 그 이유는 데이터를 graphics로 표현해서 분석가들이 좀 더 개방적인 마인드로 데이터안에 숨겨져있는 의미/패턴이나 새로운 아이디어를 도출하도록 하기 위함이다. 주로 활용되는 techniques는:

3) plotting raw data (e.g., data traces, histograms, probability plots, lag plots, block plots, Youden plots) 변수의 variation을 이해하거나, outlier를 인식하는데에 용이하다.

4) plotting simple statistics (e.g., mean plots, standard deviation plots, box plots)

5) pattern을 인식에 용이하도록 plot positioning (e.g., multiple plots per page)


#### note: IDA와 다른점?

IDA(Initial Data Analysis)는 좀 더 model fitting 또는 hypothesis testing에 필요한 assumptions를 확인하는데에 focus가 맞춰져있다. missing values를 처리하거나 variables에 필요한 transformation을 적용하는 등, EDA안에 포함되는 더 좁은 범위를 뜻한다.


<br>
<hr>

#### references

1. [nist engineering statistics handbook][https://www.itl.nist.gov/div898/handbook/eda/section1/eda11.htm]

2. [software carpentry github blog][https://statkclee.github.io/ml/ml-eda.html#:~:text=%ED%83%90%EC%83%89%EC%A0%81%20%EC%9E%90%EB%A3%8C%20%EB%B6%84%EC%84%9D%EA%B3%BC%EC%A0%95%EC%9D%80%20%EB%AF%B8%EA%B5%AD%EC%9D%98%20%ED%8A%9C%ED%82%A4%EB%B0%95%EC%82%AC,%EB%8B%A4%EC%96%91%ED%95%9C%20%EB%B0%A9%EB%B2%95%EC%9D%84%20%EC%A0%81%EC%9A%A9%ED%95%9C%EB%8B%A4.]

3. [Wikipedia - Exploratory data analysis][https://en.wikipedia.org/wiki/Exploratory_data_analysis]

4. 스터디 모임의 은상님 notes
