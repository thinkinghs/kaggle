# -*- coding: utf-8 -*-
"""
Created on Thu Feb 13 16:22:36 2020

@author: Abraham
"""
# https://www.kaggle.com/kaggle/kaggle-survey-2017
# https://www.edwith.org/boostcourse-ds-kaggle/lecture/57564/
# 2017 Kaggle ML & DS Survey
# 캐글러를 대상으로 한 설문 조사

import pandas as pd
import numpy as np
from scipy import stats
import matplotlib.pyplot as plt
import seaborn as sns

# warning 보이지 않도록 처리
import warnings
warnings.filterwarnings('ignore')

import os
os.getcwd()
os.chdir('C:\\Users\\Abraham\\Desktop\\코딩공부')

question = pd.read_csv('data/schema.csv')
question.shape # 몇행 몇 열인지

question.head()

# 선다형 질문 
mcq = pd.read_csv('data/multipleChoiceResponses.csv', encoding="ISO-8859-1", low_memory=False)
mcq.shape
mcq.columns
mcq.head(10)

# missingno는 NaN 데이터들에 대해 시각화를 해준다.
# 여기서는 NaN 데이터의 컬럼이 많아 아래 그래프만으로는 내용 파악이 어렵다. 그러나 NaN 이 어떻게 분포되어 있는지 보면 데이터 가공을 생각하는데 도움이 됨.
import missingno as msno
msno.matrix(mcq, figsize=(12,5))

# 성별
# seaborn 은 그래프 편하게 그려주는 library
sns.countplot(y='GenderSelect', data=mcq)

# 국가별 응답수. value_count 이용하면 sql에서 count 함수와 같은 기능
con_df = pd.DataFrame(mcq['Country'].value_counts())
# 국가별 숫자가 country라는 컬럼명으로 되어 있음.
# 그래서 국가 컬럼을 '국가'라는 이름으로 추가하고, 'Country, 국가'라고 되어 있는 컬럼명을 '응답수, '국가'로 바꾼다.
con_df['국가'] = con_df.index
con_df.columns = ['응답 수', '국가']
con_df = con_df.reset_index().drop('index', axis=1)
con_df

# describe 통하면 여러 통계 정보 볼 수 있음
mcq['Age'].describe()

# seaborn 이용한 시각화
# 아래 mcq[mcq['Age'] > 0]['Age'] 문법이 가능한 이유는 정확히 모르겠으나 mcq가 Series 타입의 객체이기 때문인 것 같음. Series type 객체는 dictionary처럼 key값을 이용하는데 뭔가 iterable하게 작동해서 연산이 가능한 것 같음.
sns.distplot(mcq[mcq['Age'] > 0]['Age'])
# 응답자가 20대부터 급격히 증가하며 30대가 가장 많음을 볼 수 있음.

# 학력
sns.countplot(y='FormalEducation', data=mcq)
# 석사가 학사보다 많고 박사도 많음

# value_counts 사용하면 그룹화된 데이터의 카운트 값을 보여준다.
# normalize=True 옵션을 사용하면 해당 데이터가 전체 데이터에서 어느정도 비율인지 알 수 있다.
mcq_major_count = pd.DataFrame(mcq['MajorSelect'].value_counts())
mcq_major_percent = pd.DataFrame(mcq['MajorSelect'].value_counts(normalize=True))
# left_index와 right_index를 True로 해주면 merge할 때 index 기준으로 합침. 이 데이터는 어차피 count와 percent의 순서가 같을 수 밖에 없으므로 같은 index로 merge 시켜도 됨
mcq_major_df = mcq_major_count.merge(mcq_major_percent, left_index=True, right_index=True)
mcq_major_df.columns = ['응답 수', '비율']
mcq_major_df
# 컴퓨터 전공자가 제일 많고 그 다음 수학, 공학, 전기 공학 순

# 재학 중인 사람들의 전공 현황 시각화
plt.figure(figsize=(6,8))
sns.countplot(y='MajorSelect', data=mcq)

# employment 현황, 위의 major와 동일하다
mcq_es_count = pd.DataFrame(mcq['EmploymentStatus'].value_counts())
mcq_es_percent = pd.DataFrame(mcq['EmploymentStatus'].value_counts(normalize=True))
# left_index와 right_index를 True로 해주면 merge할 때 index 기준으로 합침. 이 데이터는 어차피 count와 percent의 순서가 같을 수 밖에 없으므로 같은 index로 merge 시켜도 됨
mcq_es_df = mcq_es_count.merge(mcq_es_percent, left_index=True, right_index=True)
mcq_es_df.columns = ['응답 수', '비율']
mcq_es_df

sns.countplot(y='EmploymentStatus', data=mcq)

# data science에서의 경험
sns.countplot(y='Tenure', data=mcq)

# 우리나라만 따로 뽑아서
korea = mcq.loc[(mcq['Country'] == 'South Korea')]
#shape은 (row수, column수) 를 return하므로 shape[0] 하면 row수를 알 수 있음
print('The number of interviewees in Korea: ' + str(korea.shape[0]))

# na 데이터 제거하고 시각화 
sns.distplot(korea['Age'].dropna())
plt.title('Korea')

# subplots를 사용하면 2개의 그래프를 한번에 그릴 수 있음
figure, (ax1, ax2) = plt.subplots(ncols=2)
figure.set_size_inches(12, 5)

sns.distplot(korea['Age'].loc[korea['GenderSelect'] == 'Female'].dropna(), norm_hist=False, color=sns.color_palette('Paired')[4], ax = ax1)
plt.title("korean female")
sns.distplot(korea['Age'].loc[korea['GenderSelect']=='Male'].dropna(), norm_hist=False, color=sns.color_palette("Paired")[0], ax=ax2)
plt.title("korean male")
# jupyter notebook이 아니여서 plt.title이 안 먹히는거 같다. 위 6라인 한번에 컴파일 시키면 title 표기되긴 하나 male꺼만 표기 됨 

# xticks 을 이용하면 하단에 label이 기울여져서 표기되서, label 간에 겹치지 않음
sns.barplot(x=korea['EmploymentStatus'].unique(), y=korea['EmploymentStatus'].value_counts()/len(korea))
plt.xticks(rotation=30, ha='right')
plt.title('Employment status of the korean')
plt.ylabel('')
plt.show()

# data science 공부에 얼마나 많은 시간을 사용하는지?
plt.figure(figsize=(6, 8))
# hue 값을 주면 해당 parameter로 세분화해서 볼 수 있음.
sns.countplot(y='TimeSpentStudying', data=mcq, hue='EmploymentStatus').legend(loc='center left', bbox_to_anchor=(1, 0.5))


# 현재 full time 고용 상태인지, not employed인지에 따라 분석
full_time = mcq.loc[(mcq['EmploymentStatus'] == 'Employed full-time')]
print(full_time.shape)
looking_for_job = mcq.loc[(mcq['EmploymentStatus'] == 'Not employed, but looking for work')]
print(looking_for_job.shape)

# 두 개의 그래프가 다 보이게.
figure, (ax1, ax2) = plt.subplots(ncols=2)
figure.set_size_inches(12,5)
sns.countplot(x='TimeSpentStudying', data=full_time, hue='EmploymentStatus', ax=ax1).legend(loc='center right',bbox_to_anchor=(1, 0.5))
sns.countplot(x='TimeSpentStudying', data=looking_for_job, hue='EmploymentStatus', ax=ax2).legend(loc='center right',bbox_to_anchor=(1, 0.5))

# seaborn으로 boxplot 그래프도 그릴 수 있음.
# https://colab.research.google.com/drive/1XKbJKEIfg43G1QfncB6jhUrNQ8jJIIdD 참고


############################## 질문한 설문 내용 데이터 ######
question = pd.read_csv('data/schema.csv')


### 데이터 사이언스 직업을 찾는데 가장 고려해야 할 요소는?
# JobFactor라는 단어를 포함하고 있는 Column을 qc에 할당
qc = question.loc[question['Column'].str.contains('JobFactor')]

job_factors = [x for x in mcq.columns if x.find('JobFactor') != -1]

jfdf = {}
for feature in job_factors:
    a = mcq[feature].value_counts()
    a = a/a.sum()
    # feature 변수에서 jobfactor 단어가 항상 맨 앞에 나와서 이를 제외하기 위해 len('Jobfactor'): 구문이 있음
    jfdf[feature[len('JobFactor'):]] = a

jfdf = pd.DataFrame(jfdf).transpose()

plt.figure(figsize=(6,10))
plt.xticks(rotation=60, ha='right')
# 히트맵으로 표현
sns.heatmap(jfdf.sort_values('Very Important', ascending=False), annot=True)

# 일반 plot으로 표현, 공부할 수 있는 환경(learning)이 제일 중요함을 볼 수 있음 
jfdf.plot(kind='bar', figsize=(18,6), title="Things to look for while considering Data Science Jobs")
plt.xticks(rotation=60, ha='right')
plt.show()

