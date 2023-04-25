# DAEM-ERC: Data Augmentation and Ensemble-based Model for Emotion Recognition in Conversation
# Abstract
대화에서의 감정 인식(Emotion Recognition in Conversations, ERC)은 사람과 컴퓨터 간의 상호작용을 위한 핵심 기술이다. 
감정 인식을 위해 사용되는 데이터에는 오디오, 비디오, 텍스트 등이 있고, 이러한 데이터들로부터 얻은 정서적 정보를 결합하여 멀티모달 감정 인식을 구현할 수 있다. 

본 연구에서는 발화 텍스트, 오디오, 생체 신호 데이터를 사용하여 각 데이터에 특화된 개별 분류기를 생성한 뒤 Weighted soft voting 앙상블을 통해 최종 감정 분류를 진행하는 멀티모달 감정 인식 모델을 제안한다. 

또한 우리는 각 데이터 특성을 고려한 증강 기법을 사용하여 심각한 클래스 불균형 문제를 완화했다. 
결과적으로 우리의 제안 모델은 소수 클래스 감정 분류에 강점을 가지며 Weighted f1 스코어 0.91을 달성했다.


# Model Architecture
<img src="/images/DAEM-ERC%20모델%20구조.png" width="70%" height="60%" title="모델 구조" alt="Model Architecture"></img><br/>

#  Directory
```commandline
+-- READ.ME
+-- total_preprocessing.ipynb            : 공통 데이터 전처리 실행 코드 (가장 먼저 실행해야 함.) 
+-- ensemble.ipynb                       : Weighted soft voting 앙상블 진행 코드
+-- text
    +-- 1. text_preprocessing.ipynb      : 텍스트 데이터 전처리를 위한 코드
    +-- 2. text_tokenization.ipynb       : 텍스트 데이터 토큰화를 위한 코드
    +-- 3. text_augmentation.ipynb       : 텍스트 데이터 증강을 위한 코드
    +-- 4. TextCNN for ERC.ipynb         : TextCNN 분류 모델 훈련 및 평가 과정 코드
+-- wav
    +-- wav_run.ipynb                    : 전처리 ~ 감정 분류(+출력값 저장)까지의 실행 코드
    +-- wav_preprocess.py                : 오디오 데이터 전처리를 위한 클래스 정의 코드
    +-- wav_classifier.py                : 오디오 데이터에 대한 분류기 정의 코드
+-- bio
    +-- 1. biosensor_preprocessing.ipynb : 생체 신호 데이터 전처리 과정 코드
    +-- 2. biosensor_model.ipynb         : 생체 신호 분류 모델 훈련 및 평가 과정 코드
```
# Dataset

- [한국어 멀티모달 감정 데이터셋 2020(KEMDy2020)](https://nanum.etri.re.kr/share/kjnoh/KEMDy20?lang=ko_KR)

# How to use?
### ❗프로젝트 디렉터리❗
- 코드를 실행하기 앞서, 프로젝트는 아래와 같은 디렉터리로 구성되어 있어야 함
```
+-- ERC_ETRI(사용자지정)
	+-- dataset
		+-- KEMDy20_v1_1
```
- 모든 코드는 **ERC_ETRI(사용자지정) 폴더에 위치하여 실행**되어야 함
### 1) 먼저 위 데이터 셋을 다운받은 후 KEMDy20_v1_1 폴더에 아래와 같은 새 폴더들을 생성
```commandline
+-- KEMDy20_v1_1
    +-- new
	    +-- annotation
	    +-- text
	    +-- wav
	    +-- sensor
	    	+-- EDA
		+-- Temp
		+-- bio_train
```
### 2) 이후 total_preprocessing.ipynb 파일의 전처리 셀들을 모두 실행
    (이때 전처리 2 진행하기 전, 생체 신호 데이터의 전처리를 일부 수행할 것)   
   
--> KEMDy20_v1_1/new/annotation 폴더 아래에 all_annotation, train_origin, test_origin 피클 파일 저장됨

### 3) 위 데이터 셋을 이용하여 데이터 각각에 대한 전처리 및 감정 분류 진행   

     - Wav: KEMDy20_v1_1/new/wav/1. wav_run.ipynb 파일의 모든 셀 실행 
     - bio:
       - KEMDy20_v1_1/new/sensor/1. biosensor_preprocessing.ipynb 파일의 모든 셀 실행
       - KEMDy20_v1_1/new/sensor/2. biosensor_model.ipynb 파일의 모든 셀 실행
     - text:
       - ETRI_ERC/1. text_preprocessing.ipynb 파일의 모든 셀 실행
       - ETRI_ERC/2. text_tokenization.ipynb 파일의 모든 셀 실행
       - ETRI_ERC/3. text_augmentation.ipynb 파일의 모든 셀 실행
       - ETRI_ERC/4. TextCNN for ERC.ipynb 파일의 모든 셀 실행

--> 개별 분류기에 따른 감정 분류 결과 피클 파일 저장됨

### 4) 마지막으로 Ensemble.ipynb 파일을 실행하여 weighted soft voting 앙상블 진행

# Results
### 1) 개별 분류 모델의 F1 스코어 
<table>
  <tr align="center" bgcolor="lightgray">
    <td rowspan="2">모델</td>
    <td colspan="9">F1 score</td>
  </tr>
<tr align="center">
    <td>분노</td>
    <td>혐오</td>
    <td>공포</td>
    <td>기쁨</td>
    <td>중립</td>
    <td>슬픔</td>
    <td>놀람</td>
    <td>Macro</td>
    <td>Weighted</td>
</tr>
  <tr bgcolor="none" align="center">
    <td>텍스트</td>
    <td>0.09</td>
    <td>0.29</td>
    <td>0.47</td>
    <td>0.18</td>
    <td>0.92</td>
    <td>0.11</td>
    <td>0.22</td>
    <td>0.33</td>
    <td>0.86</td>
  </tr>
  <tr bgcolor="none" align="center">
    <td>오디오</td>
    <td>0.07</td>
    <td>0.12</td>
    <td>0.20</td>
    <td>0.50</td>
    <td>0.94</td>
    <td>0.00</td>
    <td>0.14</td>
    <td>0.27</td>
    <td>0.90</td>
  </tr>
  <tr bgcolor="none" align="center">
    <td>생체 신호</td>
    <td>0.05</td>
    <td>0.00</td>
    <td>0.00</td>
    <td>0.04</td>
    <td>0.91</td>
    <td>0.00</td>
    <td>0.00</td>
    <td>0.14</td>
    <td>0.85</td>
  </tr>
</table>

### 2) 앙상블 모델 성능 비교
<table>
  <tr align="center">
    <td rowspan="2">앙상블 기법</td>
    <td colspan="9">F1 score</td>
  </tr>
<tr align="center">
    <td>분노</td>
    <td>혐오</td>
    <td>공포</td>
    <td>기쁨</td>
    <td>중립</td>
    <td>슬픔</td>
    <td>놀람</td>
    <td>Macro</td>
    <td>Weighted</td>
</tr>
  <tr bgcolor="none"  align="center">
    <td>Soft voting</td>
    <td>0.00</td>
    <td>0.00</td>
    <td>0.29</td>
    <td>0.15</td>
    <td>0.94</td>
    <td>0.00</td>
    <td>0.13</td>
    <td>0.22</td>
    <td>0.93</td>
  </tr>
  <tr bgcolor="none" align="center">
    <td>Weighted soft voting</td>
    <td>0.04</td>
    <td>0.21</td>
    <td>0.50</td>
    <td>0.19</td>
    <td>0.93</td>
    <td>0.11</td>
    <td>0.19</td>
    <td>0.31</td>
    <td>0.91</td>
  </tr>
</table>

# Requirments
### 1) Files
- [FastText 사전 학습 모델](https://fasttext.cc/docs/en/pretrained-vectors.html) - 한국어 bin+text 
### 2) Libraries
```commandline
 - torchmetrics == 0.11.4
 - torch == 2.0.0+cu118
 - scipy == 1.10.1
 - pandas == 1.5.3
 - numpy == 1.22.4
 - sklearn == 1.2.2
 - gensim == 3.8.3
 - librosa == 0.10.0.post2
 - imbalanced-learn == 0.10.1
```
# References

[1] Wei et al., "Eda: Easy data augmentation techniques 
for boosting performance on text classification tasks." 
arXiv preprint arXiv:1901.11196 2019. https://arxiv.org/abs/1901.11196

[2] Fernández et al. "SMOTE for learning from imbalanced data: 
progress and challenges, marking the 15-year anniversary." 
Journal of artificial intelligence research 61, 863-905, 2018. https://jair.org/index.php/jair/article/view/11192

[3] Pandey, Amit, and Achin Jain. "Comparative analysis 
of KNN algorithm using various normalization techniques." 
International Journal of Computer Network and Information 
Security 11.11, 36, 2017. https://mecs-press.net/ijcnis/ijcnis-v9-n11/IJCNIS-V9-N11-4.pdf

[4] Chen et al., "Convolutional neural network for sentence 
classification," MS thesis. University of Waterloo, 2015. https://arxiv.org/abs/1408.5882

[5] Hu, Jie et al., "Squeeze-and-excitation networks." 
Proceedings of the IEEE conference on computer vision 
and pattern recognition. 2018. https://arxiv.org/abs/1709.01507

