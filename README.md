# DAEM-ERC: Data Augmentation and Ensemble-based Model for Emotion Recognition in Conversation
# Abstract
> 대화에서의 감정 인식(Emotion Recognition in Conversations, ERC)은 사람과 컴퓨터 간의 상호작용을 위한 핵심 기술이다. 
> 감정 인식을 위해 사용되는 데이터에는 오디오, 비디오, 텍스트 등이 있고, 이러한 데이터들로부터 얻은 정서적 정보를 결합하여 멀티모달 감정 인식을 구현할 수 있다. 
> 
> 본 연구에서는 발화 텍스트, 오디오, 생체 신호 데이터를 사용하여 각 데이터에 특화된 개별 분류기를 생성한 뒤 Weighted soft voting 앙상블을 통해 최종 감정 분류를 진행하는 멀티모달 감정 인식 모델을 제안한다. 
> 
> 또한 우리는 각 데이터 특성을 고려한 증강 기법을 사용하여 심각한 클래스 불균형 문제를 완화했다. 
> 결과적으로 우리의 제안 모델은 소수 클래스 감정 분류에 강점을 가지며 Weighted f1 스코어 0.91을 달성했다.
> 


# Model Architecture
![model architecture](/images/DAEM-ERC%20모델%20구조.png)

#  Directory
```commandline
+-- text
    +-- TextCNN for ERC.ipynb
    +-- text_augmentation.ipynb
    +-- text_tokenization.ipynb
+-- audio
    +-- 
+-- bio
    +-- biosensor_model.ipynb
    +-- biosensor_preprocessing.ipynb
```
# Dataset

- [한국어 멀티모달 감정 데이터셋 2020(KEMDy2020)](https://nanum.etri.re.kr/share/kjnoh/KEMDy20?lang=ko_KR)

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
</tr align="center">
  <tr bgcolor="none">
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
- FastText 한국어 사전 학습 모델
### 2) Libraries
```commandline
 - torchmetrics == 0.11.4
 - torch == 2.0.0+cu118
 - scipy == 1.10.1
 - pandas == 1.5.3
 - numpy == 1.22.4
 - sklearn == 1.2.2

 (for text)
 - gensim == 3.8.3

 (for audio)
 - librosa == 0.10.0.post2
 
 (for bio)
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

