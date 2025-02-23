# :microscope: 반도체 이상 탐지 프로그램 :microscope:
<p align="center">
 <img src=https://github.com/user-attachments/assets/3e579722-8422-4e49-9876-10d7e4262eab>
</p>
<br>

# 프로젝트 소개
* 연구 동기
  * 반도체 제조 공정에서 웨이퍼는 수백, 수천 개의 작은 칩으로 구성된 기판이다. 이 칩들은 최종 제품이 되기 전에 여러 단계의 복잡한 공정을
    거쳐야 한다. 웨이퍼의 결함은 제품 수율 저하와 품질 문제로 이어질 수 있기 때문에 조기 발견 및 제거가 매우 중요하다.
  * 그러나 반도체 생산 공정은 매우 복잡하고 정밀하기 때문에 미세한 결함도 제품 불량으로 이어질 수 있다.  
    a. 낮은 정확도 : 미세한 결함이나 복잡한 패턴의 결함을 정확하게 탐지하기 어렵다. 
    b. 높은 오류율 : 정상 제품을 결함으로 판단하는 오류가 발생할 수 있다.   
    c. 낮은 처리 속도 : 대량의 데이터를 처리하는데 시간이 오래 걸린다.   
    d. 전문가 의존도가 높음 : 시스템 구축 및 운영에 전문 지식이 필요하다.   
  * 딥러닝 기반 반도체 이상 탐지 프로그램은 이러한 한계를 극복하기 위해 개발하기 시작되었다.
 <br>

 # 개발 계획 
 a. 데이터 수집 및 전처리 :
  * 반도체 생산 공정에서 휙득한 이미지 데이터를 수집한다.
  * 수집된 데이터를 전처리하여 모델 학습에 적합한 형태로 변환한다.

b. 데이터 증강
  * One-hot-encoding과 Convolutional AutoEncoder로 부족한 데이터를 증강한다.
    
c. AI 모델 개발 :
  * 이미지 인식, 패턴 분석, 딥러닝 기술을 활용한 VGGNet, ResNet, EfficientNet 모델로 반도체 이상 탐지 시스템을 개발한다.
  * 위 세 가지 모델을 학습하고 평가하여 최적의 모델(VGGNet16, ResNet50, EfficientNetB0)을 선정한다.

d. 웹 구축 및 평가
  * 개발된 AI 모델을 기반으로 실제 반도체 생산 공정에 적용할 수 있도록 Django 웹 프레임워크를 활용해 시스템을 구축한다.
  * 구축된 시스템을 실제 생산 환경에서 휙득한 웨이퍼 이미지 데이터로 평가하여 성능을 검증한다.

# 개발 기간
2024.03.04 ~ 2024.06.14  
1~2주차 – 이상탐지라는 주제를 놓고 갖가지 관련 자료 수집 및 아이디어 설계를 진행하였으며, 모델 학습을 위한 데이터에 관한 조사를 진행하였다.

3~5주차 – 이상 탐지를 위한 이미지 분류 관련 AI 모델에 대해 알아보았으며, 2주차에 조사한 데이터에 대한 전처리를 진행하였다.

6~11주차 – VGGNet, ResNet, EfficientNet AI모델의 각각에 대한 이해와 학습을 진행한 결과에 대해 도출해냈으며 해당 AI모델을 적용하고 이미지 입력에 대한 결과를 나타내기 위한 WEB을 구현.

12~14주차 – 학습된 AI 모델을 적용하였으며 그에 대한 테스트 진행.

15주차 – 불량 유형 인식 못하거나 결과 화면이 나오지 않는 문제점 보완.

# 프로젝트 구조도
  > 프로젝트 전체 구조도
  <img src=https://github.com/user-attachments/assets/a88cd921-35b4-4016-8e31-f54fff99d599>  
  <br>
  > 프로젝트 AI 구조도
  <img src=https://github.com/user-attachments/assets/806e3ff2-9a06-4054-a2e5-e51572d94e9a>

  
