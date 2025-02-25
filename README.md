# :microscope: 반도체 이상 탐지 프로그램 :microscope:
<p align="center">
 <img src=https://github.com/user-attachments/assets/3e579722-8422-4e49-9876-10d7e4262eab>
</p>
<br>

# 프로젝트 소개
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
<br>

# 개발 기간
2024.03.04 ~ 2024.06.14  
<dir>
  1~2주차 – 이상탐지라는 주제를 놓고 갖가지 관련 자료 수집 및 아이디어 설계를 진행하였으며, 모델 학습을 위한 데이터에 관한 조사를 진행하였다.
  
  3~5주차 – 이상 탐지를 위한 이미지 분류 관련 AI 모델에 대해 알아보았으며, 2주차에 조사한 데이터에 대한 전처리를 진행하였다.
  
  6~11주차 – VGGNet, ResNet, EfficientNet AI모델의 각각에 대한 이해와 학습을 진행한 결과에 대해 도출해냈으며 해당 AI모델을 적용하고 이미지 입력에 대한 결과를 나타내기 위한 WEB을 구현.
  
  12~14주차 – 학습된 AI 모델을 적용하였으며 그에 대한 테스트 진행.
  
  15주차 – 불량 유형 인식 못하거나 결과 화면이 나오지 않는 문제점 보완.
</dir>
<br>

# 기술 스택
* Tools
<div>
  <img src="https://img.shields.io/badge/tensorflow-FF6F00?style=for-the-badge&logo=tensorflow&logoColor=white">
  <img src="https://img.shields.io/badge/keras-D00000?style=for-the-badge&logo=keras&logoColor=white">
  <img src="https://img.shields.io/badge/django-092E20?style=for-the-badge&logo=django&logoColor=white">
</div>
<br>

* AI models
<div>
  <strong>VGG16, ResNet50, EfficientB0</strong>
</div>
</br>

# 데이터셋
> [MIR-WM811K](http://mirlab.org/dataSet/public/)
<br>

# 프로젝트 구조
> 프로젝트 전체 구조도
![image](https://github.com/user-attachments/assets/6ec73f6d-d981-4a1f-b705-9c956d1ff605)
<br>

> 프로젝트 AI 구조도
![image](https://github.com/user-attachments/assets/806e3ff2-9a06-4054-a2e5-e51572d94e9a)
<br>

# 전처리 및 AI 모델 학습 및 평가
* 전처리 방식
  * 결측치 제거
    > 데이터셋(LSWMD.pkl)의 info  
    ![image](https://github.com/user-attachments/assets/6bc6238e-a2db-4e3a-b31b-0b4a952619d6)
    
    IEEE에서 제공한 WM-811k wafer map에 대한 데이터 세트로 811,457개의 waferMap과 wafer die size, lot Name 및 wafer index와 같은
    추가 정보로 구성되어 있다.

    > Failure Type 분포도
    ![image](https://github.com/user-attachments/assets/45e1ef39-b83c-441e-8205-111714c0b587)

    FaiureType이 ‘Center’,‘Donout’,‘Edge-Loc’,‘Edge-Ring’,‘Loc’,‘Random’,‘Scratch’,‘Near-full’,‘none’으로 구성되어 있고
    trainTestLabel은 ‘Training’ 과 ‘Test’로 나뉘어져 있다.
    
 * 데이터 증강(Data Augmentation)
   
   a. 결측치 제거로 인한 훈련 데이터가 부족한 현상으로 인해, 기존 데이터 세트의 데이터를 인위적으로 변형하거나 수정하여 새로운 데이터를 만드는 기술인 데이터 증강이 필요하다.  

   b. 27 x 27 wafer map  

   c. One-hot-Encoding  

   d. Convolutional AutoEncoder  
   ![image](https://github.com/user-attachments/assets/780c22a3-7731-4f56-a3ec-c86c111fa04a)

* AI 모델 학습 및 평가
  * VGG16  
    > VGG16 Confusion Matrix  
    ![image](https://github.com/user-attachments/assets/16c0fcce-53b7-4dd3-a914-39d4e29c2b0e)
    
  * ResNet50  
    > ResNet50 Confusion Matrix  
    ![image](https://github.com/user-attachments/assets/b9e48da3-4a0c-431b-8a8f-0badfc8ec6e2)

  * EfficientNetB0  
    > EfficientNetB0 Confusion Matrix  
    ![image](https://github.com/user-attachments/assets/3f0fa82d-9433-4c87-be63-db0135e26798)

  Confusion Matrix (오차행렬)은 학습된 분류 모델이 예측을 수행하면서 얼마나 헷갈려(confused)하고 있는지 보여주는 지표이다.
  즉, 이진 분류의 예측 오류가 얼마인지 와 더불어 어떤 유형의 예측 오류가 발생하고 있는지를 함게 나타내는 지표이다.

  VGG16모델과 ResNet50 모델은 정규화된 오차 행렬을 나타내고 있다. Donut 유형을 예측했을 때 Donut이 아닌 값을 예측하는 FN(False Negative)를 나타낸다.
  
  EfficientNetB0 모델도 Donut 유형을 예측했을 때 Donut이 아닌 값을 예측하는 FN(False Negative)가 나타나는 걸 미루어 볼 때, 데이터셋 load 혹은 전처리 과정에서 donut 유형의 값이 깨진게 아닌지 추측해본다.
<br>

# 웨이퍼 불량 검출 과정
* 메인 화면 
  > 메인 화면 UI  
  ![image](https://github.com/user-attachments/assets/e81552d1-4a85-457b-b89c-c0d1cf1b9980)

  이미지 업로드에는 업로드하기 위한 이미지들과 해당 이미지셋에 대한 설명을 저장할 수 있는 Create Imageset 버튼과 사용자가 만들었던 이미지셋을 확인 가능한 My ImageSet List 버튼이 있다.

  AI 모델 조회에 모델 목록 버튼은 해당 연구에 사용되는 VGGNet, ResNet, EfficientNet 에 대한 설명을 확인 가능하다. 또한 Dashboard는 사용자 정보를 확인 가능하다.

* 이미지 셋 제작 및 업로드  
  > Create ImageSet  
  ![image](https://github.com/user-attachments/assets/8fd8f22d-abe0-4433-96da-7dddda543ab2)
  
  > Upload Image  
  ![image](https://github.com/user-attachments/assets/34259a7a-5aac-42da-8880-f79e3470cb90)

  메인 화면에서 Create ImageSet 버튼을 누르면 먼저 이름과 해당 이미지 셋에 대한 설명을 적을 수 있는 화면이 나오며 그 이후에 드래그 앤 드롭 혹은 클릭 후 파일 업로드로 이미지를 업로드로 최종 결과화면 전 불량 검출을 위한 이미지를 업로드 할 수 있다.

* 최종 결과 화면  
  > 결과 화면 UI  
  ![image](https://github.com/user-attachments/assets/b47b08ee-06f0-4d4a-b1fc-e1c7c255ed68)  

  이미지 업로드 후에 왼쪽 네비게이터 바에서 업로드한 이미지들 확인할 수 있고 해당 이미지에 적용하고 싶은 AI Model을 적용할 수 있는데 본 연구에서는 VGGNet, ResNet, EfficientNet의 가중치를 적용한 모델을 활용해서 진행하였다.

  Start detection 버튼을 누를 시, 해당 이미지의 파일 이름과, Pass / Fail 로 나뉘는 검사 결과(해당 이미지가 정상이라면 불량 유형은 나오지 않고 검사 결과는 Pass로 나온다.),
  불량 유형은 Edge-Ring, Edge-Local, Center, Local, Scratch, Random, Donut, Near-Full로 총 8가지중 한가지로 도출된다.
<br>

# 기대 효과
- 불량률 감소: AI 기반 이상 탐지 시스템은 인간보다 높은 정확도로 결함을 탐지하여 제품 불량률을 크게 줄일 수 있다. 이는 생산 비용 감소와 수익 증대에 직접적인 영향을 미친다. 

 - 생산량 증대: 불량 제품 생산을 줄임으로써 생산량을 증대시킬 수 있다. 특히, 고수익 제품의 경우 생산량 증대는 수익에 큰 영향을 미칠 수 있다.  

 - 생산 속도 향상: AI 시스템은 실시간으로 대량의 데이터를 처리하여 이상을 빠르게 탐지할 수 있다. 이는 생산 공정의 속도를 높이고 생산성을 향상시킬 수 있다.
<br>

# Reference
[1]  Confusion Matrix, Sharp Sight,
https://www.sharpsightlabs.com/blog/sklearn-confusion_matrix-explained/

[2] VGGNet, 혁펜하임 AI & 딥러닝 강의, https://www.youtube.com/watch?v=TzKJ-Ucyh_I/

[3] Jongkuk Lim, https://www.youtube.com/watch?v=MaDakbMDBrI/

[4] Insight Mountain, https://dlaguddnr.tistory.com/15/

[5] wikidocs, https://wikidocs.net/164796/

[6] Read the Docs,
https://oi.readthedocs.io/en/latest/computer_vision/cnn/vggnet.html/ 

[7] ResNet: Deep Residual Learning for Image Recognition,
https://www.youtube.com/watch?v=671BsKl8d0E/

[8] EfficientNet, https://www.youtube.com/watch?v=uLKqMbOA_vU/

[9] EfficientNet: Rethinking Model Scaling for Convolutional Neural Networks
https://www.youtube.com/watch?v=Vhz0quyvR7I/

[10] Image classification via fine tuning with EfficientNet
https://www.youtube.com/watch?v=ArlkLCJKW54/

[11] Dacon, EfficientNet, https://dacon.io/forum/406054/

[12] Keras 3 API documentation / Keras Applications / EfficientNet B0 to B7
https://keras.io/api/applications/efficientnet/
