# Drink classification toy project (in microsoft ai school toy project)

## 목차
1. 데이터 소개
2. 데이터 구축 방법
3. 업무 분담
4. 데이터 구축 현황
5. 모델 선택
6. 하이퍼파라미터 최적화 및 Augmentation 시각화
7. 학습 결과
8. WebCam 활용 테스트 결과
9. GUI
10. 회고
11. 참고 자료

<hr>

## 1. 데이터 소개
음료를 웹캠에 인식시켰을 때, 제대로 음료를 잘 분류하는 지에 대한 프로젝트를 진행했습니다. 즉, 10개의 음료를 분류하는 분류기를 제작 후 카메라에 음료를 보여주었을 때, 어떤 음료인지 확인할 수 있는 서비스를 제작하게 되었습니다.

<hr>

## 2. 데이터 구축 방법
데이터는 음료 사진을 직접 촬영하거나 음료 사진의 배경화면을 합성하는 방식으로 구축했습니다. 또한 수집한 이미지를 jpg에서 png로 변환하였을 때 5GB(300장 당)을 차지했는데, 용량이 지나치게 크다고 판단하여 사진들을 224x224사이즈로 리사이즈 하였습니다. 나아가 약 3000x3000정도의 이미지를 224x224로 리사이즈를 하였을 때 이미지가 깨지는 현상이 발생하여, 패딩을 사용해 리사이즈 진행 시 이미지가 깨지는 현상을 해결하였습니다.

```
# 배경합성코드
import glob
import os
from PIL import Image

def image_synthesis(label) :
  img_path = f'./del_back/{label}'
  img_list = glob.glob(os.path.join(img_path, '*.png'))
  back_list = glob.glob(os.path.join('./utils/background/', '*.jpeg'))
  cnt = 0
  for i in img_list :
    for j in back_list :
      my_image = Image.open(j)
      my_image = my_image.resize((224, 224))
      watermark = Image.open(i)
      watermark = watermark.resize((224, 224))  # 배경제거된 상품 이미지 사이즈 결정
      x = my_image.size[0] - watermark.size[0]  # 새로운 배경에 넣을 좌표 설정 부분
      y = my_image.size[1] - watermark.size[1]
      my_image.paster(watermark, (x,y), watermark)  # 배경에 이미지 합성
      my_image.save(f'./test/{label}/{label}_{cnt}.png')
      
      cnt += 1
img_path = './del_back/'
label_list = [jinro', 'choch']
for i in label_list :
  image_synthesis(i)
```

```
# 패딩 후 리사이즈 코드
import os
from PIL import Image

os.chdir('./0106/seven/test/')  # 해당 폴더로 이동
files = os.listdir(os.getcwd()) # 현재 폴더에 있는 모든 파일을 list로 불러오기
cnt = 0
for file in files :
  img = Image.open(file)  # 이미지 불러오기
  img_size = img.size     # 이미지의 크기 측정
  # 직사각형 이미지가 256*512 라면, img_size = (256,512)
  x = img_size[0]
  y = img_size[1]
  
  if x != y :
    size = max(x, y)
    resized_img = Image.new(mode='RGB', size=(size,size), color=(0,0,0))
    offset = (round((abs(x - size)) / 2), round((abs(y - size) / 2 ))
    resized_img.paste(img, offset)
    resized_img = resized_img.resize((224, 224))
    resized_img.save('padding' + str(cnt) + '.png')
    cnt += 1
```

### 1) 음료 촬영 기준 및 예시
⦁	기준: 정방향, 왼쪽 회전, 오른쪽 회전 및 그 외 다양한 위치, 각도에서 이미지 촬영.
⦁	유의 사항: 음료가 잘리거나 한 가지의 배경화면이면 안됨
⦁	초반10장씩 촬영 후 컨펌 실시

![d1](https://user-images.githubusercontent.com/48282708/234458511-c617d55c-6f94-4f1d-b934-67f2deb5f860.png)

### 2) 데이터 수집 폴더 구성
   - data 
      - 상품명(예시:Pokari)
         - Pokari_1.png
         - Pokari_2.png
         - …
         
### 3) 음료(라벨) 리스트

⦁	Bong(봉봉)

⦁	Twopro(이프로)

⦁	Choco(초코에몽)

⦁	Jinro(진로)

⦁	Seven(칠성 사이다)

⦁	Mil(밀키스)

⦁	Pokari(포카리스웨트)

⦁	Cass(카스 라이트)

⦁	Pepper(닥터페퍼)

⦁	Narangd(나랑드 사이다)


<hr>

## 3. 업무 분담
### 1) 팀원별 맡은 음료(직접 찍거나 이미지 배경 합성)

⦁	팀원1 - 봉봉(뚱캔), 이프로(500ml 페트병)

⦁	팀원2 - 포카리스웨트(620ml 페트병), 카스 라이트(500ml 캔)

⦁	팀원3 - 초코에몽(긴우유팩), 진로(소주병)

⦁	남호기 - 밀키스(긴캔), 칠성 사이다(500ml 페트병)

⦁	팀원4 - 나랑드 사이다(긴캔), 닥터페퍼(뚱캔)


<hr>


## 4. 데이터 구축 현황
### 1) 데이터 검수결과

![d2](https://user-images.githubusercontent.com/48282708/234459504-cabedc73-1cdc-40de-8b2f-5c173f82054c.png)

### 2) 데이터 이미지 예시

![d3](https://user-images.githubusercontent.com/48282708/234459946-a4edeabf-b40e-4cf9-a367-22ea3c8c6ae9.png)

### 3) Data Split 결과

![d4](https://user-images.githubusercontent.com/48282708/234460201-27b5b2f7-f430-4b58-919b-e589b4b58bb7.png)

<hr>

## 5. 모델 선택
ResNet18, ResNet34, ResNet50, Efficient_Net_b0, DeiT_tiny 총 5개를 선택했습니다. ResNet18, Efficient_Net_b0, DeIt_tiny은 배웠던 모델들 중 가벼운 편에 속해 선택했고, 나머지 ResNet34, Rest50은 이미지 데이터의 경우 ResNet의 성능이 일반적으로 우수하다는 사전 지식에 근거하여 선정하게 되었습니다.

<hr>

## 6. 하이퍼파라미터 최적화 및 Augmentation 시각화
음료 이미지의 경우 라벨지의 색상이 주요 식별 포인트이므로, 색상을 과도하게 변경시키지 않는 Augmentation들을 선택하였습니다.

![d5](https://user-images.githubusercontent.com/48282708/234460650-f6538b1b-9b22-4990-a0e0-ded4e6672ace.png)


![d6](https://user-images.githubusercontent.com/48282708/234460657-cad2a0d2-f4e0-4816-bbb3-a52f1494d097.png)


![d7](https://user-images.githubusercontent.com/48282708/234460660-55252dfb-1a2b-4954-93e7-97b89e6757b3.png)

<hr>

## 7. 학습 결과 (Train, Validation Loss, Train, Validation Accuracy)
### 1) ResNet18

![d8](https://user-images.githubusercontent.com/48282708/234485385-53a9bba6-b40f-4576-8fd5-cceeacaa7c05.png)

![d9](https://user-images.githubusercontent.com/48282708/234485389-edcdf482-ef2f-4949-8ac9-e9b6a08d25c2.png)

- Batch size = 128, Epoch = 20, Learning Rate = 0.001
- Test 실행 시 99.67% Accuracy 기록

### 2) ResNet34

![d10](https://user-images.githubusercontent.com/48282708/234485399-791f99b2-a0d1-4ff8-aa17-7c177c1c2635.png)

![d11](https://user-images.githubusercontent.com/48282708/234485477-1cf3a528-c187-4bb6-a39b-0cc20ea31c1e.png)

- Batch size = 128, Epoch = 5, Learning Rate = 0.001
- Epoch 5, 10, 20, 35 설정 시, 5이상에서 과적합 현상이 발생

### 3) ResNet50

![d12](https://user-images.githubusercontent.com/48282708/234485489-272b9abe-79da-4edd-9841-f27c191d48d7.png)

![d13](https://user-images.githubusercontent.com/48282708/234485502-45d5c03e-9a6c-4f31-8445-68c5905790d5.png)

- Batch size = 128, Epoch = 35, Learning-Rate = 0.01
- 
![d14](https://user-images.githubusercontent.com/48282708/234485513-3efc5c5f-3dc8-4ecf-84d6-c22628f7cd05.png)

<hr>

## 8. WebCam 활용 테스트 결과
### 1) ResNet18

![d15](https://user-images.githubusercontent.com/48282708/234485520-10d449c4-4c25-49fb-a241-0859a109215d.png)

### 2) ResNet34

![d16](https://user-images.githubusercontent.com/48282708/234485526-be6d8657-86f5-48cc-b5f9-a82509c49a0b.png)

### 3) ResNet50
이프로 음료를 제외한 모든 라벨의 인식이 가능하지만 정확도를 고려하였을때 초기단계 모델의 한계성이 보입니다.

![d17](https://user-images.githubusercontent.com/48282708/234485537-f5900f4f-4755-4dda-97c4-f2987606303e.png)

<hr>

## 9. GUI
부트스트랩 웹페이지 테마를 기반으로 간단하게 웹 캠 표시, 웹 캠을 켜고 끄기, 캡쳐 기능 제작했습니다.
Start bootstrap : https://startbootstrap.com/

![d18](https://user-images.githubusercontent.com/48282708/234486553-0cd60520-2e83-44c3-b9d8-0c6cd3835168.png)

![d19](https://user-images.githubusercontent.com/48282708/234486555-28e14f89-0b00-466b-b39d-6dbf75399d00.png)

![d20](https://user-images.githubusercontent.com/48282708/234486557-7f807e20-d3c5-468b-b964-5dc75b954fa8.png)

웹페이지에 on/off 버튼과 capture 버튼을 추가하여 사용성 확장

<hr>

## 10. 회고

### 1) 데이터(Train 이미지)편향
- 합성 이미지 : 한 가지 음료를 300장 이미지로 계획하였고 음료의 위치와 각도가 다른 이미지 15장과 서로 다른 배경 20장을 합성해 데이터를 완성시켰습니다. 본래 수집한 여러 위치(각도 포함) 이미지는 42장이며 배경은 70여장정도를 수집하였습니다. 처음에 개수를 고려하지 않고 합성하여 1,800장 정도를 만들었었는데 여기서 랜덤으로 뽑아서 300장의 데이터를 구축하는 방법을 사용해야 더 좋은 데이터가 될 수 있었다고 생각합니다.
- 직접 찍은 이미지 : 한 가지 음료의 300장을 모두 찍어서 데이터를 구축하였고 배경과 음료 위치와 각도가 매우 다양하여 좋은 데이터로 보여졌으나, 우리가 선정한 학습모델로 테스트를 진행했을 때 테스트 환경과 유사한 벽지(색)와 조명이 배경으로 많이 포함된 이미지일수록 인식률이 달라지게 되는 것을 발견하였고 이후 진행되는 프로젝트에서는 환경적 요소에 따른 변수까지 고려하는 방향으로 논의하였습니다.

### 2) 사용자 웹캠 카메라에 따라 인식률의 차이를 보인다.
- 데이터 수집에 사용했던 카메라에 따라 인식률이 좋아지는 경향을 보임

### 3) 데이터의 배경화면과 시간, 조명에 따라 인식에 차이가 생긴다.
- 같은 위치에서 오전, 오후, 저녁에 웹캠으로 음료를 인식하여 분류하였을 때, 오전에는 잘 분류가 되던 음료가 오후에는 분류가 잘되지 않는 현상이 발생하였고, 저녁에는 다른 시간 때 음료 분류가 잘되지 않던 게 잘 분류되는 현상이 발생하게 되었습니다. 
- 데이터를 직접 다루어 보면서 들게 된 것은 데이터의 수가 많고 적음 보다 이미지 데이터의 시간, 조명 배경 등 여러 가지 환경적인 요소들이 음료 분류를 위한 인식에 영향을 끼친다는 것을 알 수 있는 계기가 되었고 차후 진행되는 팀프로젝트에서 개선해야할 방향과 데이터 측면에서 고려해야할 부분들에 대한 시각을 넓히게 되었습니다.

<hr>

## 11. 참고 자료
   I. Video Streaming Using Webcam in Flask Web Framework
(https://github.com/krishnaik06/Flask-Web-Framework/tree/main/Tutorial%207)
