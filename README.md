# Drink classification 3day project (in microsoft ai school toy project)

<hr>

## 1. 데이터 소개
음료를 웹캠에 인식시켰을 때, 제대로 음료를 잘 분류하는 지에 대한 프로젝트를 진행했습니다. 즉, 10개의 음료를 분류하는 분류기를 제작 후 카메라에 음료를 보여주었을 때, 어떤 음료인지 확인할 수 있는 서비스를 제작하게 되었습니다.

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


## 3. 업무 분담
### 1) 팀원별 맡은 음료(직접 찍거나 이미지 배경 합성)

⦁	팀원1 - 봉봉(뚱캔), 이프로(500ml 페트병)

⦁	팀원2 - 포카리스웨트(620ml 페트병), 카스 라이트(500ml 캔)

⦁	팀원3 - 초코에몽(긴우유팩), 진로(소주병)

⦁	남호기 - 밀키스(긴캔), 칠성 사이다(500ml 페트병)

⦁	팀원4 - 나랑드 사이다(긴캔), 닥터페퍼(뚱캔)



## 4. 데이터 구축 현황
### 1) 데이터 검수결과

![d2](https://user-images.githubusercontent.com/48282708/234459504-cabedc73-1cdc-40de-8b2f-5c173f82054c.png)

### 2) 데이터 이미지 예시

![d3](https://user-images.githubusercontent.com/48282708/234459946-a4edeabf-b40e-4cf9-a367-22ea3c8c6ae9.png)

### 3) Data Split 결과

![d4](https://user-images.githubusercontent.com/48282708/234460201-27b5b2f7-f430-4b58-919b-e589b4b58bb7.png)

## 5. 모델 선택
ResNet18, ResNet34, ResNet50, Efficient_Net_b0, DeiT_tiny 총 5개를 선택했습니다. ResNet18, Efficient_Net_b0, DeIt_tiny은 배웠던 모델들 중 가벼운 편에 속해 선택했고, 나머지 ResNet34, Rest50은 이미지 데이터의 경우 ResNet의 성능이 일반적으로 우수하다는 사전 지식에 근거하여 선정하게 되었습니다.

## 6. 하이퍼파라미터 최적화 및 Augmentation 시각화
음료 이미지의 경우 라벨지의 색상이 주요 식별 포인트이므로, 색상을 과도하게 변경시키지 않는 Augmentation들을 선택하였습니다.

![d5](https://user-images.githubusercontent.com/48282708/234460650-f6538b1b-9b22-4990-a0e0-ded4e6672ace.png)


![d6](https://user-images.githubusercontent.com/48282708/234460657-cad2a0d2-f4e0-4816-bbb3-a52f1494d097.png)


![d7](https://user-images.githubusercontent.com/48282708/234460660-55252dfb-1a2b-4954-93e7-97b89e6757b3.png)

## 7. Train, Validation Loss, Train, Validation Accuracy 결과
