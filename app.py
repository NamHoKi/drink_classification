import datetime

from flask import Flask, render_template, Response, url_for, redirect, send_from_directory
import cv2
import numpy as np
import torch
import torch.nn
import albumentations as A
from albumentations.pytorch.transforms import ToTensorV2
import torchvision
from torch.autograd import Variable
from torchvision import transforms
import PIL
from torchvision import models
import torch.nn as nn
import cv2
import torch.nn.functional as F
from PIL import Image, ImageFont, ImageDraw

# This is the Label
Labels = {0: 'Bong',
          1: 'Cass',
          2: 'Choco',
          3: 'Jinro',
          4: 'Mil',
          5: 'Narangd',
          6: 'Pepper',
          7: 'Pokari',
          8: 'Seven',
          9: 'TwoPro'
          }

# app 모듈 이름, 경로 저장
app = Flask(__name__, template_folder='./html', static_folder='./html/css')

camera = cv2.VideoCapture(-1)
global push_btn
push_btn = True
global capture_btn
capture_btn = False
camera.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
camera.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)


data_transforms = transforms.Compose(
    [
        transforms.Resize((224, 224)),
        # transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize([0.485,0.456,0.406],[0.229,0.224,0.225])
    ]
)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")  ##Assigning the Device which will do the calculation

# model = torch.load("Resnet50_Left_Pretraine%d_ver1.1.pth") #Load model to CPU
model = models.resnet18()
model.fc = nn.Linear(in_features=512, out_features=10)
# model.load_state_dict(torch.load("resnet18_AdamW_Lr001_E40.pt", map_location=torch.device('cpu')))
model.load_state_dict(torch.load("best.pt", map_location=torch.device('cpu')))
model = model.to(device)  # set where to run the model and matrix calculation
model.eval()  # set the device to eval() mode for testing


def argmax(prediction):
    prediction = prediction.cpu()
    prediction = prediction.detach().numpy()
    top_1 = np.argmax(prediction, axis=1)
    score = np.amax(prediction)
    score = '{:6f}'.format(score)
    prediction = top_1[0]
    result = Labels[prediction]
    return result, score


def preprocess(image):
    image = PIL.Image.fromarray(image)  # Webcam frames are numpy array format
    # Therefore transform back to PIL image
    image = data_transforms(image)
    image = image.float()
    # image = Variable(image, requires_autograd=True)
    image = image.cpu()
    image = image.unsqueeze(0)  # I don't know for sure but Resnet-50 model seems to only
    # accpets 4-D Vector Tensor so we need to squeeze another
    return image  # dimension out of our 3-D vector Tensor


# Let's start the real-time classification process!

show_score = 0
show_res = 'Nothing'
sequence = 0


def generate_frames():
    global push_btn  # push_btn을 전역변수로 지정
    global capture_btn  # capture_btn을 전역변수로 지정
    global show_res
    global show_score
    while True:
        now = datetime.datetime.now()  # 현재시각 받아옴
        nowDatetime = now.strftime('%Y-%m-%d %H:%M:%S')  # 현재시각을 문자열 형태로 저장
        nowDatetime_path = now.strftime('%Y-%m-%d %H_%M_%S')
        ref, frame = camera.read()  # Capture each frame
        if not ref:
            break
        else:
            if push_btn:                            # 버튼을 눌렀을때(화면을 끄면)
                frame = np.zeros([480, 640, 3], dtype="uint8")      # 검은화면을 생성
                frame = Image.fromarray(frame)
                frame = np.array(frame)
            else:                                   # 버튼을 누르지 않은 상태(화면을 켜면)
                frame = Image.fromarray(frame)
                frame = np.array(frame)
                frame1 = frame

                image = frame[200:550, 250:670]
                # image = cv2.normalize(image, None, 80, 220, cv2.NORM_MINMAX)
                image_data = preprocess(image)
                prediction = model(image_data)

                softmax_result = F.softmax(prediction)
                result, score = torch.topk(softmax_result, 1)
                acc = ": " + str(round(result.item() * 100, 3)) + "%"

                if float(round(result.item() * 100, 3)) > 50:
                    # if score > 1.7:
                    show_res = Labels[score.item()]
                    show_score = acc
                    # show_res = result
                    # show_score = score
                else:
                    show_res = "Nothing"
                    show_score = acc

                cv2.putText(frame, '%s' % (show_res), (380, 80), cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 255, 0), 3)
                cv2.putText(frame, '{}'.format(show_score), (400, 120), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0),
                            2)  # 수정한거


                cv2.rectangle(frame, (10, 10), (630, 470), (0, 255, 0), 2)
                # cv2.imshow("ASL SIGN DETECTER", frame)

        ref, buffer = cv2.imencode('.png', frame)
        frame = buffer.tobytes()

        if capture_btn:  # is_capture가 참이면
            capture_btn = False  # is_capture를 거짓으로 만들고
            cv2.imwrite("capture " + nowDatetime_path + ".png", frame1)  # 이미지로 영상을 캡쳐하여 그림파일로 저장

        yield (b'--frame\r\n'
                b'Content-Type: image/png\r\n\r\n' + frame + b'\r\n')





# index.html 파일을 로더 한다.
@app.route('/')  # 127.0.0.1
def index():
    global push_btn
    global capture_btn
    return render_template('index.html', push_btn=push_btn, capture_btn=capture_btn)


# 웹캠 비디오를 로드한다.
@app.route('/video_feed')
def video_feed():
    return Response(generate_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')


# on/off 버튼을 지정한다
@app.route('/push_switch')
def push_switch():                                  # 버튼을 눌렀을때 실행되는 함수
    global push_btn                                 # push_btn을 전역변수로 불러옴
    push_btn = not push_btn                         # push_btn의 상태를 토글
    return redirect(url_for('index'))


@app.route('/capture_switch')
def capture_switch():                               # capture_btn을 눌렀을때 실행되는 함수
    global capture_btn                              # capture_btn을 전역변수로 불러옴
    capture_btn = not capture_btn                   # capture_btn을 참으로 만듬
    return redirect(url_for('index'))


if __name__ == '__main__':
    app.run(host="localhost", port=8000, debug=True)
