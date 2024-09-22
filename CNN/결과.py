# 데이터
import torch
from torchvision import transforms
from PIL import Image

# 학습
import torch.nn as nn
from torchvision import models

# 추론
from PIL import Image
import cv2

transform = transforms.Compose([
    transforms.Grayscale(num_output_channels=1),  # 이미지를 그레이스케일로 변환
    transforms.Resize((224, 224)),  # CNN에 넣기 위한 이미지 크기 조정
    transforms.ToTensor(),  # 텐서로 변환 (H, W, C -> C, H, W)
])
category = {0:'Up', 1:'Down'}
device = 'cuda' if torch.cuda.is_available() else 'cpu'

# 추론
def inference_path(img_path):
  
    img = Image.open(img_path)
    
    max_size = max(img.size) # 큰 사이즈에 맞춘다
    resized_img = Image.new(mode='RGB', size = (max_size, max_size), color = (0,0,0))
    offset = (round((abs(img.size[0] - max_size)) / 2), round((abs(img.size[1] - max_size)) / 2))
    resized_img.paste(img, offset)
    
    img = transform(resized_img)
    img = img.unsqueeze(0)
    
    
    model.eval()
    with torch.no_grad():
      pred = model(img.to(device))
      # print(f'pred: {pred}')
      result = pred.max(dim=1)[1] # torch.max(pred, 1) 이렇게도 됨
      # print(f'result: {result.item()}')
      return result.item()


model = models.efficientnet_b0(pretrained=True)
# 1차원으로
model.features[0][0] = nn.Conv2d(1, 32, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1), bias=False)
# freeze
for param in model.parameters():
    param.requires_grad = False
    fc = nn.Sequential(
    nn.Dropout(p=0.2, inplace=True),
    nn.Linear(model.classifier[1].in_features, 2),
)
model.classifier = fc

model.load_state_dict(torch.load('CNN/efficient_epoch18.pth'))
model.to(device)


cap = cv2.VideoCapture(0)
width = 1280
height = 720
cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
while cap.isOpened():
    ret, frame = cap.read()
    
    frame= cv2.flip(frame, 1)
    
    if not ret:
        break
    
    cv2.imwrite('CNN/here.jpg', frame)
    res = inference_path('CNN/here.jpg')
    cv2.putText(frame, category[res], (0, 100), cv2.FONT_HERSHEY_SIMPLEX, 2, (0,0,0), 3)
    cv2.imshow('', frame)
    
    if cv2.waitKey(1) == 27:
        break
    
cap.release()
cv2.destroyAllWindows()

# print(category[inference_path('here.jpg')])