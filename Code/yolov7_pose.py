

#%%
import torch
print(torch.__version__)
print(torch.cuda.is_available())
print(torch.cuda.get_device_name())
#%%
from IPython.display import Image, display
import os
import glob

#%%
%pwd

#%%
## 데이터 분리 - 이미지 나눠져 있는 경우
# train/test/valid 목록생성
from glob import glob
train_list = glob('C:/Users/rmsgh/Desktop/pose/yolov7-pose/Falldown/train/images/*.jpg')
test_list = glob('C:/Users/rmsgh/Desktop/pose/yolov7-pose/Falldown/test/images/*.jpg')
valid_list = glob('C:/Users/rmsgh/Desktop/pose/yolov7-pose/Falldown/valid/images/*.jpg')


len(train_list),  len(valid_list), len(test_list)

#%%
# txt 파일생성
with open('C:/Users/rmsgh/Desktop/pose/yolov7-pose/Falldown/images/train.txt', 'w') as f:
    f.write('/n'.join(train_list) + '/n')
    
# with open('c:/Users/admin/Desktop/yolo/yolov7/www/test/test.txt', 'w') as f:
#     f.write('/n'.join(test_list) + '/n')
    
with open('C:/Users/rmsgh/Desktop/pose/yolov7-pose/Falldown/images/valid.txt', 'w') as f:
    f.write('/n'.join(valid_list) + '/n')

#%%
import os
os.listdir('C:/Users/rmsgh/Desktop/pose/yolov7-pose/Falldown')

#%%
## data.yaml 확인
# 수정전
import yaml
with open('C:/Users/rmsgh/Desktop/pose/yolov7-pose/Falldown/data.yaml', encoding= 'utf-8') as f:
    film = yaml.load(f, Loader=yaml.FullLoader)
    display(film) 

#%%
# 수정후
import yaml
with open('C:/Users/rmsgh/Desktop/pose/yolov7-pose/Falldown/data.yaml', encoding= 'utf-8') as f:
    film = yaml.load(f, Loader=yaml.FullLoader)
    display(film) 

#%%
## yolov7.yaml 확인
# 수정전
import yaml
with open('C:/Users/rmsgh/Desktop/pose/yolov7-pose/cfg/yolov7-w6-pose.yaml', encoding= 'utf-8') as f:
    film = yaml.load(f, Loader=yaml.FullLoader)
    display(film) 

#%%
# 수정후 
import yaml
with open('C:/Users/rmsgh/Desktop/pose/yolov7-pose/cfg/yolov7-w6-pose.yaml', encoding= 'utf-8') as f:
    film = yaml.load(f, Loader=yaml.FullLoader)
    display(film) 

#%%
%cd c:/Users/admin/Desktop/yolo/yolov7

#%%
# 환경설정
pip install -r requirements.txt

#%%
# 학습
python train.py --data ./Falldown/data.yaml --cfg ./cfg/yolov7-w6-pose.yaml --weights ./yolov7-w6-person.pt --batch-size 2 --img 416 --kpt-label --sync-bn --device 0 --name yolov7-w6-pose --hyp ./data/hyp.pose.yaml

