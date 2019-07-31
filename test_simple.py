from models.nvidia_lstm import LSTMModel
from models.simple_model import MyModel
from models.backbone import ResNetBackbone
from data_loading.F1DatasetSimple import ToTensor, Crop, Normalize
from torchvision import transforms
import torch
import numpy as np
import cv2
from grabscreen import grab_screen
from getkeys import pressed_keys
import time
import pyvjoy

MAX_VJOY = 32767
j = pyvjoy.VJoyDevice(1)

model = ResNetBackbone(no_outputs=3)
# 11_259, s11_232[best] s11_290 s1_277
model.load_state_dict(torch.load('C:\\Users\\Andrei\\workspace\\Robotics\\checkpoints\\checkpoint_train_resnet_328'))
model = model.cuda()
model.eval()

paused = False

sequence_length = 3
frame_delta = 10
image_sequence = []

seq_no = 0

def normalize(val, min_value, max_value, scale):
    result = ((val - min_value) / (max_value - min_value)) * scale
    return result

while True:
    if not paused:
        # Get raw pixels from the screen, save it to a Numpy array
        img = np.array(grab_screen(region=(0, 40, 1280, 758)))

        img = cv2.resize(img, (320, 180))
        # run a color convert:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        seq_no += 1

        np_image_sequence = np.array(img)
        tensor_image_sequence = transforms.Compose([Crop((0.4, 0.39, 0.13, 0.13)), ToTensor(), Normalize()])\
            ({'images': np_image_sequence, 'commands': np.array(0)})

        with torch.no_grad():
            outputs = model(tensor_image_sequence['images'].unsqueeze(0).cuda())
            j.data.wAxisX = outputs[0][2] * MAX_VJOY
            j.data.wAxisY = outputs[0][0] * MAX_VJOY
            j.data.wAxisZ = outputs[0][1] * MAX_VJOY
            j.update()

        # Display the picture
        #cv2.imshow('OpenCV/Numpy normal', img)

        # Press "q" to quit
        if cv2.waitKey(25) & 0xFF == ord('q'):
            cv2.destroyAllWindows()
            break

    keys = pressed_keys()
    if 'P' in keys:
        if paused:
            paused = False
            print('Unpaused!')
            time.sleep(1)
        else:
            print('Pausing!')
            paused = True
            time.sleep(1)
    if 'Q' in keys:
        cv2.destroyAllWindows()
        break
