import time
import cv2
import numpy as np
from grabscreen import grab_screen
from getkeys import pressed_keys
import datetime

import pygame

DEBUG = False
HOST = "localhost"
PORT = "50000"
WHEEL = "G27 Racing Wheel"


def pedal_value(value):
    '''
    Steering Wheel returns pedal reading as value
    between -1 (fully pressed) and 1 (not pressed)
    normalizing to value between 0 and 100%
    '''
    return (1 - value) * 5


pygame.init()

wheel = None
for j in range(0, pygame.joystick.get_count()):
    if pygame.joystick.Joystick(j).get_name() == WHEEL:
        wheel = pygame.joystick.Joystick(j)
        wheel.init()
        print(wheel.get_axis(4))
        print("Found", wheel.get_name())
if not wheel:
    print("No G27 steering wheel found")
    exit(-1)

paused = False

f = open('data\\F1_Australia_Line\\driving_log.csv', 'a+')

acc = 0.0
brake = 0.0
steer = 0.0
time.sleep(10)

while 'Screen capturing':
    if not paused:
        last_time = time.time()

        for event in pygame.event.get(pygame.QUIT):
            exit(0)
        for event in pygame.event.get(pygame.JOYAXISMOTION):
            if DEBUG:
                print("Motion on axis: ", event.axis)
            if event.axis == 0:
                steer = event.value * 100
            elif event.axis == 2:
                acc = pedal_value(event.value)
            elif event.axis == 3:
                brake = pedal_value(event.value)

        # Get raw pixels from the screen, save it to a Numpy array
        img = np.array(grab_screen(region=(0, 40, 1280, 758)))

        img = cv2.resize(img, (320, 180))
        # run a color convert:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        image_name = datetime.datetime.now().strftime('%Y-%m-%d-%H-%M-%S-%f') + '.jpg'
        f.write('{},{},{},{}\n'.format(image_name, acc, brake, steer))
        cv2.imwrite('data\\F1_Australia_Line\\' + image_name, img)
        # Display the picture
        #cv2.imshow('OpenCV/Numpy normal', img)

        # Display the picture in grayscale
        # cv2.imshow('OpenCV/Numpy grayscale',
        #            cv2.cvtColor(img, cv2.COLOR_BGRA2GRAY))

        #print('fps: {0}'.format(1 / (time.time()-last_time)))

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

f.close()