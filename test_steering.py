import pyvjoy
import time

MAX_VJOY = 32767
j = pyvjoy.VJoyDevice(1)

nr = 0

left = True
acc = True

step = 110
time.sleep(3)
while True:
    j.data.wAxisX = nr
    j.data.wAxisY = nr
    j.data.wAxisZ = 0
    j.update()
    if nr >= MAX_VJOY:
        acc = False
    if acc:
        nr += step
    else:
        nr -= step
    if nr == 0:
        acc = True
    time.sleep(1e-8)