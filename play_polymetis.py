# coding: utf-8
import time
import redis
import numpy as np

r = redis.Redis()

x0 = [0.40, 0.00, 0.30]
N = 1000

x, y, z = x0
r.xadd("cart_cmd", {"cmd": "GOTO", "x": x, "y": y})
time.sleep(3)

while 1:
    for i in range(N):
        offset = np.array([0, np.sin((i / N) * np.pi * 2) * 0.1, 0])
        x, y, z = offset + x0
        r.xadd("cart_cmd", {"cmd": "GOTO", "x": x, "y": y})
        print(x0 + offset)
        time.sleep(1 / 50)
