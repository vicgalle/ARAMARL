import numpy as np

from agent import DummyAgent, RandomAgent

adv = DummyAgent([0, 1, 2, 3])
a = adv.act()
print(a)
a = adv.act()
print(a)


adv = RandomAgent([0, 1], p=0.1)
cont, n_iter = 0., 10000
for i in range(n_iter):
    a = adv.act()
    if a == 0:
        cont += 1
print(cont/n_iter)
