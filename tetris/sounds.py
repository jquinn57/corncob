import sys, pygame
import numpy as np
import time
import matplotlib.pyplot as plt

def play_sound(x):
    x = x.astype(np.float)
    y = (x - np.min(x)) / (np.max(x) - np.min(x))
    y = ((y-0.5)*100).astype(np.int8)
    sound = pygame.mixer.Sound(y.tobytes())
    sound.play()

sample_rate = 22050
pygame.mixer.init(sample_rate, -16, 1)
print(pygame.mixer.get_init())

T = 4
ts = np.arange(0, T*sample_rate)/sample_rate
f = 220
z = np.zeros_like(ts)
for n in range(1,2):
    z = z +  (1/n) * np.sin(2*np.pi*f*ts*n)

z = z * np.sin(2*np.pi*ts*(f-4))
play_sound(z)

plt.plot(ts, z)
plt.show()

