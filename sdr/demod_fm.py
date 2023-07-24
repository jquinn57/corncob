import matplotlib.pyplot as plt
import numpy as np
from IPython import embed
from scipy.ndimage import gaussian_filter
from scipy.signal import decimate
import pygame
import os


def play_sound(x):
    x = x.astype(float)
    y = (x - np.min(x)) / (np.max(x) - np.min(x))
    y = ((y-0.5)*120).astype(np.int8)
    sound = pygame.mixer.Sound(y.tobytes())
    sound.play()


f_carrier = 100700000
input_sample_rate = 1024000
T = 10.0
num_samples = input_sample_rate * T
cmd = f'rtl_sdr -f {f_carrier} -s {input_sample_rate} -n {num_samples} temp.dat'
#  rtl_sdr -f 100.7M  -s 1024000 -n 10240000 radio_test2.dat
os.system(cmd)

filename = 'temp.dat'
data = np.fromfile(filename, dtype=np.uint8)
z = np.zeros((len(data)//2), dtype=complex)
z.real = (data[0::2].astype(float)/255) - 0.5
z.imag = (data[1::2].astype(float)/255) - 0.5

modulation = 'FM'
if modulation == 'FM':
    theta = np.unwrap(np.angle(z))
    output = gaussian_filter(theta, sigma=10, order=1)
elif modulation == 'AM':
    output = np.abs(z)

sample_rate = 51200
output = decimate(output, input_sample_rate//(2*sample_rate))
pygame.mixer.init(sample_rate, -16, 1)
print(pygame.mixer.get_init())

play_sound(output)
plt.plot(output)
plt.show()
