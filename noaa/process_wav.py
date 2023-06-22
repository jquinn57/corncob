import wave
from IPython import embed
import numpy as np
import matplotlib.pyplot as plt
from scipy.ndimage import gaussian_filter
from scipy.signal import decimate
from scipy.interpolate import interp1d
from PIL import Image

filename = 'argentina.wav'
#filename = '06200232.wav'
wave_reader = wave.open(filename)
N = 11025 //2
sample_rate = wave_reader.getframerate()
sample_dt = 1 / sample_rate
carrier_freq = 2400
ts = sample_dt * np.arange(N)
carrier_signal0 = np.sin(2 * np.pi * carrier_freq * ts)
carrier_signal90 = np.cos(2 * np.pi * carrier_freq * ts)
pixels_per_line = 2080  # actually can be anything
sample_type = np.int8 if wave_reader.getsampwidth() == 1 else np.int16
pixel_ts = np.linspace(ts[0], ts[-1], pixels_per_line)
prod_complex = np.zeros(N, dtype=complex)

num_lines = 2000
image = np.zeros((num_lines, pixels_per_line))

is_northbound = True
error = 0

for line_num in range(num_lines):
    if error > 1:
        error_int = int(round(error))
        wave_reader.readframes(error_int)
        error = error - error_int
    error += 0.75

    samples = wave_reader.readframes(N)
    y = np.frombuffer(samples, dtype=sample_type)
    y = y.astype(float) / np.iinfo(sample_type).max
    if len(y) != N:
        break

    # I Q demod
    prod_complex.real = y * carrier_signal0
    prod_complex.imag = y * carrier_signal90
    prod_complex = gaussian_filter(prod_complex, sigma=2.5)

    prod_interp_fn = interp1d(ts, np.abs(prod_complex))
    pixel_intens = prod_interp_fn(pixel_ts)

    image[line_num, :] = pixel_intens

image = (255 * image / image.max())
image = np.clip(image, 0, 255).astype(np.uint8)

if is_northbound:
    image = np.rot90(image, k=2)
offset = -220

image_pil = Image.fromarray(np.roll(image, offset, axis=1))
image_pil.show()
