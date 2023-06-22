import wave
from IPython import embed
import numpy as np
import matplotlib.pyplot as plt
from scipy.ndimage import gaussian_filter
from scipy.signal import decimate
from scipy.interpolate import interp1d
from PIL import Image

'''
https://webapp1.dlib.indiana.edu/virtual_disk_library/index.cgi/2790181/FID1497/klm/html/c4/sec4-2.htm
https://noaa-apt.mbernardi.com.ar/how-it-works.html
'''

def sync_lines(image):
    # TODO: bias the shift to be close to shift for adjacent lines?
    # or look for both sync A and sync B

    sync_template = -1 * np.ones(39, dtype=float)
    j = 4
    for n in range(7):
        sync_template[j:j+2] = 1
        j += 4

    sync_template = sync_template - sync_template.mean()
    img_debug = np.stack([image] *3, axis=2)
    pts = np.zeros((image.shape[0], 2))
    for line_j in range(image.shape[0]):
        line = image[line_j, :].astype(float) / 127 - 1
        line = line - line.mean()
        xc = np.convolve(line, sync_template, mode='same')
        sync_i = np.argmax(xc)
        pts[line_j, 0] = sync_i
        pts[line_j, 1] = xc[sync_i]
        #print(f'{line_j}: {sync_i}')
        image[line_j, :] = np.roll(image[line_j, :],  15-sync_i)
        img_debug[line_j, sync_i, :] = (255, 0, 0)

    plt.imshow(img_debug)
    plt.show()

    plt.plot(pts[:, 0]/2080, '.', label='position')
    plt.plot(pts[:, 1], label='weight')
    plt.show()

    return image

def extract_channels(image):
    width = 909
    image_A = image[:, 85:85 + width ]
    image_B = image[:, 1126:1126 + width]
    return image_A, image_B

def extract_telemetry(image, chan):
    if chan == 'A':
        tele_img = image[:, 995: 995+45]
    else:
        tele_img = image[:, -45:]
    return tele_img.mean(axis=1)


is_northbound = True
filename = 'argentina.wav'
#filename = '06200232.wav'
wave_reader = wave.open(filename)
N = 11025 # loda 1 second
sample_rate = wave_reader.getframerate()
sample_type = np.int8 if wave_reader.getsampwidth() == 1 else np.int16

data = []
while True:
    samples = wave_reader.readframes(N)
    y = np.frombuffer(samples, dtype=sample_type)
    y = y.astype(float) / np.iinfo(sample_type).max
    if len(y) != N:
        break
    data.append(y)

ys = np.concatenate(data)

N = len(ys)
sample_dt = 1 / sample_rate
carrier_freq = 2400
ts = sample_dt * np.arange(N)
carrier_signal0 = np.sin(2 * np.pi * carrier_freq * ts)
carrier_signal90 = np.cos(2 * np.pi * carrier_freq * ts)

prod_complex = np.zeros(N, dtype=complex)
prod_complex.real = ys * carrier_signal0
prod_complex.imag = ys * carrier_signal90
prod_complex = gaussian_filter(prod_complex, sigma=2.5)

pixels_per_line = 2080
pixels_per_second = 2 * pixels_per_line
num_pixels = int(round(pixels_per_second * ts[-1]))

pixel_ts = np.linspace(ts[0], ts[-1], num_pixels)
prod_interp_fn = interp1d(ts, np.abs(prod_complex))
pixel_intens = prod_interp_fn(pixel_ts)

image = np.reshape(pixel_intens, (-1, pixels_per_line))


image = (image / image.max())
image = np.clip(255 * image, 0, 255).astype(np.uint8)

#image = np.rot90(image, k=2)
# for some reason syncing works better if the image is first rotated
image = sync_lines(image)

image_A, image_B = extract_channels(image)
image_AB = np.concatenate([image_A, image_B], axis=1)

# TODO: fix the rotation, extraction should happen before rotation
# plt.plot(extract_telemetry(image, 'A'), label='A')
# plt.plot(extract_telemetry(image, 'B'), label='B')
# plt.legend()
# plt.show()
image_pil = Image.fromarray(image)
image_pil.show()
