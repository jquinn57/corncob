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

def sync_lines(image, show_debug=False):

    sync_templateA = -1 * np.ones(39, dtype=float)
    j = 4
    for n in range(7):
        sync_templateA[j:j+2] = 1
        j += 4

    sync_templateA = sync_templateA - sync_templateA.mean()


    sync_templateB = -1 * np.ones(39, dtype=float)
    j = 4
    for n in range(7):
        sync_templateB[j:j+3] = 1
        j += 5

    sync_templateB = sync_templateB - sync_templateB.mean()

    img_debug = np.stack([image] *3, axis=2)
    pts = np.zeros((image.shape[0], 4))
    for line_j in range(image.shape[0]):
        line = image[line_j, :].astype(float) / 127 - 1
        line = line - line.mean()
        xc = np.correlate(line, sync_templateA, mode='same')
        sync_i = np.argmax(xc)
        pts[line_j, 0] = sync_i
        pts[line_j, 1] = xc[sync_i]
        img_debug[line_j, sync_i, :] = (255, 0, 0)

        xc = np.correlate(line, sync_templateB, mode='same')
        sync_i = np.argmax(xc)
        pts[line_j, 2] = sync_i
        pts[line_j, 3] = xc[sync_i]
        img_debug[line_j, sync_i, :] = (0, 255, 0)

    # could add additional logic here to smooth out sync if needed
    delta = np.abs(pts[:, 0] - pts[:, 2]) - image.shape[1] // 2
    is_good = np.abs(delta) < 4
    shifts = np.zeros(image.shape[0], dtype=int)
    prev_shift = 0
    for line_j in range(image.shape[0]):
        if is_good[line_j]:
            shifts[line_j] = pts[line_j, 0]
            prev_shift = pts[line_j, 0]
        else:
            shifts[line_j] = prev_shift
        image[line_j, :] = np.roll(image[line_j, :],  len(sync_templateA)//2 - shifts[line_j])

    if show_debug:
        plt.imshow(img_debug)
        plt.show()
        plt.plot(pts[:, 0], '.', label='start A')
        plt.plot(pts[:, 2], '.', label='start B')
        plt.plot(shifts, '.', label='shift')
        plt.legend()
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

    x = tele_img.mean(axis=1)
    x_max = x.max()
    x_min = x.min()
    x = (x - x_min) / (x_max - x_min)

    # each wedge is 8 lines long
    template = np.zeros(9 * 8, dtype=float)
    wedge_targets = [31, 63, 95, 127, 159, 191, 224, 255, 0]
    for i, target in enumerate(wedge_targets):
        template[i * 8: (i + 1) * 8] = target / 255


    xc = np.correlate(x, template, mode='same')
    # index of start of best set of wedges
    i_start = np.argmax(xc) - len(template) // 2
    # probably should use all repeats instead of just selecting one (there could be errors in the important part)
    x = x[i_start: i_start + 16 * 8]
    x = np.reshape(x, (-1, 8))
    x =  x.mean(axis=1)
    # unnormalize
    x = x * (x_max - x_min) + x_min
    return x

def identify_channel(tele):
    names = ['1', '2', '3A', '4', '5', '3B']
    wavelengths_min = [0.58, 0.725, 1.58, 10.30, 11.50, 3.55]
    wavelengths_max = [0.68, 1.00, 1.64, 11.30, 12.50, 3.93]
    description = ['Visible, Daytime cloud and surface mapping',
                   'Near-infrared, Land-water boundaries',
                   'Infrared, Snow and ice detection',
                   'Infrared, Night cloud mapping, sea surface temperature',
                   'Infrared, Sea surface temperature',
                   'Infrared, Night cloud mapping, sea surface temperature']
    targets = tele[:6]
    dist = np.abs(targets - tele[-1])
    i = np.argmin(dist)

    chan_data = {'name': names[i], 'wavelengths': [wavelengths_min[i], wavelengths_max[i]], 'description': description[i]}
    return chan_data

def normalize_image(image, tele):
    max_wedge = tele[7]
    min_wedge = tele[8]
    img = (image.astype(float) - min_wedge) / (max_wedge - min_wedge)
    img = np.clip(255 * img, 0, 255).astype(np.uint8)
    return img


def process_wav(filename, is_northbound=True):

    wave_reader = wave.open(filename)
    N = 11025  # load 1 second at a time
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

    image = sync_lines(image)

    tele_A = extract_telemetry(image, 'A')
    tele_B = extract_telemetry(image, 'B')

    chan_A = identify_channel(tele_A)
    chan_B = identify_channel(tele_B)

    image = normalize_image(image, tele_A)

    image_A, image_B = extract_channels(image)
    image_AB = np.stack([image_A, image_A, image_B], axis=2)

    if is_northbound:
        image_AB = np.rot90(image_AB, k=2)
    return image_AB, chan_A, chan_B


if __name__ == '__main__':

    image_AB, chan_A, chan_B = process_wav('argentina.wav')
    print(chan_A)
    print(chan_B)
    image_pil = Image.fromarray(image_AB)
    image_pil.show()
