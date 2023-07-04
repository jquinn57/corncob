import wave
import numpy as np
from PIL import Image
import argparse
from skimage.exposure import equalize_adapthist

'''
Decode a wav recording from NOAA 15, 18, 19 satellites which use the APT format.
The wav file should be mono with 11,025 samples per second, but higher sample rates or stereo will work too.


References:
https://webapp1.dlib.indiana.edu/virtual_disk_library/index.cgi/2790181/FID1497/klm/html/c4/sec4-2.htm
https://noaa-apt.mbernardi.com.ar/how-it-works.html
'''

class APTProcessor():
    def __init__(self):
        pass


    def sync_lines(self, image):
        '''
        input: image: grayscale image as ndarray of uint8
        output: image: same as input but with each line shifted so that 
        all lines start at column 0 
        '''

        # channel A sync signal
        sync_templateA = -1 * np.ones(39, dtype=float)
        j = 4
        for n in range(7):
            sync_templateA[j:j+2] = 1
            j += 4
        sync_templateA = sync_templateA - sync_templateA.mean()

        # channel B sync signal
        sync_templateB = -1 * np.ones(39, dtype=float)
        j = 4
        for n in range(7):
            sync_templateB[j:j+3] = 1
            j += 5
        sync_templateB = sync_templateB - sync_templateB.mean()

        pts = np.zeros((image.shape[0], 4))
        for line_j in range(image.shape[0]):
            line = image[line_j, :].astype(float) / 127 - 1
            line = line - line.mean()
            xc = np.correlate(line, sync_templateA, mode='same')
            sync_i = np.argmax(xc)
            pts[line_j, 0] = sync_i
            pts[line_j, 1] = xc[sync_i]

            xc = np.correlate(line, sync_templateB, mode='same')
            sync_i = np.argmax(xc)
            pts[line_j, 2] = sync_i
            pts[line_j, 3] = xc[sync_i]

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

        return image

    def extract_channels(self, image):
        '''
        input: image: grayscale image as ndarray of uint8, assumed to be aligned
        output: image_A, image_B crops of the chan A and B images
        '''
        width = 909
        image_A = image[:, 85:85 + width ]
        image_B = image[:, 1126:1126 + width]
        return image_A, image_B

    def extract_telemetry(self, image, chan):
        '''
        select the cleanest set of telemetry data and average across x and y
        input: image: grayscale image as ndarray of uint8, assumed to be aligned
                chan: either 'A' or 'B' to select channel 
        output: telemetry_vals: ndarray of 16 values starting with the 1st telemetry wedge
        '''
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
        # probably should use all repeats instead of just selecting one
        x = x[i_start: i_start + 16 * 8]
        x = np.reshape(x, (-1, 8))
        x =  x.mean(axis=1)
        # unnormalize
        telemetry_vals = x * (x_max - x_min) + x_min
        return telemetry_vals

    def identify_channel(self, tele):
        '''
        figure out which sensor the image came from based on telemetry data
        by comparing the last entry to the first 6 reference wedges
        input: tele: ndarray of 16 telemetry values
        output: chan_data: dict with name and description of sensor
        https://noaa-apt.mbernardi.com.ar/how-it-works.html
        '''
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

        chan_data = {'name': names[i], 'wavelengths': [wavelengths_min[i], wavelengths_max[i]],
                    'description': description[i]}
        return chan_data

    def normalize_image(self, image, tele, use_clahe=True):
        '''
        Normalize the image
        input: image: either imageA or imageB after cropping
                tele: telemetry values
                use_clahe: if True use the histogram equilization, 
                        if False use the telemetry values
        output: img: image after normalization has been applied
        '''
        if use_clahe:
            img = (255 * equalize_adapthist(image)).astype(np.uint8)
        else:
            max_wedge = tele[7]
            min_wedge = tele[8]
            img = (image.astype(float) - min_wedge) / (max_wedge - min_wedge)
            img = np.clip(255 * img, 0, 255).astype(np.uint8)
        return img

    def apply_false_color(self, image_A, image_B, palette='noaa-apt-daylight.png'):
        '''
        apply false color by looking up each grayscale pixel pair (imageB, imageA) 
        into the palette which maps from (imageA, imageB) -> (R, G, B)
        '''
        # may want to add scaling to A and B before the lookup
        pal_img = np.array(Image.open(palette))
        img2 = pal_img[image_B.flatten(), image_A.flatten()]
        img2 = img2.reshape( (image_A.shape[0], image_A.shape[1], -1))
        return img2


    def decode_wav(self, filename):
        '''
        Turn the wav file into a grayscale image 
        input: filename: path to the wav file
        output: image: grayscale image as ndarray of uint8
        '''

        pixels_per_line = 2080
        carrier_freq = 2400

        wave_reader = wave.open(filename)
        sample_rate = wave_reader.getframerate()
        sigma = sample_rate / (2 * pixels_per_line)

        N_chunk = sample_rate  # load 1 second at a time (2 lines)
        sample_type = np.int8 if wave_reader.getsampwidth() == 1 else np.int16

        data = []
        while True:
            samples = wave_reader.readframes(N_chunk)
            y = np.frombuffer(samples, dtype=sample_type)
            if wave_reader.getnchannels() == 2:
                y = y[::2]  # use every other sample if stereo
            y = y.astype(float) / np.iinfo(sample_type).max
            if len(y) != N_chunk:
                break
            data.append(y)

        wave_reader.close()
        ys = np.concatenate(data)

        N = len(ys)
        sample_dt = 1 / sample_rate
        ts = sample_dt * np.arange(N)
        carrier_signal0 = np.sin(2 * np.pi * carrier_freq * ts)
        carrier_signal90 = np.cos(2 * np.pi * carrier_freq * ts)

        prod_complex = np.zeros(N, dtype=complex)
        prod_complex.real = ys * carrier_signal0
        prod_complex.imag = ys * carrier_signal90

        nfilter =  int(round(8 * sigma)) + 1
        xs = np.arange(nfilter) - 0.5 * nfilter
        guassian_kernel = np.exp(-xs **2 / (2 * sigma ** 2))
        prod_complex = np.convolve(prod_complex, guassian_kernel, mode='same')

        pixels_per_second = 2 * pixels_per_line
        num_pixels = int(round(pixels_per_second * ts[-1]))

        pixel_ts = np.linspace(ts[0], ts[-1], num_pixels)
        pixel_intens = np.interp(pixel_ts, ts, np.abs(prod_complex))

        image = np.reshape(pixel_intens, (-1, pixels_per_line))
        image = (image / image.max())
        image = np.clip(255 * image, 0, 255).astype(np.uint8)
        return image



    def process(self, filename, is_northbound=True, pal=None, use_clahe=False):

        image = self.decode_wav(filename)
        image = self.sync_lines(image)

        tele_A = self.extract_telemetry(image, 'A')
        tele_B = self.extract_telemetry(image, 'B')

        chan_A = self.identify_channel(tele_A)
        chan_B = self.identify_channel(tele_B)

        image_A, image_B = self.extract_channels(image)

        image_A = self.normalize_image(image_A, tele_A, use_clahe=use_clahe)
        image_B = self.normalize_image(image_B, tele_B, use_clahe=use_clahe)

        if pal is not None:
            image_AB = self.apply_false_color(image_A, image_B, palette=pal)
        else:
            image_AB = np.concatenate([image_A, image_B], axis=1)

        if is_northbound:
            image_AB = np.rot90(image_AB, k=2)
        return image_AB, chan_A, chan_B


if __name__ == '__main__':

    parser = argparse.ArgumentParser(description="NOAA APT Processor")
    parser.add_argument('--input', default=None, help='Path to wav file')
    parser.add_argument('--pal', default=None, help='Path to Palette PNG')
    parser.add_argument('--southbound', action='store_true', help='Satellite was going North to South')
    parser.add_argument('--use_clahe', action='store_true', help='Apply histogram equilization')
    parser.add_argument('--save', action='store_true', help='Save output images')
    args = parser.parse_args()

    apt_processor = APTProcessor()
    image_AB, chan_A, chan_B = apt_processor.process(args.input, is_northbound=not args.southbound, pal=args.pal, use_clahe=args.use_clahe)
    
    print(f'{chan_A=}')
    print(f'{chan_B=}')

    image_pil = Image.fromarray(image_AB)
    if args.save:
        image_pil.save(args.input.replace('.wav', '.png'))
    image_pil.show()
