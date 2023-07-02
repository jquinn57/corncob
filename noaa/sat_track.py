
#curl -o weather.txt https://celestrak.org/NORAD/elements/weather.txt
from skyfield.api import load, wgs84, utc
import os
import datetime
import numpy as np
from IPython import embed
import matplotlib.pyplot as plt
from PIL import Image

import cartopy.crs as ccrs
import cartopy.feature as cfeature
import wave

# sudo apt -y install libgeos-dev
# pip3 install cartopy
# pip3 install skyfield



def plot_map():
    fig = plt.figure(figsize=(10, 5))
    ax = fig.add_subplot(1, 1, 1, projection=ccrs.PlateCarree())

    # make the map global rather than have it zoom in to
    # the extents of any plotted data
    ax.set_global()

    ax.add_feature(cfeature.LAND)
    ax.add_feature(cfeature.OCEAN)
    ax.add_feature(cfeature.COASTLINE)
    ax.add_feature(cfeature.BORDERS, linestyle=':')
    ax.add_feature(cfeature.STATES, linestyle=':')

    return ax


def plot_map_and_track(lat, long, heights, img=None):

    npts = len(lat)
    long_mid = long[npts // 2]
    lat_mid = lat[npts // 2]

    # long extent at the midpoint given a distance extent of +/- 2080 km
    dist_extent_km = 909 * 2.8
    earth_radius_km = 6371
    long_extent_deg = 360 * dist_extent_km /( 2 * np.pi * earth_radius_km * np.cos(np.pi * lat_mid / 180))

    # height in meters above the surface of the earth
    heights_mid_m = heights[npts // 2] * 1000

    # estimate rotation angle based on mid point
    delta_lat = lat[npts //2 + 1] -lat[npts //2 - 1]
    delta_long = long[npts //2 + 1] -long[npts //2 - 1]
    alpha = (180 / np.pi) * np.arctan(np.cos(np.pi * lat_mid / 180) * delta_long / delta_lat)
    print(alpha)

    # Create the projection
    #projection = ccrs.NearsidePerspective(central_longitude=long_mid, central_latitude=lat_mid, satellite_height=heights_mid_m)
    projection = ccrs.Orthographic(central_longitude=long_mid, central_latitude=lat_mid)
    #projection = ccrs.Gnomonic(central_longitude=long_mid, central_latitude=lat_mid)

    #projection = ccrs.PlateCarree()

    fig = plt.figure(figsize=(10, 10))
    ax = fig.add_subplot(1, 1, 1, projection=projection)
    ax.set_extent([long_mid - long_extent_deg / 2,  long_mid + long_extent_deg / 2, lat.min(), lat.max()])

    if img is not None:
        img = img.rotate(-alpha)
        plt.imshow(img, extent=ax.get_extent())
    # Add geographic features
    #ax.add_feature(cfeature.LAND, edgecolor='black')
    #ax.add_feature(cfeature.OCEAN)
    ax.add_feature(cfeature.COASTLINE, color='yellow')
    ax.add_feature(cfeature.BORDERS, linestyle=':')
    ax.add_feature(cfeature.STATES, linestyle=':')
    #ax.gridlines(color='red')

    # Set the extent of the map
    ax.plot(long, lat, 'g', transform=ccrs.PlateCarree())



    return ax



stations_url = 'weather.txt'
satellites = load.tle_file(stations_url)
print('Loaded', len(satellites), 'satellites')
by_name = {sat.name: sat for sat in satellites}
satellite = by_name['NOAA 19']

print(satellite.epoch.utc_jpl())

wav_filename = 'gqrx_20230624_170718_137100000.wav'
#wav_filename = 'gqrx_20230626_164306_137103000.wav'

# last timestamp
mod_timestamp = os.path.getmtime(wav_filename)

#first timstep
wave_reader = wave.open(wav_filename)
sample_rate = wave_reader.getframerate()
duration_sec = wave_reader.getnframes() / sample_rate
wave_reader.close()

times = []
for i in range(int(duration_sec)):
    ti = datetime.datetime.fromtimestamp( mod_timestamp - i)
    times.append(ti.astimezone(utc))

# You can instead use ts.now() for the current time
ts = load.timescale()
t = ts.from_datetimes(times)

geocentric = satellite.at(t)
print(geocentric.position.km)

lat, lon = wgs84.latlon_of(geocentric)
heights = wgs84.height_of(geocentric)
print('Latitude:', lat.degrees)
print('Longitude:', lon.degrees)
print('heights:', heights.km)

img = Image.open(wav_filename.replace('.wav', '.png'))

ax = plot_map_and_track(lat.degrees, lon.degrees, heights.km, img=img)
plt.show()
