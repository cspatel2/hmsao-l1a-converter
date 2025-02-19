# %%
# %% Level 1A converter - Level 1A (L1A) data are reconstructed, unprocessed instrument data at full resolution, time-referenced, and annotated with ancillary information, including radiometric and geometric calibration coefficients and georeferencing parameters (e.g., platform ephemeris) computed and appended but not applied to L0 data.

# %%
import argparse
from datetime import datetime, timezone, timedelta
import gc
from genericpath import isfile
import os
import sys
from time import perf_counter_ns
from matplotlib import pyplot as plt
import numpy as np
from PIL import Image
from glob import glob
import astropy.io.fits as fits
import pytz
import xarray as xr
from tqdm import tqdm
from typing import Dict, Iterable, List, SupportsFloat as Numeric
from skmpython import datetime_in_timezone

from misdesigner import MisInstrumentModel, MisCurveRemover
# %%


def find_outlier_pixels(data, tolerance=3, worry_about_edges=True):
    # This function finds the hot or dead pixels in a 2D dataset.
    # tolerance is the number of standard deviations used to cutoff the hot pixels
    # If you want to ignore the edges and greatly speed up the code, then set
    # worry_about_edges to False.
    #
    # The function returns a list of hot pixels and also an image with with hot pixels removed

    from scipy.ndimage import median_filter
    blurred = median_filter(data, size=2)
    difference = data - blurred
    threshold = tolerance*np.std(difference)

    # find the hot pixels, but ignore the edges
    hot_pixels = np.nonzero((np.abs(difference[1:-1, 1:-1]) > threshold))
    # because we ignored the first row and first column
    hot_pixels = np.array(hot_pixels) + 1

    # This is the image with the hot pixels removed
    fixed_image = np.copy(data)
    for y, x in zip(hot_pixels[0], hot_pixels[1]):
        fixed_image[y, x] = blurred[y, x]

    if worry_about_edges == True:
        height, width = np.shape(data)

        ### Now get the pixels on the edges (but not the corners)###

        # left and right sides
        for index in range(1, height-1):
            # left side:
            med = np.median(data[index-1:index+2, 0:2])
            diff = np.abs(data[index, 0] - med)
            if diff > threshold:
                hot_pixels = np.hstack((hot_pixels, [[index], [0]]))
                fixed_image[index, 0] = med

            # right side:
            med = np.median(data[index-1:index+2, -2:])
            diff = np.abs(data[index, -1] - med)
            if diff > threshold:
                hot_pixels = np.hstack((hot_pixels, [[index], [width-1]]))
                fixed_image[index, -1] = med

        # Then the top and bottom
        for index in range(1, width-1):
            # bottom:
            med = np.median(data[0:2, index-1:index+2])
            diff = np.abs(data[0, index] - med)
            if diff > threshold:
                hot_pixels = np.hstack((hot_pixels, [[0], [index]]))
                fixed_image[0, index] = med

            # top:
            med = np.median(data[-2:, index-1:index+2])
            diff = np.abs(data[-1, index] - med)
            if diff > threshold:
                hot_pixels = np.hstack((hot_pixels, [[height-1], [index]]))
                fixed_image[-1, index] = med
        ### Then the corners###

        # bottom left
        med = np.median(data[0:2, 0:2])
        diff = np.abs(data[0, 0] - med)
        if diff > threshold:
            hot_pixels = np.hstack((hot_pixels, [[0], [0]]))
            fixed_image[0, 0] = med

        # bottom right
        med = np.median(data[0:2, -2:])
        diff = np.abs(data[0, -1] - med)
        if diff > threshold:
            hot_pixels = np.hstack((hot_pixels, [[0], [width-1]]))
            fixed_image[0, -1] = med

        # top left
        med = np.median(data[-2:, 0:2])
        diff = np.abs(data[-1, 0] - med)
        if diff > threshold:
            hot_pixels = np.hstack((hot_pixels, [[height-1], [0]]))
            fixed_image[-1, 0] = med

        # top right
        med = np.median(data[-2:, -2:])
        diff = np.abs(data[-1, -1] - med)
        if diff > threshold:
            hot_pixels = np.hstack((hot_pixels, [[height-1], [width-1]]))
            fixed_image[-1, -1] = med

    return hot_pixels, fixed_image


def get_all_dirs(rootdir: str) -> list:
    fdirs = os.listdir(rootdir)
    alldirs = []
    subdirsfound = False
    for d in fdirs:
        path = os.path.join(rootdir, d)
        if os.path.isdir(path):
            subdirsfound = True
            alldirs.append(path)
    if not subdirsfound:
        alldirs.append(rootdir)
    return alldirs


def get_tstamp_from_fname(fname: str | Iterable[str], use_name: bool=True) -> Numeric | List[Numeric]:
    if isinstance(fname, str):
        try:
            if use_name:
                dirname = os.path.dirname(fname)
                dirname = os.path.basename(dirname)
                uname = os.path.basename(fname)
                while True:
                    name, ext = uname.rsplit('.', 1)
                    if ext.isdecimal():
                        break
                    else:
                        uname = name
                time = datetime.strptime(f'{dirname} {uname} +0100', '%Y%m%d %H%M%S.%f %z')
                return time.timestamp()
            else:
                raise UserWarning('Default fallback')
        except Exception as e:
            with fits.open(fname) as hdul:
                if len(hdul) > 1:
                    idx = 1
                else:
                    idx = 0

                hdu = hdul[idx]
                if 'TIMESTAMP_S' in hdu.header:
                    time_s = hdu.header['TIMESTAMP_S']
                    if 'TIMESTAMP_NS' in hdu.header:
                        time_ns = hdu.header['TIMESTAMP_NS']
                    else:
                        time_ns = 0
                    time = time_s + 1e-9*time_ns
                    return time
                else:
                    raise ValueError('Invalid file')
    else:
        return list(map(get_tstamp_from_fname, fname))


def get_tstamp_from_hdu(hdu) -> Numeric:
    if 'TIMESTAMP_S' in hdu.header:
        time_s = hdu.header['TIMESTAMP_S']
        if 'TIMESTAMP_NS' in hdu.header:
            time_ns = hdu.header['TIMESTAMP_NS']
        else:
            time_ns = 0
        time = time_s + 1e-9*time_ns
        return time
    else:
        raise ValueError('Invalid file')


def get_exposure_from_hdu(hdu) -> Numeric:
    if 'EXPOSURE_S' in hdu.header:
        exp_s = hdu.header['EXPOSURE_S']
        if 'EXPOSURE_NS' in hdu.header:
            exp_ns = hdu.header['EXPOSURE_NS']
        else:
            exp_ns = 0
        exp = exp_s + 1e-9*exp_ns
        return exp
    else:
        raise ValueError('Invalid File')


# %%
PATH = os.path.dirname(os.path.realpath(__file__))

# argument parser
parser = argparse.ArgumentParser(
    description='Convert HiT&MIS L0 data to L1A data, with exposure normalization and dark subtraction. It uses MisInstrument Model to extract ROI and perfoms line straightening. This program requires a instrument defination JSON file (See MisInstrument implementation for details.).'
)
# %%
# arguments
# parser.add_argument(
#     'NAME',
#     metavar = 'NAME',
#     required = BOOL,
#     type = TYPE,
#     default = AS REQUIRED,
#     nargs = '?',
#     help = 'DESCRIPTION OF ARGUMENT'
#     )
parser.add_argument(
    'rootdir',
    metavar='rootdir',
    # required = True,
    type=str,
    # default =
    nargs='?',
    help='Root Directory containing HiT&MIS data'
)

parser.add_argument(
    'dest',
    metavar='dest',
    # required = False,
    type=str,
    default=os.getcwd(),
    nargs='?',
    help='Root directory where L1 data will be stored.'
)

parser.add_argument(
    'dest_prefix',
    metavar='dest_prefix',
    # required = False,
    type=str,
    default=os.getcwd(),
    nargs='?',
    help='Prefix of the saved L1 data finename.'
)

parser.add_argument(
    '--overwrite',
    required=False,
    type=bool,
    default=False,
    nargs='?',
    help='If you want to rewrite an existing file, then True. Defaults to False.'
)

parser.add_argument(
    '--window',
    # metavar = 'NAME',
    required=True,
    type=str,
    # default = AS REQUIRED,
    nargs='?',
    help='Window to process.'
)

parser.add_argument(
    '--dark',
    # metavar = 'NAME',
    required=False,
    type=str,
    # default = None,
    nargs='?',
    help='Dark data (.nc) file path.'
)

parser.add_argument(
    '--model',
    # metavar = 'NAME',
    required=True,
    type=str,
    # default = AS REQUIRED,
    nargs='?',
    help='Instrument Model file path.'
)
# %%
args = parser.parse_args()

# 0. Paths
if os.path.exists(args.dest):
    if os.path.isfile(args.dest):
        print('Destination path provided is a file. Directory path required.')
        sys.exit()
else:
    os.makedirs(args.dest, exist_ok=True)

# 1. Check provided arguments and Initialize

# Create model and confirm that the Instrument file provided works
model = MisInstrumentModel.load(args.model)
predictor = MisCurveRemover(model)  # line straightening

# check destination path is a real directory
if os.path.exists(args.dest) and os.path.isfile(args.dest):
    print('Destination path provided is a file. Directory path required.')

# check if root dir exists
if not os.path.isdir(args.rootdir):
    print("Root Directory provided does not exist.")
    sys.exit()

# get all the fits files from all the subdirs, sorted by time
dirlist = get_all_dirs(args.rootdir)
print(dirlist[0])
files = None
tstamps = None
print(f'Total Number of Directories in Rootdir: {len(dirlist)}')
dirlist.sort()
for d in dirlist:
    f = glob(os.path.join(d, '*.fit*'))
    tstamp = get_tstamp_from_fname(f)
    idx = np.argsort(tstamp)
    tstamp = np.asarray(tstamp)[idx]
    f = np.asarray(f, dtype=object)[idx]
    if files is None:
        files = f
        tstamps = tstamp
    else:
        files = np.concatenate([files, f])
        tstamps = np.concatenate([tstamps, tstamp])
if len(files) < 1:
    raise ValueError('No .fit files in rootdir')
else:
    print(f'Total Number of Files to Process: {len(files)}\n')

# final list of files to process
idx = np.argsort(tstamps)
tstamps = np.sort(tstamps)
files = files[idx]

# get start and end date of the full dataset
start_date = datetime.fromtimestamp(tstamps[0], tz=pytz.utc)
end_date = datetime.fromtimestamp(tstamps[-1], tz=pytz.utc)
print(f'Start DateTime: {start_date}')
print(f'End DateTime: {end_date} \n')

# break up into individual days, day is midnight to midnight
st_date = start_date.date() - timedelta(days=1)
lst_date = end_date.date() + timedelta(days=1)
main_flist: Dict[datetime, List[str]] = {}
all_files = []
print('Dates with data: ', end='')
data_found = False
first = True
while st_date <= lst_date:
    _st_date = st_date
    start = datetime_in_timezone(
        pytz.utc, st_date.year, st_date.month, st_date.day)  # midnight
    st_date += timedelta(days=1)
    stop = start + timedelta(days=1)  # to midnight
    start_ts = start.timestamp()
    stop_ts = stop.timestamp()
    valid_files = [f if start_ts <= t <
                   stop_ts else '' for f, t in zip(files, tstamps)]
    while '' in valid_files:
        valid_files.remove('')
    if len(valid_files) > 0:
        data_found = True
        main_flist[_st_date] = valid_files
        all_files += valid_files
        if first:
            print(_st_date, end='')
            first = False
        else:
            print(',', _st_date, end='')
        sys.stdout.flush()
if not data_found:
    print('None')
print('\n')

print(f'data will be saved to: {args.dest}\n')
# get dark data
if args.dark is not None:
    dfile = args.dark
    darkds = xr.load_dataset(dfile)
else:
    darkds = None

# Check that user provided windows can be processed
window = args.window
if window not in predictor.windows:
    raise ValueError(
        f'Invalid window name: {window}. Available window names are {predictor.windows}.')
else:
    print(f'Processing ROI: {window} nm')

# each nc file should contain frames from midnight to midnight
current_start_time = datetime(
    start_date.year, start_date.month, start_date.day, 0, 0, 0)
# print(f'current start time: = {current_start_time}')

for key, filelist in main_flist.items():
    # print(f'[{key:%Y-%m-%d}] Starting conversion...')
    yymmdd = f'{key:%Y%m%d}'
    prefix = args.dest_prefix
    outfname = f"{prefix}_{yymmdd}_{window}.nc"
    outfpath = os.path.join(args.dest, outfname)
    if os.path.exists(outfpath):
        if not args.overwrite:
            print(f'{outfname} already exists. skipping.')
            continue
        else:
            os.remove(outfpath)

    # Initialize
    output = []  # straighten images
    imgsize = (len(predictor.beta_grid), len(predictor.gamma_grid))

    # fig, ax = plt.subplots(1, 1, figsize=(6.4, 4.8), dpi=150)
    # fig.subplots_adjust(right=0.85)
    # cax = fig.add_axes([0.9, 0.1, 0.03, 0.8])
    # ax.set_title(f'{key:%Y-%m-%d}')
    # plt.ion()
    # plt.show()
    # fig.canvas.draw_idle()
    # fig.canvas.flush_events()
    # Process Images
    for fidx, fn in enumerate(tqdm(filelist, desc=f'{key:%Y-%m-%d}')):
        # initialize the index of the hdul data using the first file
        # key = 'IMAGE'  # use hdul.info() to see all keys in file
        with fits.open(fn) as hdul:
            hdu = hdul['IMAGE']
            header = hdu.header
            tstamp = get_tstamp_from_hdu(hdu)  # s
            ststamp = datetime.fromtimestamp(tstamp, tz=pytz.utc)
            exposure = get_exposure_from_hdu(hdu)  # s
            temp = header['CCD-TEMP']  # C
            # 1. get img
            data = np.asarray(hdu.data, dtype=float)  # counts
            # plt.figure()
            # plt.imshow(data, vmin = np.nanpercentile(data,0.1), vmax = np.nanpercentile(data,99) )
            # plt.colorbar()
            # plt.savefig('img_step1.png')
            # 1a. hot pixel correction for long exposures
            _, data = find_outlier_pixels(data)
            # 2. dark/bias correction
            if darkds is not None:
                dark = np.asarray(darkds['darkrate'].values, dtype=float)
                bias = np.asarray(darkds['bias'].values, dtype=float)
                data -= bias + dark * exposure  # counts
            # plt.figure()
            # plt.imshow(data, vmin = np.nanpercentile(data,0.1), vmax = np.nanpercentile(data,99) )
            # plt.colorbar()
            # plt.savefig('img_step2.png')
            # 3. total counts -> counts.sec
            data = data/exposure  # counts/sec
            # plt.figure()
            # plt.imshow(data, vmin = np.nanpercentile(data,0.1), vmax = np.nanpercentile(data,99) )
            # plt.colorbar()
            # plt.savefig('img_step3.png')
            # 4. Crop and resize image
            data = Image.fromarray(data)
            data = data.rotate(-.311,
                               resample=Image.Resampling.BILINEAR, fillcolor=np.nan)
            data = data.transpose(Image.Transpose.FLIP_LEFT_RIGHT)
            image = Image.new('F', imgsize, color=np.nan)
            image.paste(data, (110, 410))
            data = np.asarray(image).copy()
            # plt.figure()
            # plt.imshow(data)
            # plt.savefig('img_step4.png')
            # array -> DataArray
            data = xr.DataArray(
                data,
                dims=['gamma', 'beta'],
                coords={
                    'gamma': predictor.gamma_grid,
                    'beta': predictor.beta_grid
                },
                attrs={'unit': 'ADU/s'}
            )
            # 5. straighten img
            data = predictor.straighten_image(data, window)
            # 6. Plot
            # tstamp_str = ststamp.strftime("%Y-%m-%d %H:%M:%S")
            # fig.suptitle(f'{tstamp_str} UTC')
            # im = data.plot(ax=ax, cmap='viridis', add_colorbar=False)
            # fig.colorbar(im, cax=cax)
            # fig.savefig('plot.png')
            # fig.canvas.draw_idle()
            # fig.canvas.flush_events()
            # 7. Save
            data = data.expand_dims(
                dim={'tstamp': (tstamp,)}).to_dataset(name='intensity', promote_attrs=True)
            data['exposure'] = xr.Variable(
                dims='tstamp', data=[exposure], attrs={'unit': 's'}
            )
            data['ccdtemp'] = xr.Variable(
                dims='tstamp', data=[temp], attrs={'unit': 'C'}
            )
            output.append(data)
            # ax.clear()
    # plt.close(fig)

    # Create Dataset and save
    isornah = 'is'
    if darkds is None: isornah += ' not'

    tnow = datetime.now(timezone.utc).strftime('%d/%m/%Y, %H:%M:%S UTC')
    ds: xr.Dataset = xr.concat(output, dim='tstamp')
    del output
    gc.collect()
    ds.attrs.update(
        dict(Description=" HMSA-O Straighted Spectra",
             ROI=f'{str(window)} nm',
             DataProcessingLevel='1A',
             FileCreationDate=tnow,
             ObservationLocation='Swedish Institute of Space Physics/IRF (Kiruna, Sweden)',
             Note = f'data {isornah} dark corrected.',
             )
    )

    ds['intensity'].attrs['unit'] = 'ADU/s'
    ds['tstamp'].attrs['unit'] = 's'
    ds['tstamp'].attrs['description'] = 'Seconds since UNIX epoch 1970-01-01 00:00:00 UTC'
    encoding = {var: {'zlib': True}
                for var in (*ds.data_vars.keys(), *ds.coords.keys())}

    print('Saving %s...\t' % (outfname), end='')
    sys.stdout.flush()
    tstart = perf_counter_ns()
    ds.to_netcdf(outfpath, encoding=encoding)
    tend = perf_counter_ns()
    print(f'Done. [{(tend-tstart)*1e-9:.3f} s]')


# %%

# %%
# import xarray as xr
# import matplotlib.pyplot as plt

# ds = xr.open_dataset('/home/charmi/Projects/hms-ao/l1a_converter/l1a_data/trial_20250117_5577.nc')
# ds.intensity.isel(tstamp = 1).plot()
# plt.axvline(557.7)
# # %%
# darkds = xr.open_dataset('zwoasi533_darkbias_T-15_G100.nc')
# # %%
# darkds
# # %%
# data = darkds.darkrate.values*120
# plt.imshow(data)
# plt.colorbar()
# %%
# np.nanmax(data)
# %%
