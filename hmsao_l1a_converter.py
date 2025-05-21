# %%
# %% Level 1A converter - Level 1A (L1A) data are reconstructed, unprocessed instrument data at full resolution, time-referenced, and annotated with ancillary information, including radiometric and geometric calibration coefficients and georeferencing parameters (e.g., platform ephemeris) computed and appended but not applied to L0 data.

# %%
import argparse
from datetime import datetime, timezone, timedelta
import gc
from genericpath import isfile
from math import ceil
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
from skimage import transform


from misdesigner import MisInstrumentModel, MisCurveRemover
# %% functions


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


def get_tstamp_from_fname(fname: str | Iterable[str], use_name: bool = True) -> Numeric | List[Numeric]:
    if isinstance(fname, str):
        try:
            if use_name:
                uname = os.path.basename(fname)
                while True:
                    name, ext = uname.rsplit('.', 1)
                    if ext.isdecimal():
                        break
                    else:
                        uname = name
                time = datetime.strptime(
                    f'{uname} +0000', '%Y%m%d%H%M%S.%f %z')
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


def zenith_angle(gamma_mm: Numeric | Iterable[Numeric], f1: Numeric = 30, f2: Numeric = 30, D: Numeric = 24, yoffset: Numeric = 12.7) -> Numeric:
    """Calculates the zenith angle in degrees from the gamma(mm) in slit coordinates.

    Args:
        gamma_mm (Numeric | Iterable[Numeric]): gamma (mm) in slit (instrument coordinate system) coordinates.
        f1 (Numeric, optional): focal length (mm) of the 1st lens in the telecentric foreoptic. Defaults to 30 mm.
        f2 (Numeric, optional): focal length (mm) of the 2nd lens in the telecentric foreoptic. Defaults to 30 mm.
        D (Numeric, optional): Distance (mm) between the two lens. Defaults to 24 mm.
        yoffset (Numeric, optional): the distance between the optic axis of the telescope to the x-axis of the instrument coordinate system. Defaults to 12.7 mm.

    Returns:
        Numeric: the zenith angle in degrees.
                Note: result is non linear b/c of arctan()

    """
    if isinstance(gamma_mm, (int, float)):
        return [zenith_angle(x) for x in gamma_mm]
    if np.min(gamma_mm) < 0:
        sign = -1
    else:
        sign = 1
    num = -(gamma_mm-(sign*yoffset))*(f1+f2-D)
    den = f1*f2
    return np.rad2deg(np.arctan(num/den))


def convert_gamma_to_zenithangle(ds: xr.Dataset, plot: bool = False, returnboth: bool = False):
    """converts gamma(mm) in slit coordinate to zenith angle (degrees) in a straightened dataset.

    Args:
        ds (xr.Dataset): straightened dataset.
        plot (bool, optional): if True, left plot is raw zenith angle and right plot is linearized zenith angle. Defaults to False.
        returnboth (bool, optional): if True, returns both datasets i.e. with raw (non linear) zenith angle and second with linear zenith angles. If false, only returns dataset with linear zenith angles. Defaults to False.

    Returns:
        _type_: dataset with gamma(mm) replaced with zenith angle (deg)
                Note: calculated zenith angles are non-linear b/c of arctann(). This is corrected using ndimage.transform.warp() to a linearized zenith angles.
    """
    # initilize the new dataset with linear za
    nds = ds.copy()

    # gamma -> zenith angle
    angles = zenith_angle(ds.gamma.values)

    # coordinate map in the input image
    mxi, myi = np.meshgrid(ds.wavelength.values, angles)
    imin, imax = np.nanmin(myi), np.nanmax(myi)
    myi -= imin  # shift to 0
    myi /= (imax - imin)  # normalize to 1
    myi *= (len(angles))  # adjust

    # coordinate map in the output image
    if np.nanmin(angles) < 0:
        sign = 1
    else:
        sign = -1
    linangles = np.linspace(np.min(angles), np.max(angles), len(
        angles), endpoint=True)[::sign]  # array of linear zenith angles
    mxo, myo = np.meshgrid(ds.wavelength.values, linangles)
    omin, omax = np.nanmin(mxo), np.nanmax(mxo)
    mxo -= omin  # shift to 0
    mxo /= (omax - omin)  # normalize to 1
    mxo *= (len(ds.wavelength.values))  # adjust

    # inverse map
    imap = np.zeros((2, *(ds.shape)), dtype=float)
    imap[0, :, :] = myi  # input image map
    imap[1, :, :] = mxo  # output image map

    # nonlinear za -> linear za
    timg = transform.warp(ds.values, imap, order=1, cval=np.nan)

    # replace gamma to raw za values
    ds['gamma'] = angles
    ds['gamma'] = ds['gamma'].assign_attrs(
        {'unit': 'deg', 'long_name': 'Zenith Angle'})
    ds = ds.rename({'gamma': 'za'})
    # replace gamma to linear za values
    nds.values = timg
    nds['gamma'] = linangles
    nds['gamma'] = nds['gamma'].assign_attrs(
        {'unit': 'deg', 'long_name': 'Zenith Angle'})
    nds = nds.rename({'gamma': 'za'})
    if plot:
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 6), dpi=300)
        fig.tight_layout()

        vmin = np.nanpercentile(ds.values, 1)
        vmax = np.nanpercentile(ds.values, 99)
        ds.plot(ax=ax1, vmin=vmin, vmax=vmax)
        ax1.set_title('Zenith Angle (NL)')

        vmin = np.nanpercentile(timg, 1)
        vmax = np.nanpercentile(timg, 99)
        nds.plot(ax=ax2, vmin=vmin, vmax=vmax)
        ax2.set_title('Zenith Angle (Warped Linear)')

    if returnboth:
        return nds, ds
    else:
        return nds


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
    default=None,
    nargs='?',
    help='Prefix of the saved L1 data finename.'
)


def str2bool(value: str) -> bool:
    if value.lower() in ('true', '1', 't', 'y', 'yes'):
        return True
    elif value.lower() in ('false', '0', 'f', 'n', 'no'):
        return False
    raise ValueError("Invalid boolean value: {}".format(value))


parser.add_argument(
    '--overwrite',
    required=False,
    type=str2bool,
    default=False,
    nargs='?',
    help='If you want to rewrite an existing file, then True. Defaults to False.'
)


def list_of_strings(arg: str) -> List[str]:
    return arg.split(',')


parser.add_argument(
    '--windows',
    # metavar = 'NAME',
    # action='append',
    required=False,
    type=list_of_strings,
    default=None,
    nargs='?',
    help='Window(s) to process (list of str i.e. "1235", "3456").'
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

parser.add_argument(
    '--chunksize',
    # metavar = 'NAME',
    required=False,
    type=int,
    default=10,  # fix this later depending on what the ideal number of files should be per chunk
    nargs='?',
    help='Number of files per chunk.'
)

parser.add_argument(
    '--readnoise',
    # metavar = 'NAME',
    required=False,
    type=float,
    default=None,
    nargs='?',
    help='Readnoise value (ADU) to be used for readnoise correction.'
)
# %%


def main(parser: argparse.ArgumentParser):
    args = parser.parse_args()

    # 0. Paths
    if os.path.exists(args.dest):
        if os.path.isfile(args.dest):
            print('Destination path provided is a file. Directory path required.')
            sys.exit()
    else:
        os.makedirs(args.dest, exist_ok=True)
    # print(f'destination dir set to: {args.dest}')

    # 1. Check provided arguments and Initialize

    # Create model and confirm that the Instrument file provided works
    model = MisInstrumentModel.load(args.model)
    predictor = MisCurveRemover(model)  # line straightening

    if args.dest_prefix is None:
        args.dest_prefix = model.get_instrument().system.replace(' ', '').lower()

    # Check that user provided windows can be processed
    if args.windows is not None and isinstance(args.windows, list):
        windows = [
            window for window in args.windows if window in predictor.windows]
        if len(windows) == 0:
            raise ValueError(
                f'Invalid Window names: {args.windows}. Available window names are {predictor.windows}')
    else:
        windows = predictor.windows

    print(f'Windows to be processed: {', '.join(windows)}')

    # check if root dir exists
    if not os.path.isdir(args.rootdir):
        print("Root Directory provided does not exist.")
        sys.exit()

    # get all the fits files from all the subdirs, sorted by time
    dirlist = get_all_dirs(args.rootdir)
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

    # readnoise option
    readnoise = None

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

    del idx, tstamps, files
    gc.collect()

    # get dark data
    is_dark_subtracted = 'is'
    if args.dark is not None:
        dfile = args.dark
        darkds = xr.load_dataset(dfile)
    else:
        darkds = None
        is_dark_subtracted += ' not'

    for key, filelist in main_flist.items():
        # print(f'[{key:%Y-%m-%d}] Starting conversion...')
        yymm = f'{key:%Y%m}'
        yymmdd = f'{key:%Y%m%d}'
        prefix = args.dest_prefix
        # each month is a new directory at dest
        os.makedirs(os.path.join(args.dest, yymm), exist_ok=True)
        # file names/paths of the complete file of a given day and window
        outfnames = [
            f"{yymm}/{prefix}_{yymmdd}_{window}*.nc" for window in windows]
        outfpaths = [os.path.join(args.dest, outfname)
                     for outfname in outfnames]

        # check if any of the output files already exist
        skip_processing = False
        for pathidx, outfpath in enumerate(outfpaths):
            numfiles = glob(outfpath)
            print(f'{outfpath} has {len(numfiles)} related files')
            if len(numfiles) > 0:
                print(f'overwrite = {args.overwrite}')
                if args.overwrite:
                    for i in glob(outfpath):
                        print(f'{i} removed.')
                        os.remove(i)
                elif not args.overwrite:  # if overwrite is false and file exist
                    for i in glob(outfpath):
                        print(f'{i} already exists, skipping')
                    skip_processing = True

        if not skip_processing:
            absstart = perf_counter_ns()
            # split 1 day into len(filesperday)/n loops
            n = args.chunksize
            chunks = ceil(len(filelist) / n)
            ndigits = ceil(np.log10(chunks))
            iterlim = chunks * n
            # subfilelists = [filelist[i:i+n] for i in np.arange(0, len(filelist), n)]
            # print(f'Number of chunks of day: {chunks}')
            imgsize = (len(predictor.beta_grid), len(predictor.gamma_grid))
            # for subidx, sublist in enumerate(subfilelists):
            for subidx in range(chunks):
                out_countrate = {k: [] for k in windows}
                out_noise = {k: [] for k in windows}
                sublist = filelist[subidx*n:(subidx + 1)*n]
                for _, fn in enumerate(tqdm(sublist, desc=f'{key:%Y-%m-%d} - [{subidx+1:0{ndigits}}/{chunks}]')):
                    # initialize the index of the hdul data using the first file
                    # key = 'IMAGE'  # use hdul.info() to see all keys in file
                    with fits.open(fn) as hdul:
                        hdu = hdul['IMAGE']
                        header = hdu.header
                        tstamp = get_tstamp_from_hdu(hdu)  # s
                        # ststamp = datetime.fromtimestamp(tstamp, tz=pytz.utc)
                        exposure = get_exposure_from_hdu(hdu)  # s
                        temp = header['CCD-TEMP']  # C
                        # 1. get img
                        data = np.asarray(hdu.data, dtype=float)  # counts
                        # 1a. hot pixel correction for long exposures
                        _, data = find_outlier_pixels(data)
                        # 2. dark/bias correction
                        if darkds is not None:
                            dark = np.asarray(
                                darkds['darkrate'].values, dtype=float)
                            bias = np.asarray(
                                darkds['bias'].values, dtype=float)
                            data -= bias + dark * exposure  # counts
                        
                        # 3. total counts -> counts.sec
                        data = data/exposure  # counts/sec
                        # 4. Crop and resize image
                        data = Image.fromarray(data)
                        data = data.rotate(-.311,resample=Image.Resampling.BILINEAR, fillcolor=np.nan)
                        data = data.transpose(Image.Transpose.FLIP_LEFT_RIGHT)
                        image = Image.new('F', imgsize, color=np.nan)
                        image.paste(data, (110, 410))
                        data = np.asarray(image).copy() 
                        del image
                        # array -> DataArray
                        data_ = xr.DataArray(
                            data,
                            dims=['gamma', 'beta'],
                            coords={
                                'gamma': predictor.gamma_grid,
                                'beta': predictor.beta_grid
                            },
                            attrs={'unit': 'ADU/s'}
                        )
                        # 5. straighten img
                        for window in windows:
                            data = predictor.straighten_image(data_, window, coord='Slit')
                            data = convert_gamma_to_zenithangle(data)
                            # 6. Save
                            data = data.expand_dims(dim={'tstamp': (tstamp,)}).to_dataset(name='intensity', promote_attrs=True)
                            data['exposure'] = xr.Variable(dims='tstamp', data=[exposure], attrs={'unit': 's'})
                            data['ccdtemp'] = xr.Variable(dims='tstamp', data=[temp], attrs={'unit': 'C'})
                            out_countrate[window].append(data)
                        
                        if readnoise is None and args.readnoise is not None:
                            readnoise = np.full(
                                data_.data.shape, args.readnoise, dtype=float)
                            readnoise = Image.fromarray(readnoise)
                            readnoise = readnoise.rotate(-.311,
                                                         resample=Image.Resampling.BILINEAR, fillcolor=np.nan)
                            readnoise = readnoise.transpose(
                                Image.Transpose.FLIP_LEFT_RIGHT)
                            image = Image.new('F', imgsize, color=np.nan)
                            image.paste(readnoise, (110, 410))
                            readnoise = np.asarray(image).copy()
                            noise = np.sqrt(readnoise**2 + data_.data*exposure)/exposure
                            del image
                            noise = xr.DataArray(
                                noise,
                                dims=['gamma', 'beta'],
                                coords={
                                    'gamma': predictor.gamma_grid,
                                    'beta': predictor.beta_grid
                                },
                                attrs={'unit': 'ADU'}
                            )

                            for window in windows:
                                rn = predictor.straighten_image(noise, window, coord='Slit')
                                rn = convert_gamma_to_zenithangle(rn)
                                rn = rn.expand_dims(dim={'tstamp': (tstamp,)}).to_dataset(name='noise', promote_attrs=True)
                                rn['exposure'] = xr.Variable(dims='tstamp', data = [exposure], attrs={'unit': 's'})
                                rn['ccdtemp'] = xr.Variable(dims='tstamp', data = [temp], attrs={'unit': 'C'})

                                out_noise[window].append(rn)

                # Create Dataset and save
                for window in windows:
                    sub_outfname = f"{yymm}/{prefix}_{yymmdd}_{window}[{subidx:0{ndigits}}].nc"
                    sub_outfpath = os.path.join(args.dest, sub_outfname)
                    ds: xr.Dataset = xr.concat(out_countrate[window], dim='tstamp')
                    gc.collect()
                    ds.attrs.update(
                        dict(Description=" HMSA-O Straighted Spectra",
                             ROI=f'{str(window)} nm',
                             DataProcessingLevel='1A',
                             # FileCreationDate=tnow,
                             ObservationLocation='Swedish Institute of Space Physics/IRF (Kiruna, Sweden)',
                             Note=f'data {is_dark_subtracted} dark corrected.',
                             )
                    )
                    if args.readnoise is not None:
                        ds = ds.merge(xr.concat(out_noise[window], dim='tstamp'))
                    ds['intensity'].attrs['unit'] = 'ADU/s'
                    ds['intensity'].attrs['long_name'] = 'Line Intensity'
                    ds['noise'].attrs['unit'] = 'ADU/s'
                    ds['noise'].attrs['long_name'] = 'Noise'
                    ds['noise'].attrs['eqn'] = r'Noise is given by sqrt{RN^2 + Counts}/exp'
                    ds['tstamp'].attrs['unit'] = 's'
                    ds['tstamp'].attrs['description'] = 'Seconds since UNIX epoch 1970-01-01 00:00:00 UTC'
                    encoding = {var: {'zlib': True}
                                for var in (*ds.data_vars.keys(), *ds.coords.keys())}

                    print('Saving %s...\t' % (sub_outfname), end='')
                    sys.stdout.flush()
                    tstart = perf_counter_ns()
                    ds.to_netcdf(sub_outfpath, encoding=encoding)
                    tend = perf_counter_ns()
                    print(f'Done. [{(tend-tstart)*1e-9:.3f} s]')
                del out_countrate
                gc.collect()
            absend = perf_counter_ns()

            print(f'\nDone: {key:%Y-%m-%d}, {(absend - absstart)*1e-9:.3f} s')
        else:
            continue


# %%
if __name__ == '__main__':
    main(parser)
# %%
