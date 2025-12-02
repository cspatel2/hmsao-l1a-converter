# %%
# %% Level 1A converter - Level 1A (L1A) data are reconstructed, unprocessed instrument data at full resolution, time-referenced, and annotated with ancillary information, including radiometric and geometric calibration coefficients and georeferencing parameters (e.g., platform ephemeris) computed and appended but not applied to L0 data.

# %%
import argparse
from datetime import datetime, timezone, timedelta
import gc
from genericpath import isfile
from math import ceil
import os
import re
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
from dataclasses import dataclass

# %%
LOCALPATH = os.path.dirname(os.path.realpath(__file__))
sys.path.append(os.path.join(LOCALPATH))
from l1_helpers import *

# %%


@dataclass
class L1AConfig:
    rootdir: str
    dest: str
    dest_prefix: str
    overwrite: bool
    windows: List[str]
    dates: List[str]
    dark: str
    slitsizeum: str
    chunksize: int
    readnoise: float
    model: str = os.path.join(LOCALPATH, 'hmsa_origin_ship.json')


def list_of_strings(arg: str) -> List[str]:
    return arg.split(',')


def str2bool(value: str) -> bool:
    if value.lower() in ('true', '1', 't', 'y', 'yes'):
        return True
    elif value.lower() in ('false', '0', 'f', 'n', 'no'):
        return False
    raise ValueError("Invalid boolean value: {}".format(value))


# %%


def main(config: L1AConfig):
    slitsize = config.slitsizeum

    # 0. Paths
    if os.path.exists(config.dest):
        if os.path.isfile(config.dest):
            print('Destination path provided is a file. Directory path required.')
            sys.exit()
    else:
        os.makedirs(config.dest, exist_ok=True)
    # print(f'destination dir set to: {config.dest}')

    # 1. Check provided arguments and Initialize

    # Create model and confirm that the Instrument file provided works
    model = MisInstrumentModel.load(config.model)
    predictor = MisCurveRemover(model)  # line straightening

    if config.dest_prefix is None:
        config.dest_prefix = model.get_instrument().system.replace(' ', '').lower()
    else:
        if 'l1a' not in config.dest_prefix:
            config.dest_prefix += '_l1a'

    # Check that user provided windows can be processed
    if config.windows is not None and isinstance(config.windows, list):
        windows = [
            window for window in config.windows if window in predictor.windows]
        if len(windows) == 0:
            raise ValueError(
                f'Invalid Window names: {config.windows}. Available window names are {predictor.windows}')
    else:
        windows = predictor.windows

    print(f'Windows to be processed: {', '.join(windows)}')

    # check if root dir exists
    if not os.path.isdir(config.rootdir):
        print("Root Directory provided does not exist.")
        sys.exit()

    # get all the fits files from all the subdirs, sorted by time
    dirlist = get_all_dirs(config.rootdir)
    files = np.array([], dtype=object)
    tstamps = np.array([])
    print(f'Total Number of Directories in Rootdir: {len(dirlist)}')
    dirlist.sort()
    for d in dirlist:
        f = glob(os.path.join(d, '*.fit*'))
        tstamp = get_tstamp_from_fname(f)
        idx = np.argsort(tstamp)  # type: ignore
        tstamp = np.asarray(tstamp)[idx]
        f = np.asarray(f, dtype=object)[idx]
        if files is None:
            files = f
            tstamps = tstamp
        else:
            files = np.concatenate([files, f])
            tstamps = np.concatenate([tstamps, tstamp])  # type: ignore
    if len(files) < 1:  # type: ignore
        raise ValueError('No .fit(s) files in rootdir')
    else:
        # type: ignore
        print(f'Total Number of Files to Process: {len(files)}\n')

    # final list of files to process
    idx = np.argsort(tstamps)
    tstamps = np.sort(tstamps)
    files = files[idx]

    if config.dates is not None:  # if processing dates are given, then use those
        dates = np.sort(config.dates)
        start_date = datetime.strptime(
            dates[0], '%Y%m%d').replace(tzinfo=pytz.UTC)
        end_date = datetime.strptime(
            dates[-1], '%Y%m%d').replace(tzinfo=pytz.UTC)
    else:  # get start and end date of the full dataset
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
        valid_files = [
            f if start_ts <= t < stop_ts else '' for f, t in zip(files, tstamps)
        ]
        while '' in valid_files:
            valid_files.remove('')
        if len(valid_files) > 0:
            data_found = True
            main_flist[_st_date] = valid_files  # type: ignore
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

    print(f'data will be saved to: {config.dest}\n')

    del idx, tstamps, files
    gc.collect()

    # get dark data
    is_dark_subtracted = 'is'
    if config.dark is not None:
        dfile = config.dark
        darkds = xr.load_dataset(dfile)
    else:
        darkds = None
        is_dark_subtracted += ' not'

    for key, filelist in main_flist.items():
        # print(f'[{key:%Y-%m-%d}] Starting conversion...')
        yymm = f'{key:%Y%m}'
        yymmdd = f'{key:%Y%m%d}'
        prefix = config.dest_prefix
        # each month is a new directory at dest
        os.makedirs(os.path.join(config.dest, yymm), exist_ok=True)
        # file names/paths of the complete file of a given day and window
        outfnames = [
            f"{yymm}/{prefix}_{yymmdd}_{window}*.nc" for window in windows]
        outfpaths = [os.path.join(config.dest, outfname)
                     for outfname in outfnames]

        # check if any of the output files already exist
        skip_processing = False
        for pathidx, outfpath in enumerate(outfpaths):
            numfiles = glob(outfpath)
            # print(f'{outfpath} has {len(numfiles)} related files')/
            if len(numfiles) > 0:
                print(f'overwrite = {config.overwrite}')
                if config.overwrite:
                    for i in glob(outfpath):
                        print(f'{i} removed.')
                        os.remove(i)
                elif not config.overwrite:  # if overwrite is false and file exist
                    for i in glob(outfpath):
                        print(f'{i} already exists, skipping')
                    skip_processing = True

        if not skip_processing:
            absstart = perf_counter_ns()
            # split 1 day into len(filesperday)/n loops
            n = config.chunksize
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
                        header = hdu.header  # type: ignore
                        tstamp = get_tstamp_from_hdu(hdu)  # s
                        # ststamp = datetime.fromtimestamp(tstamp, tz=pytz.utc)
                        exposure = get_exposure_from_hdu(hdu)  # s
                        temp = header['CCD-TEMP']  # C
                        # 1. get img, and read noise
                        # counts
                        data = np.asarray(
                            hdu.data,  # type: ignore
                            dtype=float
                        )
                        readnoise = np.full(
                            data.shape, config.readnoise, dtype=float
                        )  # counts

                        # 1a. photometric calibration

                        # 1b. hot pixel correction for long exposures
                        _, data = find_outlier_pixels(data)

                        # 2. dark/bias correction
                        if darkds is not None:
                            dark = np.asarray(
                                darkds['countrate'].values, dtype=float)
                            dark_noise = np.asarray(
                                darkds['countrate_err'].values, dtype=float)
                            bias = np.asarray(
                                darkds['bias'].values, dtype=float)
                            bias_noise = np.asarray(
                                darkds['bias_err'].values, dtype=float)
                            data -= bias + dark * exposure  # counts
                            readnoise = np.sqrt(
                                readnoise**2 +
                                (dark_noise * exposure)**2 + bias_noise**2
                            )

                        # 3. total counts -> counts.sec
                        data = data/exposure  # counts/sec
                        # 4. Crop and resize image
                        data = Image.fromarray(data)
                        data = data.rotate(
                            -0.311,
                            resample=Image.Resampling.BILINEAR,
                            fillcolor=np.nan,
                        )
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
                            data = predictor.straighten_image(
                                data_, window, coord='Slit'
                            )
                            data = convert_gamma_to_zenithangle(
                                data)  # type: ignore
                            # 6. Save
                            data = data.expand_dims(  # type: ignore
                                dim={'tstamp': (tstamp,)}
                            ).to_dataset(
                                name='countrate', promote_attrs=True
                            )  # type: ignore
                            data['exposure'] = xr.Variable(
                                dims='tstamp', data=[exposure], attrs={'unit': 's'})
                            data['ccdtemp'] = xr.Variable(
                                dims='tstamp', data=[temp], attrs={'unit': 'C'})
                            out_countrate[window].append(data)
                        # 7. readnoise propogation
                        readnoise = Image.fromarray(readnoise)
                        readnoise = readnoise.rotate(
                            -.311, resample=Image.Resampling.BILINEAR, fillcolor=np.nan)
                        readnoise = readnoise.transpose(
                            Image.Transpose.FLIP_LEFT_RIGHT)
                        image = Image.new('F', imgsize, color=np.nan)
                        image.paste(readnoise, (110, 410))
                        readnoise = np.asarray(image).copy()
                        readnoise = np.sqrt(
                            readnoise**2 + data_.data*exposure)/exposure
                        del image
                        readnoise = xr.DataArray(
                            readnoise,
                            dims=['gamma', 'beta'],
                            coords={
                                'gamma': predictor.gamma_grid,
                                'beta': predictor.beta_grid
                            },
                            attrs={'unit': 'ADU/s'}
                        )

                        for window in windows:
                            rn = predictor.straighten_image(
                                readnoise, window, coord='Slit')
                            rn = convert_gamma_to_zenithangle(
                                rn)  # type: ignore
                            rn = rn.expand_dims(  # type: ignore
                                dim={'tstamp': (tstamp,)}
                            ).to_dataset(
                                name='noise', promote_attrs=True
                            )
                            # rn['exposure'] = xr.Variable(dims='tstamp', data = [exposure], attrs={'unit': 's'})
                            # rn['ccdtemp'] = xr.Variable(dims='tstamp', data = [temp], attrs={'unit': 'C'})
                            out_noise[window].append(rn)

                # print(len(out_noise[window]))
                # Create Dataset and save
                for window in windows:
                    sub_outfname = f"{yymm}/{prefix}_{yymmdd}_{window}[{subidx:0{ndigits}}].nc"
                    sub_outfpath = os.path.join(config.dest, sub_outfname)
                    ds: xr.Dataset = xr.concat(
                        out_countrate[window], dim='tstamp')
                    gc.collect()
                    ds.attrs.update(
                        dict(
                            Description=" HMSA-O Straighted Spectra",
                            ROI=f'{int(window)/10:0.1f} nm',
                            slit_size_um=str(slitsize),
                            DataProcessingLevel='1A',
                            FileCreationDate=datetime.now().strftime("%m/%d/%Y, %H:%M:%S EDT"),
                            ObservationLocation='Swedish Institute of Space Physics/IRF (Kiruna, Sweden)',
                            Note=f'data {is_dark_subtracted} dark corrected.',
                        )
                    )
                    if config.readnoise is not None:
                        ds = xr.merge(
                            [ds, xr.concat(out_noise[window], dim='tstamp')])
                    ds['countrate'].attrs['unit'] = 'ADU/s'
                    ds['countrate'].attrs['long_name'] = 'Line Intensity'
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
    # argument parser
    parser = argparse.ArgumentParser(
        description='Convert HiT&MIS L0 data to L1A data, with exposure normalization and dark subtraction. It uses MisInstrument Model to extract ROI and perfoms line straightening. This program requires a instrument defination JSON file (See MisInstrument implementation for details.).'
    )

    parser.add_argument(
        '--rootdir',
        metavar='rootdir',
        # required = True,
        type=str,
        # default =
        nargs='?',
        help='Root Directory containing HiT&MIS data'
    )

    parser.add_argument(
        '--dest',
        metavar='dest',
        # required = False,
        type=str,
        default=os.getcwd(),
        nargs='?',
        help='Root directory where L1 data will be stored.'
    )

    parser.add_argument(
        '--dest_prefix',
        metavar='dest_prefix',
        # required = False,
        type=str,
        default=None,
        nargs='?',
        help='Prefix of the saved L1 data finename.'
    )

    parser.add_argument(
        '--overwrite',
        required=False,
        type=str2bool,
        default=False,
        nargs='?',
        help='If you want to rewrite an existing file, then True. Defaults to False.'
    )

    parser.add_argument(
        '--windows',
        # metavar = 'NAME',
        # action='append',
        required=False,
        type=list_of_strings,
        default=None,
        nargs='?',
        help='Window(s) to process (list i.e. 1235, 3456).'
    )

    parser.add_argument(
        '--dates',
        required=False,
        type=list_of_strings,
        default=None,
        nargs='?',
        help='Dates to process in the format YYYYMMDD  (list seperated by commas).'
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
        default = os.path.join(LOCALPATH, 'hmsa_origin_ship.json'),
        nargs='?',
        help='Instrument Model file path.'
    )

    parser.add_argument(
        '--slitsizeum',
        # metavar = 'NAME',
        required=True,
        type=str,
        # default = AS REQUIRED,
        nargs='?',
        help='Slit size (width) in micrometers.'
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
        default=6,
        nargs='?',
        help='Readnoise value (ADU) to be used for readnoise correction.'
    )
    args = parser.parse_args()
    config = L1AConfig(
        rootdir=args.rootdir,
        dest=args.dest,
        dest_prefix=args.dest_prefix,
        overwrite=args.overwrite,
        windows=args.windows,
        dates=args.dates,
        dark=args.dark,
        model=args.model,
        slitsizeum=args.slitsizeum,
        chunksize=args.chunksize,
        readnoise=args.readnoise
    )

    main(config)
# %%
