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
from until_functions import *

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
    help='Window(s) to process (list i.e. 1235, 3456).'
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
    else: 
        if 'l1a' not in args.dest_prefix:
            args.dest_prefix += '_l1a'

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
    files = glob(os.path.join(args.rootdir, '*.png*' ))
    

    # readnoise option
    readnoise = None

    print(f'data will be saved to: {args.dest}\n')

    # get dark data
    is_dark_subtracted = 'is'
    if args.dark is not None:
        dfile = args.dark
        darkds = xr.load_dataset(dfile)
    else:
        darkds = None
        is_dark_subtracted += ' not'
    print (len(files))
    ################### MAIN LOOP ##################################
    out_countrate = {k: [] for k in windows}
    out_noise = {k: [] for k in windows}
    for idx, file in enumerate(files):

        prefix = args.dest_prefix
        
        # file names/paths of the complete file of a given day and window
        outfnames = [f"{prefix}_{window}*.nc" for window in windows]
        outfpaths = [os.path.join(args.dest, outfname) for outfname in outfnames]

        # check if any of the output files already exist
        skip_processing = False
        for pathidx, outfpath in enumerate(outfpaths):
            numfiles = glob(outfpath)
            # print(f'{outfpath} has {len(numfiles)} related files')
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

                imgsize = (len(predictor.beta_grid), len(predictor.gamma_grid))
                # for subidx, sublist in enumerate(subfilelists):

                # out_countrate = {k: [] for k in windows}
                # out_noise = {k: [] for k in windows}

                exposure = get_exposure_from_fn(file)  # s
                # 1. get img
                data = Image.open(file)
                data = np.asarray(data, dtype=float)  # counts
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
                    data = data.expand_dims(dim={'idx': (idx,)}).to_dataset(name='intensity', promote_attrs=True)
                    data['exposure'] = xr.Variable(dims='idx', data=[exposure], attrs={'unit': 's'})
                    out_countrate[window].append(data)
            
                if args.readnoise is not None:
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
                        attrs={'unit': 'ADU/s'}
                    )

                    for window in windows:
                        rn = predictor.straighten_image(noise, window, coord='Slit')
                        rn = convert_gamma_to_zenithangle(rn)
                        rn = rn.expand_dims(dim={'idx': (idx,)}).to_dataset(name='noise', promote_attrs=True)
                        # rn['exposure'] = xr.Variable(dims='tstamp', data = [exposure], attrs={'unit': 's'})
                        # rn['ccdtemp'] = xr.Variable(dims='tstamp', data = [temp], attrs={'unit': 'C'})
                        out_noise[window].append(rn)
        
                # print(len(out_noise[window]))    
                # Create Dataset and save
                
            else:
                continue

    for window in windows:
        sub_outfname = f"{args.dest_prefix}_{window}.nc"
        sub_outfpath = os.path.join(args.dest, sub_outfname)
        ds: xr.Dataset = xr.concat(out_countrate[window], dim='idx')
        gc.collect()
        ds.attrs.update(
            dict(Description=" HMSA-O Straighted Spectra (png)",
                    ROI=f'{str(window)} nm',
                    DataProcessingLevel='1A',
                    FileCreationDate=datetime.now().strftime("%m/%d/%Y, %H:%M:%S EDT"),
                    ObservationLocation='LoCSST | Lowell, MA',
                    Note=f'data {is_dark_subtracted} dark corrected.',
                    )
        )
        if args.readnoise is not None:
            ds = xr.merge([ds, xr.concat(out_noise[window], dim='idx')])
        ds['intensity'].attrs['unit'] = 'ADU/s'
        ds['intensity'].attrs['long_name'] = 'Line Intensity'
        ds['noise'].attrs['unit'] = 'ADU/s'
        ds['noise'].attrs['long_name'] = 'Noise'
        ds['noise'].attrs['eqn'] = r'Noise is given by sqrt{RN^2 + Counts}/exp'
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

    print(f'\nDone:{(absend - absstart)*1e-9:.3f} s')


# %%
if __name__ == '__main__':
    main(parser)
# %%
