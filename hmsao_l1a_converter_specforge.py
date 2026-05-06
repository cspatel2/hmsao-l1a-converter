# %%
# %% Level 1A converter - Level 1A (L1A) data are reconstructed, unprocessed instrument data at full resolution, time-referenced, and annotated with ancillary information, including radiometric and geometric calibration coefficients and georeferencing parameters (e.g., platform ephemeris) computed and appended but not applied to L0 data.

# %%
import argparse
from datetime import datetime, timezone, timedelta
import gc
import json
from pathlib import Path
import os
import sys
from time import perf_counter_ns
from astropy import conf
from matplotlib import pyplot as plt
import numpy as np
from glob import glob
import astropy.io.fits as fits
from prism_imageproc import ImageStraightener
from spectra_shift import wavelength_shift_stretch
import pytz
import xarray as xr
from tqdm import tqdm
from typing import Dict, Iterable, List, SupportsFloat as Numeric
from skmpython import datetime_in_timezone


from dataclasses import dataclass

# import warnings
# warnings.simplefilter('once')

import warnings

warnings.filterwarnings(
    "ignore",
    message=".*astropy_xarray.index.AstropyIndex.equals.*",
    category=FutureWarning,
)


# %%
LOCALPATH = os.path.dirname(os.path.realpath(__file__))
sys.path.append(os.path.join(LOCALPATH))
from l1_helpers import *

# %%


@dataclass
class L1AConfig:
    """datacladd for L2AConfig
    Attributes:
    rootdir: Root Directory containing HiT&MIS data
    destdir: Root directory where L1 data will be stored.
    dest_prefix: Prefix of the saved L1 data finename.
    overwrite: If you want to rewrite an existing file, then True. Defaults to False.
    windows: Window(s) to process (list i.e. 1235, 3456).
    dates: Dates to process in the format YYYYMMDD  (list seperated by commas).
    dark: Dark data (.nc) file path. If not provided, then dark correction will be skipped.
    slitsizeum: Slit size (width) in micrometers.
    chunksize: Number of files per chunk.
    readnoise: Readnoise value (ADU) to be used for readnoise correction.
    model: Instrument Model file path (.bin). Create using SpectraForge.
    wl_calib_path: Wavelength calibration params file path (.json). If not provided, then wavelength correction will be skipped. The .json file should be in the format of a dictionary with window names as keys and another dictionary as values containing wl_offset, wl_stretch, and wl_pivot.
    """    
    rootdir: str | Path
    destdir: str | Path
    dest_prefix: str | None
    overwrite: bool
    windows: List[str]
    dates: List[str]
    dark: str | Path | None
    slitsizeum: str
    chunksize: int
    readnoise: float
    model: str | Path
    wl_calib_path: str | Path


def list_of_strings(arg: str) -> List[str]:
    return arg.split(",")


def str2bool(value: str) -> bool:
    if value.lower() in ("true", "1", "t", "y", "yes"):
        return True
    elif value.lower() in ("false", "0", "f", "n", "no"):
        return False
    raise ValueError("Invalid boolean value: {}".format(value))


# %%


def main(config: L1AConfig):
    """converts Hit&Mis raw data into L1A files (.nc).
    includes the following processing steps:
    1. reorganize dataset by UTC timestamps
    2. divide each raw image into panels
    3. exposure normalization (counts -> counts/s)
    4. hot pixel correction (for long exposures)
    5. dark/bias subtraction (if dark data provided)
    6. straighten each panel
    7. spectral calibration (x = wavelegth)
    8. spatial calibration (y = zenith angle)
    9. save as .nc file with appropriate metadata and attributes

    Args:
        config (L1AConfig): configuration dataclass containing all the necessary parameters for the conversion process.

    Raises:
        ValueError: provided model file does not exist.
        ValueError: provided wavelength calibration params file does not exist.
        ValueError: provided windows are not valid.
        ValueError: provided root directory does not exist.
        ValueError: no .fit files found in the provided root directory. 
    """    
    config.rootdir = Path(config.rootdir).expanduser()
    config.destdir = Path(config.destdir).expanduser()
    config.dark = Path(config.dark).expanduser() if config.dark is not None else None
    config.model = Path(config.model).expanduser()
    config.wl_calib_path = Path(config.wl_calib_path).expanduser()

    slitsize = config.slitsizeum

    # 0. Paths

    if config.destdir.is_file():
        print("Destination path provided is a file. Directory path required.")
        sys.exit()
    else:
        config.destdir.mkdir(parents=True, exist_ok=True)
    print(f"destination dir set to: {config.destdir}\n")

    # 1. Check provided arguments and Initialize

    # Create model and confirm that the Instrument file provided works
    if config.model.is_file():
        straightener = ImageStraightener.load(config.model)

    else:
        raise ValueError(
            "Model file provided does not exist. Please provide a valid model file path."
        )

    if config.wl_calib_path.is_file():
        with config.wl_calib_path.open() as f:
            wl_calib_params = json.load(f)
            print(
                f'wavelength correctioon will be applied to windows: {", ".join(wl_calib_params.keys())}'
            )
    else:
        wl_calib_params = None
        print(
            "Wavelength calibration params file (.json) not found. Wavelength correction will be skipped."
        )

    if config.dest_prefix is None:
        config.dest_prefix = f"hms"
    if "l1a" not in config.dest_prefix.lower():
        config.dest_prefix = config.dest_prefix + "_l1a"

    # Check that user provided windows can be processed
    if config.windows is not None:
        valid_windows = list(set(config.windows) & set(straightener.windows))
        if len(valid_windows) == 0:
            raise ValueError(
                f"Invalid Window names: {config.windows}. Available window names are {straightener.windows}"
            )
    else:
        valid_windows = straightener.windows

    print(f'Windows to be processed: {", ".join(valid_windows)}')

    # check if root dir exists
    if not config.rootdir.is_dir():
        print("Root Directory provided does not exist.")
        sys.exit()

    # get all the fits files from all the subdirs, sorted by time
    dirlist = get_all_dirs(config.rootdir)
    files = np.array([], dtype=object)
    tstamps = np.array([])
    print(f"Total Number of Directories in Rootdir: {len(dirlist)}")
    dirlist.sort()
    for d in dirlist:
        f = list(d.glob("*.fit*"))
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
        raise ValueError("No .fit(s) files in rootdir")
    else:
        # type: ignore
        print(f"Total Number of Files to Process: {len(files)}\n")

    # final list of files to process
    idx = np.argsort(tstamps)
    tstamps = np.sort(tstamps)
    files = files[idx]

    if config.dates is not None:  # if processing dates are given, then use those
        dates = np.sort(config.dates)
        start_date = datetime.strptime(dates[0], "%Y%m%d").replace(tzinfo=pytz.UTC)
        end_date = datetime.strptime(dates[-1], "%Y%m%d").replace(tzinfo=pytz.UTC)
    else:  # get start and end date of the full dataset
        start_date = datetime.fromtimestamp(tstamps[0], tz=pytz.utc)
        end_date = datetime.fromtimestamp(tstamps[-1], tz=pytz.utc)

    print(f"Start DateTime: {start_date}")
    print(f"End DateTime: {end_date} \n")

    # break up into individual days, day is midnight to midnight
    st_date = start_date.date() - timedelta(days=1)
    lst_date = end_date.date() + timedelta(days=1)
    main_flist: Dict[datetime, List[str]] = {}
    all_files = []
    print("Dates with data: ", end="")
    data_found = False
    first = True
    while st_date <= lst_date:
        _st_date = st_date
        start = datetime_in_timezone(
            pytz.utc, st_date.year, st_date.month, st_date.day
        )  # midnight
        st_date += timedelta(days=1)
        stop = start + timedelta(days=1)  # to midnight
        start_ts = start.timestamp()
        stop_ts = stop.timestamp()
        valid_files = [
            f if start_ts <= t < stop_ts else "" for f, t in zip(files, tstamps)
        ]
        while "" in valid_files:
            valid_files.remove("")
        if len(valid_files) > 0:
            data_found = True
            main_flist[_st_date] = valid_files  # type: ignore
            all_files += valid_files
            if first:
                print(_st_date, end="")
                first = False
            else:
                print(",", _st_date, end="")
            sys.stdout.flush()
    if not data_found:
        print("None")
    print("\n")

    print(f"data will be saved to: {config.destdir}\n")

    del idx, tstamps, files
    gc.collect()

    # get dark data
    is_dark_subtracted = "is"
    if config.dark is not None:
        dfile = config.dark
        darkds = xr.load_dataset(dfile)
    else:
        darkds = None
        is_dark_subtracted += " not"

    for key, filelist in main_flist.items():
        # print(f'[{key:%Y-%m-%d}] Starting conversion...')
        yymm = f"{key:%Y%m}"
        yymmdd = f"{key:%Y%m%d}"
        prefix = config.dest_prefix
        # each month is a new directory at dest
        (config.destdir / yymm).mkdir(parents=True, exist_ok=True)
        # file names/paths of the complete file of a given day and window
        outfnames = [
            f"{yymm}/{prefix}_{yymmdd}_{window}*.nc" for window in valid_windows
        ]
        outfpaths = [str(config.destdir / outfname) for outfname in outfnames]

        # check if any of the output files already exist
        skip_processing = False
        for pathidx, outfpath in enumerate(outfpaths):
            numfiles = glob(outfpath)
            # print(f'{outfpath} has {len(numfiles)} related files')/
            if len(numfiles) > 0:
                print(f"overwrite = {config.overwrite}")
                if config.overwrite:
                    for i in glob(outfpath):
                        print(f"{i} removed.")
                        os.remove(i)
                elif not config.overwrite:  # if overwrite is false and file exist
                    for i in glob(outfpath):
                        print(f"{i} already exists, skipping")
                    skip_processing = True

        if not skip_processing:
            absstart = perf_counter_ns()
            # split 1 day into len(filesperday)/n loops
            n = config.chunksize
            chunks = int(np.ceil(len(filelist) / n))
            ndigits = int(np.ceil(np.log10(chunks)))
            iterlim = chunks * n

            # for subidx, sublist in enumerate(subfilelists):
            for subidx in range(chunks):
                out_countrate = {k: [] for k in valid_windows}
                out_noise = {k: [] for k in valid_windows}
                sublist = filelist[subidx * n : (subidx + 1) * n]
                for _, fn in enumerate(
                    tqdm(
                        sublist,
                        desc=f"{key:%Y-%m-%d} - [{subidx+1:0{ndigits}}/{chunks}]",
                    )
                ):
                    # initialize the index of the hdul data using the first file
                    # key = 'IMAGE'  # use hdul.info() to see all keys in file
                    with fits.open(fn) as hdul:
                        hdu = hdul["IMAGE"]
                        header = hdu.header  # type: ignore
                        tstamp = get_tstamp_from_hdu(hdu)  # s
                        exposure = get_exposure_from_hdu(hdu)  # s
                        temp = header["CCD-TEMP"]  # C
                        # 1. get img, and read noise
                        # counts
                        data_ = np.asarray(hdu.data, dtype=float)  # type: ignore
                        data = data_.copy()

                        readnoise = np.full(
                            data.shape, config.readnoise, dtype=float
                        )  # counts

                        # 1b. hot pixel correction for long exposures
                        _, data = find_outlier_pixels(data)

                        # 2. dark/bias correction
                        if darkds is not None:
                            dark = np.asarray(darkds["countrate"].values, dtype=float)
                            dark_noise = np.asarray(
                                darkds["countrate_err"].values, dtype=float
                            )
                            bias = np.asarray(darkds["bias"].values, dtype=float)
                            bias_noise = np.asarray(
                                darkds["bias_err"].values, dtype=float
                            )
                            data -= bias + dark * exposure  # counts
                            data = np.clip(
                                data, a_min=0, a_max=None
                            )  # set negative values to 0 after dark/bias subtraction
                            readnoise = np.sqrt(
                                readnoise**2
                                + (dark_noise * exposure) ** 2
                                + bias_noise**2
                            )
                        # 3. total counts -> counts.sec
                        # 4. Crop and resize image
                        # 5. straighten img
                        data = data / exposure  # counts/s
                        mapped_image = straightener.load_image(data)
                        straightened_images = mapped_image.straighten_image()

                        for window in valid_windows:
                            #straighten
                            data = straightened_images[window]
                            #spatial and spectral calibration
                            if wl_calib_params is not None and window in wl_calib_params.keys():
                                corrected_wls = wavelength_shift_stretch(
                                    data.wavelength.values,  # type: ignore
                                    wl_calib_params[window]["wl_offset"],
                                    wl_calib_params[window]["wl_stretch"],
                                    wl_calib_params[window]["wl_pivot"],
                                )
                                data = apply_spatial_and_spectral_calibration(data, corrected_wls)  # type: ignore
                            else:
                                data = convert_y_to_zenithangle(data)  # type: ignore
                            # 6. Save
                            data = data.expand_dims(  # type: ignore
                                dim={"tstamp": (tstamp,)}
                            ).to_dataset(
                                name="countrate", promote_attrs=True
                            )  # type: ignore
                            data["exposure"] = xr.Variable(
                                dims="tstamp", data=[exposure], attrs={"unit": "s"}
                            )
                            data["ccdtemp"] = xr.Variable(
                                dims="tstamp", data=[temp], attrs={"unit": "C"}
                            )
                            out_countrate[window].append(data)
                        # 7. readnoise propogation
                        mapped_readnoise = straightener.load_image(readnoise)
                        straightened_readnoise = mapped_readnoise.straighten_image()

                        for window in valid_windows:
                            rn = straightened_readnoise[window]
                            if wl_calib_params is not None and window in wl_calib_params.keys():
                                corrected_wls = wavelength_shift_stretch(
                                    rn.wavelength.values,  # type: ignore
                                    wl_calib_params[window]["wl_offset"],
                                    wl_calib_params[window]["wl_stretch"],
                                    wl_calib_params[window]["wl_pivot"],
                                )
                                rn = apply_spatial_and_spectral_calibration(rn, corrected_wls)  # type: ignore
                            else:
                                rn = convert_y_to_zenithangle(rn)  # type: ignore
                            rn = rn.expand_dims(  # type: ignore
                                dim={"tstamp": (tstamp,)}
                            ).to_dataset(name="noise", promote_attrs=True)
                            out_noise[window].append(rn)

                # print(len(out_noise[window]))
                # Create Dataset and save
                for window in valid_windows:
                    sub_outfname = (
                        f"{yymm}/{prefix}_{yymmdd}_{window}[{subidx:0{ndigits}}].nc"
                    )
                    sub_outfpath = config.destdir / sub_outfname
                    ds: xr.Dataset = xr.concat(out_countrate[window], dim="tstamp")
                    gc.collect()
                    ds.attrs.update(
                        dict(
                            Description=" HMSA-O Straighted Spectra",
                            ROI=f"{int(window)/10:0.1f} nm",
                            slit_size_um=str(slitsize),
                            DataProcessingLevel="1A",
                            FileCreationDate=datetime.now().strftime(
                                "%m/%d/%Y, %H:%M:%S EDT"
                            ),
                            ObservationLocation="Swedish Institute of Space Physics/IRF (Kiruna, Sweden)",
                            Note=f"data {is_dark_subtracted} dark corrected.",
                        )
                    )
                    if config.readnoise is not None:
                        ds = xr.merge([ds, xr.concat(out_noise[window], dim="tstamp")])
                    ds["countrate"].attrs["units"] = "ADU/s/nm"
                    ds["countrate"].attrs["long_name"] = "Line Intensity"
                    ds["noise"].attrs["units"] = "ADU/s/nm"
                    ds["noise"].attrs["long_name"] = "Noise"
                    ds["noise"].attrs[
                        "eqn"
                    ] = r"Noise is given by sqrt{RN^2 + Counts}/exp"
                    ds["tstamp"].attrs["units"] = "s"
                    ds["tstamp"].attrs[
                        "description"
                    ] = "Seconds since UNIX epoch 1970-01-01 00:00:00 UTC"
                    ds["za"].attrs["units"] = "degrees"
                    ds["za"].attrs["description"] = "Zenith angle"
                    ds["za"].attrs["long_name"] = "Zenith Angle"
                    encoding = {
                        var: {"zlib": True}
                        for var in (*ds.data_vars.keys(), *ds.coords.keys())
                    }
                    print("Saving %s...\t" % (sub_outfname), end="")
                    sys.stdout.flush()
                    tstart = perf_counter_ns()
                    ds.to_netcdf(sub_outfpath, encoding=encoding)
                    tend = perf_counter_ns()
                    print(f"Done. [{(tend-tstart)*1e-9:.3f} s]")

                del out_countrate
                gc.collect()
            absend = perf_counter_ns()

            print(f"\nDone: {key:%Y-%m-%d}, {(absend - absstart)*1e-9:.3f} s")
        else:
            continue


# %%
if __name__ == "__main__":
    # argument parser
    parser = argparse.ArgumentParser(
        description="Convert HiT&MIS L0 data to L1A data, with exposure normalization and dark subtraction. It uses MisInstrument Model to extract ROI and perfoms line straightening. This program requires a instrument defination JSON file (See MisInstrument implementation for details.)."
    )

    parser.add_argument(
        "--rootdir",
        metavar="rootdir",
        # required = True,
        type=str,
        # default =
        nargs="?",
        help="Root Directory containing HiT&MIS data",
    )

    parser.add_argument(
        "--dest",
        metavar="destdir",
        # required = False,
        type=str,
        default=os.getcwd(),
        nargs="?",
        help="Root directory where L1 data will be stored.",
    )

    parser.add_argument(
        "--dest_prefix",
        metavar="dest_prefix",
        # required = False,
        type=str,
        default=None,
        nargs="?",
        help="Prefix of the saved L1 data finename.",
    )

    parser.add_argument(
        "--overwrite",
        required=False,
        type=str2bool,
        default=False,
        nargs="?",
        help="If you want to rewrite an existing file, then True. Defaults to False.",
    )

    parser.add_argument(
        "--windows",
        # metavar = 'NAME',
        # action='append',
        required=False,
        type=list_of_strings,
        default=None,
        nargs="?",
        help="Window(s) to process (list i.e. 1235, 3456).",
    )

    parser.add_argument(
        "--dates",
        required=False,
        type=list_of_strings,
        default=None,
        nargs="?",
        help="Dates to process in the format YYYYMMDD  (list seperated by commas).",
    )

    parser.add_argument(
        "--dark",
        # metavar = 'NAME',
        required=False,
        type=str,
        # default = None,
        nargs="?",
        help="Dark data (.nc) file path.",
    )

    parser.add_argument(
        "--model",
        # metavar = 'NAME',
        required=True,
        type=str,
        default=os.path.join(LOCALPATH, "hmsa_origin_ship.json"),
        nargs="?",
        help="Instrument Model file path.",
    )

    parser.add_argument(
        "--slitsizeum",
        # metavar = 'NAME',
        required=True,
        type=str,
        # default = AS REQUIRED,
        nargs="?",
        help="Slit size (width) in micrometers.",
    )

    parser.add_argument(
        "--chunksize",
        # metavar = 'NAME',
        required=False,
        type=int,
        default=10,  # fix this later depending on what the ideal number of files should be per chunk
        nargs="?",
        help="Number of files per chunk.",
    )

    parser.add_argument(
        "--readnoise",
        # metavar = 'NAME',
        required=False,
        type=float,
        default=6,
        nargs="?",
        help="Readnoise value (ADU) to be used for readnoise correction.",
    )
    args = parser.parse_args()
    config = L1AConfig(
        rootdir=args.rootdir,
        destdir=args.dest,
        dest_prefix=args.dest_prefix,
        overwrite=args.overwrite,
        windows=args.windows,
        dates=args.dates,
        dark=args.dark,
        model=args.model,
        slitsizeum=args.slitsizeum,
        chunksize=args.chunksize,
        readnoise=args.readnoise,
    )

    main(config)
# %%
