# %% functions


from datetime import datetime
import os
from typing import Iterable, List
import astropy.io.fits as fits
import numpy as np
import xarray as xr
from typing import SupportsFloat as Numeric
from skimage import transform

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
            with fits.open(fname) as hdul: #type: ignore
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
        return list(map(get_tstamp_from_fname, fname)) #type: ignore


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
        return [zenith_angle(x) for x in gamma_mm] #type: ignore
    if np.min(gamma_mm) < 0:
        sign = -1
    else:
        sign = 1
    num = -(gamma_mm-(sign*yoffset))*(f1+f2-D) #type: ignore
    den = f1*f2 #type: ignore
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
    mxi, myi = np.meshgrid(ds.wavelength.values, angles) #type: ignore
    imin, imax = np.nanmin(myi), np.nanmax(myi)
    myi -= imin  # shift to 0
    myi /= (imax - imin)  # normalize to 1
    myi *= (len(angles))  # adjust #type: ignore

    # coordinate map in the output image
    if np.nanmin(angles) < 0: #type: ignore
        sign = 1
    else:
        sign = -1
    linangles = np.linspace(np.min(angles), np.max(angles), len(
        angles), endpoint=True)[::sign]  # array of linear zenith angles #type: ignore
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
    timg = transform.warp(ds.values, imap, order=1, cval=np.nan) #type: ignore

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
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 6), dpi=300) #type: ignore
        fig.tight_layout()

        vmin = np.nanpercentile(ds.values, 1) #type: ignore
        vmax = np.nanpercentile(ds.values, 99) #type: ignore
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

def get_exposure_from_fn(fn: str) -> float:
    """ get the exposure in s from the filename for the filename formatted 'PREFIX_EXP.png'

    Args:
        fn (str): filename or filepath

    Returns:
        float: exposure in s.
    """    
    return float(fn.strip('.png').split('_')[-1])