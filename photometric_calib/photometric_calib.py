#%%
'''
This script creates the photometric calibration image for hmsao. 

General idea: 
    1. set up the midesigner model using hmsa_origin_ship.json
    2. open the calib image
    3. crop and resize the image in the form of a ds with dims ['gamma','beta']
    4. for each window, create the conversion_factor map
    5. stitch the image (conversion_factor map) back together to the orginal size
    6. check if this image and the a test image line up and make sense.
'''
#%%
import numpy as np
import matplotlib.pyplot as plt
import xarray as xr
from typing import SupportsFloat as Numeric
from typing import Dict, Iterable, List
import os
from glob import glob
from misdesigner import *
from PIL import Image
from astropy.io import fits
from tqdm import tqdm
#%%


# %%
def png_2_modelready_ds(predictor, fimg):
    imgsize = (len(predictor.beta_grid), len(predictor.gamma_grid))
    data = Image.fromarray(fimg)
    data = data.rotate(-.311,resample=Image.Resampling.BILINEAR, fillcolor=np.nan)
    data = data.transpose(Image.Transpose.FLIP_LEFT_RIGHT)
    image = Image.new('F', imgsize, color=np.nan)
    image.paste(data, (110, 410))
    data = np.asarray(image).copy() 
    del image
    # array -> DataArray
    image = xr.DataArray(
        data,
        dims=['gamma', 'beta'],
        coords={
            'gamma': predictor.gamma_grid,
            'beta': predictor.beta_grid
        },
        attrs={'unit': 'ADU/s'}
    )
    return image

def pick_out_window_ds(predictor,imgds,win_name):
    for _, window, xform, coords,res in predictor._imaps:
        if window.name != win_name:
            continue
        xran = (coords['beta'].max(), coords['beta'].min())
        yran = (coords['gamma'].min(), coords['gamma'].max())
        im =imgds.sel(gamma = slice(*yran), beta = slice(*xran))
        return im

# %%
def get_model_wlmap_of_window(model,win_name,return_diff_map:bool = True):
    mmap = model.mosaic_map(unique=True, report=False)
    for s in mmap['slit']:
        source = mmap['source'].sel(slit = s).drop_vars('slit')
        for window in model.mosaic.windows:
            if window.name != win_name: continue
            xran = window.get_xrange()
            yran = window.get_yrange()
            smap = mmap.where(source == True, drop = True)
            smap = smap.sel(gamma = slice(*yran), beta = slice(*xran))
            if any(value == 0 for value in smap.sizes.values()):
                continue
            else:
                wlmap = smap.wavelength


    diff = wlmap.diff(dim = 'beta', label = 'lower')
    zcol = diff.isel(beta = 0).values
    zcol = zcol.reshape(-1,1)
    adiff = np.hstack((zcol,diff.values))
    if return_diff_map:
        delwlmap = xr.DataArray(
            np.abs(adiff),
            coords = {
                'gamma': wlmap.gamma.values,
                'beta' : wlmap.beta.values,
            },
            name = 'Wavelength_binsize',
            attrs={
                'units': 'Å'
            }
        )
        return wlmap, delwlmap
    else:
        wlmap

def reverse_transform(image, fimg_shape):
    # image is your xarray DataArray
    data = np.asarray(image).copy()

    # Step 1: crop back out (same size as original fimg)
    data = Image.fromarray(data)
    data = data.crop((110, 410, 110 + fimg_shape[1], 410 + fimg_shape[0]))

    # Step 2: undo flip
    data = data.transpose(Image.Transpose.FLIP_LEFT_RIGHT)

    # Step 3: undo rotation (opposite sign)
    data = data.rotate(+0.311, resample=Image.Resampling.BILINEAR, fillcolor=np.nan)

    # Back to numpy
    fimg_recovered = np.asarray(data)

    return fimg_recovered
#######################################################################################

#%%
#set up the midesigner model using hmsa_origin_ship.json
model = MisInstrumentModel.load('../hmsa_origin_ship.json')
predictor = MisCurveRemover(model)

#initialize calibration image and final image (cfmap)
fdir = 'calib-data/raw'
fnames = glob(os.path.join(fdir,'*light*.png'))
fimg=[]
for f in fnames:
    img = Image.open(f) # total counts
    img = np.asarray(img, dtype = float)
    img /= int(''.join(n for n in f if n.isnumeric())) #countrate= totalcounts/exposure
    fimg.append(img)
fimg = fimg[0]
cimgds = png_2_modelready_ds(predictor,fimg) #turn png into ds with model coords
cfmap = cimgds.copy()

#initialize calib curve data
calibds = xr.open_dataset('calib-data/lightbox_calib_curve.nc')
xp = calibds['wavelength'].values #A
yp = calibds['brightness'].values #Rayleigh/A


for win in tqdm(predictor.windows):
    
    #calibration image (Countrate)
    calibimg = pick_out_window_ds(predictor,cimgds,win) # ds of chosen window
    wlds, delwlds = get_model_wlmap_of_window(model,win,True) # type: ignore #get the wl map of chosen window
    meas = calibimg/delwlds #Countrate/A -> measurement of lamp by hms, in units directly comparable to R/A

    #calibration curve (R)
    bright = [np.interp(wlds.isel(gamma=i).values, xp, yp) for i in range(len(wlds.gamma.values)) ] #Rayleigh/A
    bright = xr.DataArray(
        bright,
        coords = {
            'gamma':wlds.gamma.values,
            'beta': wlds.beta.values
        }
    )

    #Calibration Conversion Factor map for window = (R/A) / (Countrate/A)
    confac = bright/meas

    #add bck to full img
    gamslice = slice(float(calibimg.gamma.min()), float(calibimg.gamma.max()))
    betaslice = slice(float(calibimg.beta.max()), float(calibimg.beta.min()))
    cfmap.loc[dict(gamma = gamslice, beta = betaslice)] = confac.values



# %%
#add attributes
cfmap = cfmap.assign_attrs({
    'units':'CountRate/R',
    'long_name':'Photmetric Conversion Factor',
    'short_name':'CF'
})
cfmap = cfmap.to_dataset(name = 'cf')
#%%
#save cf map
encoding = {var: {'zlib': True}
                                for var in (*cfmap.data_vars.keys(), *cfmap.coords.keys())}
sub_outfname = 'hmsao_photometric_calib.nc'
print('Saving %s...\t' % (sub_outfname), end='')
cfmap.to_netcdf(sub_outfname, encoding=encoding)
print(f'Done.')
# %%
cfmap.plot(vmax = 1000)

# %%
TEST = False
if TEST:
##test image 
    fdatdir = '//home/charmi/locsststor/raw/hmsorigin'
    date = '20250908'
    fnames = glob(os.path.join(fdatdir,f'{date}/*.fit*?'))
    fnames.sort()

    with fits.open(fnames[100]) as hdul:
        header = hdul[1].header
        data = hdul[1].data
        data = np.asarray(data, dtype = float)
        data /= int(header['HIERARCH EXPOSURE_S'])

    imgds = png_2_modelready_ds(predictor,data)

    
    calibrated = imgds*cfmap
    calibrated = calibrated.assign_attrs({
        'units':'R',
        'Description':'Photometrically Calibrated image'
    })
    
    vmin = np.nanpercentile(calibrated.values, 1)
    vmax = np.nanpercentile(calibrated.values, 99.9)

    plt.figure()
    calibrated.plot(vmin = vmin, vmax = vmax)
    

    a = reverse_transform(calibrated, np.shape(fimg))
    
    plt.figure()
    plt.imshow(a, vmax=1e7)
    plt.colorbar()
# %%
