#%%
from matplotlib import colors, pyplot as plt
import numpy as np
from PIL import Image
import astropy.io.fits as fits
import xarray as xr
from typing import Dict, Iterable, List, SupportsFloat as Numeric
from misdesigner import MisInstrumentModel, MisCurveRemover
from skimage import transform
from matplotlib.ticker import MaxNLocator
#%%
model = MisInstrumentModel.load('hmsa_origin_ship.json')
# %%
predictor = MisCurveRemover(model)
# %% load image
with fits.open('trial_data/20250126/20250126000322.085.fits') as hdul:
    data = hdul[1].data
    header = hdul[1].header
#plot image
vmin = np.percentile(data, 1)
vmax = np.percentile(data, 99)
plt.imshow(data, vmin=vmin, vmax=vmax, cmap='gray')
# %% fix image 
data = Image.fromarray(data)
data = data.transpose(Image.Transpose.FLIP_LEFT_RIGHT)
IMGSIZE = (len(predictor.beta_grid), len(predictor.gamma_grid))
image = Image.new('F', IMGSIZE, color=np.nan)
image.paste(data, (110, 410))
data = np.asarray(image).copy()
del image
# create a dataarray to plug into straighten_image()
img = xr.DataArray(data, 
                   dims=['gamma', 'beta'],
                   coords={'gamma': predictor.gamma_grid, 'beta': predictor.beta_grid},
                   attrs={'unit': 'ADU'})
#straighten the image for all windows (testing coord = 'Slit')
# for win in predictor.windows:
#     ds = predictor.straighten_image(img, win, inplace=True, coord='Slit')
#     fig, ax = plt.subplots()
#     ds.plot(ax=ax)
#     plt.show()

# %%
def get_sourceslits_for_window(window: str, predictor: MisCurveRemover) -> List[str]:
    '''Gets a list of slits that illuminate the window. The slits are sorted in descending order by total number of illuminated pixels.'''
    if window not in predictor.windows:
        raise ValueError(f'Window {window} not found in the mosaic. Available windows are {predictor.windows}')
    miswindow = predictor._model.mosaic.get_window(window)
    mmap = predictor._mmap
    xran = miswindow.get_xrange()
    yran = miswindow.get_yrange()
    source = mmap['source'].sel(beta = slice(*xran), gamma = slice(*yran))
    slits = source.where(source ==True, drop=True).sum(dim=['beta', 'gamma']).sortby('slit', ascending=False)
    if len(slits) == 0:
        raise ValueError(f'No slits illuminate window {window}')
    return slits.slit.values.tolist()

# %%
window  = '5577'
slit    = get_sourceslits_for_window(window, predictor)[0]
ymid    = predictor._model.slits[slit].y
#%%
def zenith_angle(gamma_mm:Numeric|Iterable[Numeric], f1:Numeric=30, f2:Numeric=30, D:Numeric= 24, yoffset:Numeric = 12.7) -> Numeric:
    """Calculates the zenith angle in degrees from the gamma(mm) in slit coordinates.

    Args:
        gamma_mm (Numeric | Iterable[Numeric]): gamma (mm) in slit (instrument coordinate system) coordinates.
        f1 (Numeric, optional): focal length (mm) of the 1st lens in the telecentric foreoptic. Defaults to 30 mm.
        f2 (Numeric, optional): focal length (mm) of the 2nd lens in the telecentric foreoptic. Defaults to 30 mm.
        D (Numeric, optional): Distance (mm) between the two lens. Defaults to 24 mm.
        yoffset (Numeric, optional): the distance between the optic axis of the telescope to the x-axis of the instrument coordinate system. Defaults to 12.7 mm.

    Returns:
        Numeric: the zenith angle in degrees.

    """    
    if isinstance(gamma_mm, (int, float)):
        return [zenith_angle(x) for x in gamma_mm]
    if np.min(gamma_mm) < 0: sign = -1
    else : sign = 1
    num = -(gamma_mm-(sign*yoffset))*(f1+f2-D)
    den = f1*f2
    return np.rad2deg(np.arctan(num/den))


# %%
ds = predictor.straighten_image(img, '6563', inplace=True, coord='Slit')
#%%
# ds['gamma'] = zenith_angle(ds.gamma.values)
# ds['gamma'] = ds['gamma'].assign_attrs({'unit': 'deg', 'long_name': 'Zenith Angle'})
# %%
fig, ax = plt.subplots()
vmin = np.nanpercentile(ds.values, 1)
vmax = np.nanpercentile(ds.values, 99)
ds.plot(ax=ax, vmin=vmin, vmax=vmax)
plt.show()

#%%
ds = predictor.straighten_image(img, '5577', inplace=True, coord='Slit')
imap = np.zeros((2,*(ds.shape)),dtype = float)
# %%
angles = zenith_angle(ds.gamma.values)
if np.nanmin(angles) < 0: sign = 1
else : sign = -1
mxi , myi = np.meshgrid(ds.wavelength.values, angles)
imin, imax = np.nanmin(myi), np.nanmax(myi)
myi -= imin
myi /= (imax - imin)
myi *= (len(angles))

#%%
linangles = np.linspace(np.min(angles), np.max(angles), len(angles),endpoint=True)[::sign]
mxo, myo = np.meshgrid(ds.wavelength.values, linangles)
omin,omax = np.nanmin(myo), np.nanmax(myo)
myo -= omin
myo /= (omax - omin)
myo *= (len(linangles))
#%%
a = np.arange(0,ds.shape[-1])
b = np.tile(a,(ds.shape[0],1))
imap[0,:,:] = myi
imap[1,:,:] = b
#%%
timg = transform.warp(ds.values, imap, order = 0)
vmin = np.nanpercentile(timg, 1)
vmax = np.nanpercentile(timg, 99)
plt.imshow(timg, vmin=vmin,vmax = vmax)
plt.colorbar()
#%%
ds['gamma'] = angles
ds['gamma'] = ds['gamma'].assign_attrs({'unit': 'deg', 'long_name': 'Zenith Angle'})

nds = ds.copy()
nds.values = timg
nds['gamma'] = linangles
nds['gamma'] = nds['gamma'].assign_attrs({'unit': 'deg', 'long_name': 'Zenith Angle'})

# # %%
# plt.plot(imap[0,:,100],label = 'input coords')
# plt.plot(imap[1,:,100], label = 'output coords')
# plt.legend()
# # %%

# plt.plot(imap[0,0,:],label = 'input coords')
# plt.plot(imap[1,0,:], label = 'output coords')
# # plt.legend()
# # %%
# print(np.shape(imap[0]))
# print(np.shape(imap[1]))
# print(np.shape(ds.values))

# # %%
# plt.imshow(imap[0])
# %%
fig, (ax1, ax2) = plt.subplots(1, 2, figsize = (12, 6), dpi=300)
fig.tight_layout()

vmin = np.nanpercentile(ds.values, 1)
vmax = np.nanpercentile(ds.values, 99)
ds.plot(ax = ax1, vmin=vmin, vmax=vmax)
ax1.set_title('Zenith Angle (NL)')


vmin = np.nanpercentile(timg, 1)
vmax = np.nanpercentile(timg, 99)
nds.plot(ax = ax2, vmin=vmin, vmax=vmax)
ax2.set_title('Zenith Angle (Warped Linear)')


# %%
plt.plot(angles)
# %%
