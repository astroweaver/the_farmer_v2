from pickle import FALSE
from re import L
from typing import OrderedDict
import config as conf
from .utils import clean_catalog, reproject_discontinuous, SimpleGalaxy, read_wcs
from .group import Group

import logging
import os
import sep
import numpy as np
import time

import matplotlib.pyplot as plt
from  matplotlib.colors import LogNorm
from mpl_toolkits.axes_grid1 import make_axes_locatable
from astropy.stats import sigma_clipped_stats
from astropy.nddata import Cutout2D
from astropy.io import ascii
from astropy.table import Table
import astropy.units as u
from astropy.coordinates import SkyCoord
from matplotlib.patches import Rectangle
from tractor import Catalog, RaDecPos, WcslibWcs
from tqdm import tqdm

from tractor import NCircularGaussianPSF, PixelizedPSF, PixelizedPsfEx, Image, Tractor, FluxesPhotoCal, NullWCS, ConstantSky, EllipseE, EllipseESoft, Fluxes, PixPos, Catalog
from tractor.sersic import SersicIndex, SersicGalaxy
from tractor.sercore import SersicCoreGalaxy
from tractor.galaxy import ExpGalaxy, DevGalaxy, FixedCompositeGalaxy, SoftenedFracDev
from tractor.pointsource import PointSource
from tractor.psf import HybridPixelizedPSF


class BaseImage():
    """
    Base class for images: mosaics, bricks, and groups
    Useful for estimating image properties + detecting sources
    """

    def __init__(self):
        self.data = {}
        self.headers = {}
        self.segments = {}
        self.catalogs = {}
        self.bands = []

        # Load the logger
        self.logger = logging.getLogger(f'farmer.image')

   
    def get_image(self, imgtype=None, band=None):

        if self.type == 'mosaic':
            return self.data[imgtype]

        else:
            return self.data[band][imgtype].data

    def set_image(self, image, imgtype=None, band=None):
        if self.type == 'mosaic':
            self.data[imgtype] = image

        else:
            if imgtype in self.data[band]:
                self.data[band][imgtype].data = image
            else:
                self.data[band][imgtype] = Cutout2D(np.zeros_like(self.data[band]['science'].data), self.position, self.size, wcs=self.wcs[band])
                self.data[band][imgtype].data = image


    def set_property(self, value, property, band=None):
        if self.type == 'mosaic':
            self.properties[property] = value

        else:
            self.properties[band][property] = value

    def get_property(self, property, band=None):
        if self.type == 'mosaic':
            return self.properties[property]
        else:
            return self.properties[band][property]

    def get_wcs(self, band=None, imgtype='science'):

        if self.type == 'mosaic':
            return self.wcs

        else:
            return self.wcs[band]


    def estimate_background(self, band=None, imgtype='science'):
        
        image = self.get_image(imgtype, band)
        if image.dtype.byteorder == '>':
                image = image.byteswap().newbyteorder()
        self.logger.debug(f'Estimating background...')
        background = sep.Background(image, 
                                bw = conf.DETECT_BW, bh = conf.DETECT_BH,
                                fw = conf.DETECT_FW, fh = conf.DETECT_FH)

        self.set_image(background.back(), imgtype='background', band=band)
        self.set_image(background.rms(), imgtype='rms', band=band)
        self.set_property(background.globalrms, 'rms', band)
        self.set_property(background.globalback, 'background', band)

        return background


    def _extract(self, band='detection', imgtype='science', wgttype='weight', masktype='mask', background=None):
        var = None
        mask = None
        image = self.get_image(imgtype, band) # these are cutouts, remember.
        if conf.USE_DETECTION_WEIGHT:
            try:
                wgt = self.data[band][wgttype].data
                var = np.where(wgt>0, 1/np.sqrt(wgt), 0)
            except:
                raise RuntimeError(f'Weight not found!')
        if conf.USE_DETECTION_MASK:
            try:
                mask = self.data[band][masktype].data
            except:
                raise RuntimeError(f'Mask not found!')

        # Deal with background
        if background is None:
            background = 0
        else:
            assert(np.shape(background)==np.shape(image), f'Background ({np.shape(background)}) does not have the same shape as image ({np.shape(image)}!')

        # Grab the convolution filter
        convfilt = None
        if conf.FILTER_KERNEL is not None:
            dirname = os.path.dirname(__file__)
            filename = os.path.join(dirname, '../../config/conv_filters/'+conf.FILTER_KERNEL)
            if os.path.exists(filename):
                convfilt = np.array(np.array(ascii.read(filename, data_start=1)).tolist())
            else:
                raise FileExistsError(f"Convolution file at {filename} does not exist!")

        # Do the detection
        self.logger.debug(f'Detection will be performed with thresh = {conf.THRESH}')
        kwargs = dict(var=var, mask=mask, minarea=conf.MINAREA, filter_kernel=convfilt, 
                filter_type=conf.FILTER_TYPE, segmentation_map=True, 
                clean = conf.CLEAN, clean_param = conf.CLEAN_PARAM,
                deblend_nthresh=conf.DEBLEND_NTHRESH, deblend_cont=conf.DEBLEND_CONT)
        tstart = time.time()
        catalog, segmap = sep.extract(image-background, conf.THRESH, **kwargs)
        self.logger.debug(f'Detection found {len(catalog)} sources. ({time.time()-tstart:2.2}s)')
        catalog = Table(catalog)

        # Apply mask now?
        if conf.APPLY_DETECTION_MASK & (mask is not None):
            catalog, segmap = clean_catalog(catalog, mask, segmap)
        elif conf.APPLY_DETECTION_MASK & (mask is None):
            raise RuntimeError('Cannot apply detection mask when there is no mask supplied!')

        return catalog, segmap

    def estimate_properties(self, band=None, imgtype='science'):
        try:
            self.get_property('mean', band)
            self.get_property('median', band)
            self.get_property('clipped_rms', band)
            return mean, median, rms
        except:
            image = self.get_image(imgtype, band)
            mean, median, rms = sigma_clipped_stats(image)
            self.logger.debug(f'Estimated stats of \"{imgtype}\" image (@3 sig)')
            self.logger.debug(f'    Mean:   {mean:2.3f}')
            self.logger.debug(f'    Median: {median:2.3f}')
            self.logger.debug(f'    RMS:    {rms:2.3f}')

            self.set_property(mean, 'mean', band)
            self.set_property(median, 'median', band)
            self.set_property(rms, 'clipped_rms', band)
            return mean, median, rms

    def generate_weight(self, band=None, imgtype='science', overwrite=False):
        """Uses rms from image to estimate inverse variance weights
        """
        try:
            rms = self.get_property(band)
        except:
            __, __, rms = self.estimate_properties(band, imgtype)
        image = self.get_image(imgtype, band)
        if ('weight' in self.data[band].keys()) & ~overwrite:
            raise RuntimeError('Cannot overwrite exiting weight (overwrite=False)')
        weight = np.ones_like(image) * np.where(rms>0, 1/rms**2, 0)
        self.set_image(weight, 'weight', band)

        return weight

    def generate_mask(self, band=None, imgtype='weight', overwrite=False):
        """Uses zero weight portions to make an effective mask
        """

        image = self.get_image(imgtype, band)
        if ('weight' in self.data[band].keys()) & ~overwrite:
            raise RuntimeError('Cannot overwrite exiting weight (overwrite=False)')
        mask = image == 0
        self.set_image(mask, 'mask', band)

        return mask

    def get_background(self, band=None):
        if self.get_property('backtype', band=band) == 'flat':
            return self.get_property('background', band=band)
        elif self.get_property('backtype', band=band) == 'variable':
            return self.get_image(band=band, imgtype='background')


    def stage_images(self, bands=conf.MODEL_BANDS, data_imgtype='science'):
        if bands is None:
            bands = self.get_bands()
        elif np.isscalar(bands):
            bands = [bands,]

        self.images = OrderedDict()

        self.logger.debug(f'Staging images for The Tractor... (image --> {data_imgtype})')
        for band in bands:
            psfmodel = PixelizedPSF(self.data[band]['psfmodel'])

            self.images[band] = Image(
                data=self.get_image(band=band, imgtype=data_imgtype),
                invvar=self.get_image(band=band, imgtype='weight'),
                psf=psfmodel,
                wcs=read_wcs(self.wcs[band]),
                photocal=FluxesPhotoCal(band),
                sky=ConstantSky(0)
            )
            self.logger.debug(f'  ✓ {band}')

    
    def stage_models(self, bands=conf.MODEL_BANDS, data_imgtype='science', catalog_band='detection', catalog_imgtype='science'):
        """ Build the Tractor Model catalog for the group """

        if bands is None:
            bands = self.get_bands()
        elif np.isscalar(bands):
            bands = [bands,]

        if 'detection' in bands:
            bands.remove('detection')

        # Trackers
        self.logger.debug(f'Loading models for group #{self.group_id}')
        self.model_catalog = OrderedDict()

        for src in self.catalogs[catalog_band][catalog_imgtype]:

            source_id = src['ID']
            self.model_catalog[source_id] = PointSource(None, None)

            self.logger.debug(f"Source #{source_id}")
            self.logger.debug(f"  x, y: {src['x']:3.3f}, {src['y']:3.3f}")
            self.logger.debug(f"  flux: {src['flux']:3.3f}")
            self.logger.debug(f"  cflux: {src['cflux']:3.3f}")
            self.logger.debug(f"  a, b: {src['a']:3.3f}, {src['b']:3.3f}") 
            self.logger.debug(f"  theta: {src['theta']:3.3f}")

            # inital position
            position = RaDecPos(src['ra'], src['dec'])

            # initial fluxes
            qflux = np.zeros(len(bands))
            for j, band in enumerate(bands):
                src_seg = self.data[band]['segmap'].data==source_id
                qflux[j] = np.sum(self.images[band].data * src_seg)
            flux = Fluxes(**dict(zip(bands, qflux)), order=bands)

            # initial shapes
            pa = 90 + np.rad2deg(src['theta'])
            shape = EllipseESoft.fromRAbPhi(src['a'], src['b'] / src['a'], pa)
            nre = SersicIndex(2.5) # Just a guess for the seric index
            fluxcore = Fluxes(**dict(zip(bands, np.zeros(len(bands)))), order=bands) # Just a simple init condition

            if isinstance(self.model_catalog[source_id], PointSource):
                self.model_catalog[source_id] = PointSource(position, flux)
                self.model_catalog[source_id].name = 'PointSource' # HACK to get around Dustin's HACK.
            elif isinstance(self.model_catalog[source_id], SimpleGalaxy):
                self.model_catalog[source_id] = SimpleGalaxy(position, flux)
            elif isinstance(self.model_catalog[source_id], ExpGalaxy):
                self.model_catalog[source_id] = ExpGalaxy(position, flux, shape)
            elif isinstance(self.model_catalog[source_id], DevGalaxy):
                self.model_catalog[source_id] = DevGalaxy(position, flux, shape)
            elif isinstance(self.model_catalog[source_id], FixedCompositeGalaxy):
                self.model_catalog[source_id] = FixedCompositeGalaxy(
                                                position, flux,
                                                SoftenedFracDev(0.5),
                                                shape, shape)
            elif isinstance(self.model_catalog[source_id], SersicGalaxy):
                self.model_catalog[source_id] = SersicGalaxy(position, flux, shape, nre)
            elif isinstance(self.model_catalog[source_id], SersicCoreGalaxy):
                self.model_catalog[source_id] = SersicCoreGalaxy(position, flux, shape, nre, fluxcore)

            self.logger.debug(f'Source #{source_id}: {self.model_catalog[source_id].name} model at {position}')
            self.logger.debug(f'               {flux}') 
            if hasattr(self.model_catalog[source_id], 'fluxCore'):
                self.logger.debug(f'               {fluxcore}')
            if hasattr(self.model_catalog[source_id], 'shape'):
                self.logger.debug(f'               {shape}')

    def optimize(self):
        for i in range(conf.MAX_STEPS):
            dlnp, X, alpha = self.engine.optimize()
            print('dlnp', dlnp)
            if dlnp < 1e-3:
                break

        return i, dlnp, X, alpha

    def stage_engine(self, bands=conf.MODEL_BANDS):
        self.stage_images(bands=bands)
        self.stage_models(bands=bands)
        self.engine = Tractor(list(self.images.values()), list(self.model_catalog.values()))
        self.engine.bands = list(self.images.keys())
        self.engine.freezeParam('images')

    def build_all_images(self, bands=None, overwrite=True):
        if bands is None:
            bands = self.engine.bands
        elif np.isscalar(bands):
            bands = [bands,]

        self.build_model_image(bands, overwrite)
        self.build_residual_image(bands, overwrite)
        self.build_chi_image(bands, overwrite)

    def build_model_image(self, bands=None, overwrite=True):
        if bands is None:
            bands = self.engine.bands
        elif np.isscalar(bands):
            bands = [bands,]

        for band in bands:
            model = self.engine.getModelImage(self.images[band])
            self.set_image(model, 'model', band)
            self.logger.debug(f'Built model image for {band}')
        
    def build_residual_image(self, bands=None, imgtype='science', overwrite=True):
        if bands is None:
            bands = self.engine.bands
        elif np.isscalar(bands):
            bands = [bands,]

        for band in bands:
            model = self.get_image('model', band)
            self.set_image(self.get_image('science', band) - model, 'residual', band)
            self.logger.debug(f'Built residual image for {band}')

    def build_chi_image(self, bands=None, imgtype='science', overwrite=True):
        if bands is None:
            bands = self.engine.bands
        elif np.isscalar(bands):
            bands = [bands,]

        for band in bands:
            chi = self.get_image('residual', band) * np.sqrt(self.get_image('weight', band))
            self.set_image(chi, 'chi', band)
            self.logger.debug(f'Built chi image for {band}')

    def plot_image(self, band=None, imgtype=None, tag='', overwrite=True, show_catalog=True, catalog_band='detection', catalog_imgtype='science', show_groups=True):
        # for each band, plot all available images: science, weight, mask, segmap, blobmap, background, rms
        if band is None:
            bands = self.get_bands()
        elif np.isscalar(band):
            bands = [band,]
        else:
            bands = band

        if self.type == 'group':
            show_groups = False

        if (tag != '') & (not tag.startswith('_')):
            tag = '_' + tag

        in_imgtype = imgtype

        for band in bands:
            if in_imgtype is None:
                if self.type == 'mosaic':
                    imgtypes = self.data.keys()
                else:
                    imgtypes = self.data[band].keys()
            elif np.isscalar(in_imgtype):
                imgtypes = [in_imgtype,]
            else:
                imgtypes = in_imgtype

            
            for imgtype in imgtypes:
                if imgtype == 'psfmodel':
                    continue

                if self.type == 'mosaic':
                    if imgtype not in self.data.keys():
                        continue
                else:
                    if imgtype not in self.data[band].keys():
                        continue

                self.logger.debug(f'Gathering image: {band} {imgtype}')
                image = self.get_image(band=band, imgtype=imgtype)

                background = 0
                if (imgtype in ('science',)) & self.get_property('subtract_background', band=band):
                    background = self.get_background(band)

                if imgtype in ('science', 'model', 'residual'):
                    # log-scaled
                    fig = plt.figure(figsize=(10,10),)
                    ax = fig.add_subplot(projection=self.get_wcs(band))
                    vmin, vmax = np.max([self.get_property('median', band=band) + self.get_property('rms', band=band), 1E-5]), np.nanmax(image)
                    if vmin >= vmax:
                        vmax = vmin
                    norm = LogNorm(vmin, vmax, clip='True')
                    options = dict(cmap='Greys', norm=norm)
                    im = ax.imshow(image - background, **options)
                    fig.colorbar(im, orientation="horizontal", pad=0.2)
                    pixscl = self.pixel_scales[band][0].value, self.pixel_scales[band][0].value
                    brick_buffer_pix = conf.BRICK_BUFFER.to(u.deg).value * pixscl[0], conf.BRICK_BUFFER.to(u.deg).value * pixscl[1]
                    ax.add_patch(Rectangle(brick_buffer_pix, self.size[0].value, self.size[1].value,
                                     fill=False, alpha=0.3, edgecolor='purple', linewidth=1))
                    # show centroids
                    if show_catalog & (catalog_band in self.catalogs.keys()):
                        if catalog_imgtype in self.catalogs[catalog_band].keys():
                            coords = SkyCoord(self.catalogs[catalog_band][catalog_imgtype]['ra'], self.catalogs[catalog_band][catalog_imgtype]['dec'])
                            pos = self.wcs[band].world_to_pixel(coords)
                            ax.scatter(pos[0], pos[1], 20, marker='o', edgecolors='g', lw=1, facecolors='none', alpha=0.2)
                    # show group extents
                    if show_groups:
                        # groupmap = self.get_image(band=catalog_band, imgtype='groupmap')
                        for gid in tqdm(self.group_ids[catalog_band][catalog_imgtype]):
                            try:
                                group = Group(gid, self)
                            except:
                                continue
                            idy, idx = group.position.to_pixel(self.wcs[band])
                            dx, dy = group.size[0].to(u.deg).value / pixscl[0], group.size[1].to(u.deg).value / pixscl[1]
                            xlo = idx - dx / 2
                            ylo = idy - dy / 2
                            bdx = dx + 2 * conf.GROUP_BUFFER.to(u.deg).value * pixscl[0]
                            bdy = dy + 2 * conf.GROUP_BUFFER.to(u.deg).value * pixscl[1]
                            rect = Rectangle((ylo, xlo), bdy, bdx, fill=False, alpha=0.3,
                                                    edgecolor='red', zorder=3, linewidth=1)
                            ax.add_patch(rect)
                            ax.annotate(str(gid), (ylo, xlo), color='g', fontsize=2)

                    fig.tight_layout()
                    filename = self.get_figprefix(imgtype, band) + f'_log10{tag}.pdf'
                    fig.savefig(os.path.join(conf.PATH_FIGURES, filename), overwrite=overwrite, dpi=300)
                    self.logger.debug(f'Saving figure: {filename}')                
                    plt.close(fig)

                    # lin-scaled
                    fig = plt.figure(figsize=(10,10),)
                    ax = fig.add_subplot(projection=self.get_wcs(band))
                    options = dict(cmap='RdGy', vmin=-5*self.get_property('rms', band=band), vmax=5*self.get_property('rms', band=band))
                    im = ax.imshow(image - background, **options)
                    fig.colorbar(im, orientation="horizontal", pad=0.2)
                    ax.add_patch(Rectangle(brick_buffer_pix, self.size[0].value, self.size[1].value,
                                     fill=False, alpha=0.3, edgecolor='purple', linewidth=1))
                    if show_catalog & (catalog_band in self.catalogs.keys()):
                        if catalog_imgtype in self.catalogs[catalog_band].keys():
                            coords = SkyCoord(self.catalogs[catalog_band][catalog_imgtype]['ra'], self.catalogs[catalog_band][catalog_imgtype]['dec'])
                            pos = self.wcs[band].world_to_pixel(coords)
                            ax.scatter(pos[0], pos[1], 20, marker='o', edgecolors='g', lw=1, facecolors='none', alpha=0.2)
                    filename = self.get_figprefix(imgtype, band) + f'_rms{tag}.pdf'
                    fig.tight_layout()
                    fig.savefig(os.path.join(conf.PATH_FIGURES, filename), overwrite=overwrite, dpi=300)
                    self.logger.debug(f'Saving figure: {filename}')                
                    plt.close(fig)

                if imgtype in ('chi'):
                    fig = plt.figure(figsize=(10,10),)
                    ax = fig.add_subplot(projection=self.get_wcs(band))
                    options = dict(cmap='RdGy', vmin=-3, vmax=3)
                    im = ax.imshow(image, **options)
                    fig.colorbar(im, orientation="horizontal", pad=0.2)
                    fig.tight_layout()
                    filename = self.get_figprefix(imgtype, band) + '{tag}.pdf'
                    fig.savefig(os.path.join(conf.PATH_FIGURES, filename), overwrite=overwrite, dpi=300)
                    self.logger.debug(f'Saving figure: {filename}')                
                    plt.close(fig)
                
                if imgtype in ('weight', 'mask'):
                    fig = plt.figure(figsize=(10,10),)
                    ax = fig.add_subplot(projection=self.get_wcs(band))
                    options = dict(cmap='Greys', vmin=np.min(image), vmax=np.max(image))
                    im = ax.imshow(image, **options)
                    fig.colorbar(im, orientation="horizontal", pad=0.2)
                    fig.tight_layout()
                    filename = self.get_figprefix(imgtype, band) + '{tag}.pdf'
                    fig.savefig(os.path.join(conf.PATH_FIGURES, filename), overwrite=overwrite, dpi=300)
                    self.logger.debug(f'Saving figure: {filename}')                
                    plt.close(fig)
            
                if imgtype in ('segmap', 'groupmap'):
                    fig = plt.figure(figsize=(10,10),)
                    ax = fig.add_subplot(projection=self.get_wcs(band))
                    options = dict(cmap='prism', vmin=np.min(image[image!=0]), vmax=np.max(image))
                    image = image.copy().astype('float')
                    image[image==0] = np.nan
                    im = ax.imshow(image, **options)
                    fig.colorbar(im, orientation="horizontal", pad=0.2)
                    fig.tight_layout()
                    filename = self.get_figprefix(imgtype, band) + '{tag}.pdf'
                    fig.savefig(os.path.join(conf.PATH_FIGURES, filename), overwrite=overwrite, dpi=300)
                    self.logger.debug(f'Saving figure: {filename}')                
                    plt.close(fig)

    def plot_psf(self, band=None, overwrite=True):
          
        if band is None:
            bands = self.get_bands()
        else:
            bands = [band,]

        for band in bands:
            self.logger.debug(f'Plotting PSF for: {band}')    
            if 'psfmodel' not in self.data[band]:
                self.logger.warning(f'Cannot find psfmodel for {band}!')
                continue
            psfmodel = self.data[band]['psfmodel']

            pixscl = (self.pixel_scales[band][0]).to(u.arcsec).value
            fig, ax = plt.subplots(ncols=3, figsize=(30,10))
            norm = LogNorm(1e-8, 0.1*np.nanmax(psfmodel), clip='True')
            img_opt = dict(cmap='Blues', norm=norm)
            ax[0].imshow(psfmodel, **img_opt, extent=pixscl *np.array([-np.shape(psfmodel)[0]/2,  np.shape(psfmodel)[0]/2, -np.shape(psfmodel)[0]/2,  np.shape(psfmodel)[0]/2,]))
            ax[0].set(xlim=(-15,15), ylim=(-15, 15))
            ax[0].axvline(0, color='w', ls='dotted')
            ax[0].axhline(0, color='w', ls='dotted')

            xax = np.arange(-np.shape(psfmodel)[0]/2 + 0.5,  np.shape(psfmodel)[0]/2+0.5)
            [ax[1].plot(xax * pixscl, psfmodel[x], c='royalblue', alpha=0.5) for x in np.arange(0, np.shape(psfmodel)[1])]
            ax[1].axvline(0, ls='dotted', c='k')
            ax[1].set(xlim=(-15, 15), yscale='log', ylim=(1E-6, 1E-1), xlabel='arcsec')

            x = xax
            y = x.copy()
            xv, yv = np.meshgrid(x, y)
            radius = np.sqrt(xv**2 + xv**2)
            cumcurve = [np.sum(psfmodel[radius<i]) for i in np.arange(0, np.shape(psfmodel)[0]/2)]
            ax[2].plot(np.arange(0, np.shape(psfmodel)[0]/2) * pixscl, cumcurve)

            fig.suptitle(band)

            figname = os.path.join(conf.PATH_FIGURES, f'{band}_psf.pdf')
            self.logger.debug(f'Saving figure: {figname}')                
            fig.savefig(figname, overwrite=overwrite)
            plt.close(fig)

    def transfer_maps(self, catalog_band='detection'):
        # rescale segmaps and groupmaps to other bands
        segmap = self.data[catalog_band]['segmap']
        groupmap = self.data[catalog_band]['groupmap']
        catalog_pixscl = np.array([self.pixel_scales[catalog_band][0].value, self.pixel_scales[catalog_band][1].value])

        # loop over bands
        for band in self.bands:
            if band == 'detection':
                continue
            self.logger.debug(f'Rescaling maps of {band}...')
            pixscl = np.array([self.pixel_scales[band][0].value, self.pixel_scales[band][1].value])
            scale_factor = catalog_pixscl / pixscl

            if np.all(scale_factor==1):
                self.data[band]['segmap'] = self.data[catalog_band]['segmap']
                self.data[band]['groupmap'] = self.data[catalog_band]['groupmap']
                self.logger.debug(f'Copied maps of {catalog_band} to {band}')
            else:
                self.data[band]['segmap'] = Cutout2D(np.zeros_like(self.data[band]['science'].data), self.position, self.buffsize, self.wcs[band], mode='partial', fill_value = 0)
                self.data[band]['segmap'].data = reproject_discontinuous((segmap.data, segmap.wcs), self.wcs[band], np.shape(self.data[band]['segmap'].data))
                self.data[band]['groupmap'] = Cutout2D(np.zeros_like(self.data[band]['science'].data), self.position, self.buffsize, self.wcs[band], mode='partial', fill_value = 0)
                self.data[band]['groupmap'].data = reproject_discontinuous((groupmap.data, groupmap.wcs), self.wcs[band], np.shape(self.data[band]['segmap'].data))
                self.logger.debug(f'Rescaled maps of {catalog_band} to {band} by {scale_factor} ({pixscl} --> {self.pixel_scales[band]})')

            # Clean up
            if self.type == 'group':
                ingroup = self.data[band]['groupmap'].data == self.group_id
                self.data[band]['mask'].data[~ingroup] = True
                self.data[band]['weight'].data[~ingroup] = 0
                self.data[band]['segmap'].data[~ingroup] = 0
                self.data[band]['groupmap'].data[~ingroup] = 0