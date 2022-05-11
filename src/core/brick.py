import config as conf
from .image import BaseImage
from .utils import get_brick_position, dilate_and_group, clean_catalog
from .group import Group

import logging
import os
from astropy.wcs import WCS
from astropy.io import fits
from astropy.nddata import Cutout2D
import astropy.units as u
import numpy as np
from scipy.ndimage import zoom



class Brick(BaseImage):
    def __init__(self, brick_id=None, position=None, size=None) -> None:

        # Housekeeping
        self.brick_id = brick_id
        self.bands = []
        self.wcs = {}
        self.pixel_scales = {}
        self.data = {} 
        self.headers = {}
        self.properties = {}
        self.catalogs = {}
        self.type = 'brick'
        self.n_sources = {}
        self.group_ids = {}

        # Load the logger
        self.logger = logging.getLogger(f'farmer.brick_{brick_id}')

        # Position
        if (brick_id is not None) & ((position is not None) | (size is not None)):
            raise RuntimeError('Cannot create brick from BOTH brick_id AND position/size!')
        if brick_id is not None:
            self.position, self.size = get_brick_position(brick_id)
        else:
            self.position, self.size = position, size
        self.buffsize = (self.size[0]+2*conf.BRICK_BUFFER, self.size[1]+2*conf.BRICK_BUFFER)
        self.logger.info(f'Spawned brick #{self.brick_id} at ({self.position.ra:2.1f}, {self.position.dec:2.1f}) with size {self.size[0].to(u.arcmin):2.1f} X {self.size[1].to(u.arcmin):2.1f}')

        self.filename = f'b{brick_id}_{self.position.ra.value:2.3f}_{self.position.dec.value:2.3f}.fits'

    def get_figprefix(self, imgtype, band):
        return f'B{self.brick_id}_{band}_{imgtype}'

    def get_bands(self):
        return np.array(self.bands)

    def summary(self):
        print(f'Summary of brick {self.brick_id}')
        print(f'Located at ({self.position.ra:2.2f}, {self.position.dec:2.2f}) with size {self.size[0]:2.2f} x {self.size[1]:2.2f}')
        print(f'   (w/ buffer: {self.buffsize[0]:2.2f} x {self.buffsize[1]:2.2f})')
        print(f'   (w/ buffer: {self.buffsize[0]:2.2f} x {self.buffsize[1]:2.2f})')
        print(f'Has {len(self.bands)} bands: {self.bands}')
        for band in self.bands:
            print(f' --- Data {band} ---')
            for imgtype in self.data[band].keys():
                print(f'  {imgtype} ... {np.shape(self.data[band][imgtype].data)}')
            # print(f'--- Properties {band} ---')
            for attr in self.properties[band].keys():
                print(f'  {attr} ... {self.properties[band][attr]}')

    def add_band(self, mosaic, overwrite=False):

        if (~overwrite) & (mosaic.band in self.bands):
            raise RuntimeError('{mosaic.band} already exists in brick #{self.brick_id}!')

        # Add band information
        self.data[mosaic.band] = {}
        self.properties[mosaic.band] = {}
        self.headers[mosaic.band] = {}
        self.n_sources[mosaic.band] = {}
        self.catalogs[mosaic.band] = {}
        self.group_ids[mosaic.band] = {}
        self.bands.append(mosaic.band)

        # Loop over properties
        for attr in mosaic.properties.keys():
            self.properties[mosaic.band][attr] = mosaic.properties[attr]
            self.logger.debug(f'... property \"{attr}\" adopted from mosaic')

        # Loop over provided data
        for imgtype in mosaic.data.keys():
            if imgtype in ('science', 'weight', 'mask', 'segmap', 'groupmap', 'background', 'rms', 'model', 'residual'):
                fill_value = np.nan
                if imgtype == 'mask':
                    fill_value = True
                cutout = Cutout2D(mosaic.data[imgtype], self.position, self.buffsize[::-1], wcs=mosaic.wcs,
                                 copy=True, mode='partial', fill_value = fill_value)
                self.logger.debug(f'... data \"{imgtype}\" subimage cut from {mosaic.band} at {cutout.input_position_original}')
                self.data[mosaic.band][imgtype] = cutout
                if imgtype in ('science', 'weight', 'mask'):
                    self.headers[mosaic.band][imgtype] = mosaic.headers[imgtype] #TODO update WCS!
                if imgtype == 'science':
                    self.wcs[mosaic.band] = cutout.wcs
                    self.pixel_scales[mosaic.band] = cutout.wcs.proj_plane_pixel_scales()
                    self.estimate_properties(band=mosaic.band, imgtype=imgtype)
            else:
                self.data[mosaic.band][imgtype] = mosaic.data[imgtype]
                self.logger.debug(f'... data \"{imgtype}\" adopted from mosaic')

        # if weights or masks dont exist, make them as dummy arrays
        weight = np.ones_like(mosaic.data['science']) # big, but OK...
        cutout = Cutout2D(weight, self.position, self.buffsize[::-1], wcs=mosaic.wcs, mode='partial', fill_value = np.nan)
        self.logger.debug(f'... data \"weight\" subimage generated as ones at {cutout.input_position_original}')
        self.data[mosaic.band]['weight'] = cutout
        self.headers[mosaic.band]['weight'] = self.headers[mosaic.band]['science']

        mask = np.zeros_like(mosaic.data['science']).astype(bool) # big, but OK...
        cutout = Cutout2D(mask, self.position, self.buffsize[::-1], wcs=mosaic.wcs, mode='partial', fill_value = True)
        self.logger.debug(f'... data \"mask\" subimage generated as ones at {cutout.input_position_original}')
        self.data[mosaic.band]['mask'] = cutout
        self.headers[mosaic.band]['mask'] = self.headers[mosaic.band]['science']

        background = np.zeros_like(mosaic.data['science']) # big, but OK...
        cutout = Cutout2D(background, self.position, self.buffsize[::-1], wcs=mosaic.wcs, mode='partial', fill_value = np.nan)
        self.logger.debug(f'... data \"background\" subimage generated as ones at {cutout.input_position_original}')
        self.data[mosaic.band]['background'] = cutout
        self.headers[mosaic.band]['background'] = self.headers[mosaic.band]['science']

        rms = np.zeros_like(mosaic.data['science']) # big, but OK...
        cutout = Cutout2D(rms, self.position, self.buffsize[::-1], wcs=mosaic.wcs, mode='partial', fill_value = np.nan)
        self.logger.debug(f'... data \"rms\" subimage generated as ones at {cutout.input_position_original}')
        self.data[mosaic.band]['rms'] = cutout
        self.headers[mosaic.band]['rms'] = self.headers[mosaic.band]['science']


        # get background info if backregion is 'brick' -- WILL overwrite inhereted info if it exists...
        if self.properties[mosaic.band]['backregion'] == 'brick':
            self.estimate_background(band=mosaic.band, imgtype='science')
        
        
        # TODO -- should be able to INHERET catalogs from the parent mosaic, if they exist!


    def write(self, filename=None, directory=conf.PATH_BRICKS, allow_update=False):
        if filename is None:
            filename = self.filename
        self.logger.debug(f'Writing to {filename}! (allow_update = {allow_update})')
        if not filename.endswith('.fits'):
            raise RuntimeError(f'Requested filename {filename} is not a recognized FITS file! (.fits)')
        path = os.path.join(directory, filename)
        if os.path.exists(path):
            if not allow_update:
                raise RuntimeError(f'Cannot update {filename}! (allow_update = False)')
            else:
                # open file and add to it
                hdul = fits.open(path)
        else:
            # make new file
            hdul = fits.HDUList()
            hdul.append(fits.PrimaryHDU())

        for band in self.data.keys():
            for attr in self.data[band].keys():
                if attr == 'psfmodel':
                    continue
                ext_name = f'{attr}_{band}'
                try:
                    hdul[ext_name]
                except:
                    hdul.append(fits.ImageHDU(name=ext_name))
                hdul[ext_name].data = self.data[band][attr].data
                # update WCS in header
                for (key, value, comment) in self.headers[band][attr].cards:
                    hdul[ext_name].header[key] = (value, comment)
                for (key, value, comment) in self.data[band][attr].wcs.to_header().cards:
                    hdul[ext_name].header[key] = (value, comment)
                
                self.logger.debug(f'... added {attr} for {band}')

        try:
            hdul.flush()
            self.logger.info(f'Updated {filename}! (allow_update = {allow_update})')
        except:
            hdul.writeto(path, overwrite=conf.OVERWRITE)
            self.logger.info(f'Wrote to {filename}! (allow_update = {allow_update})')

    def extract(self, band='detection', imgtype='science', background=None):
        catalog, segmap = self._extract(band, imgtype='science', background=background)

        # clean out buffer -- these are bricks!
        self.logger.debug('Removing sources detected in brick buffer...')
        cutout = Cutout2D(self.data[band][imgtype].data, self.position, self.size[::-1], wcs=self.data[band][imgtype].wcs)
        mask = Cutout2D(np.zeros(cutout.data.shape), self.position, self.buffsize[::-1], wcs=cutout.wcs, fill_value=1, mode='partial').data.astype(bool)
        segmap = Cutout2D(segmap, self.position, self.buffsize[::-1], self.wcs[band], fill_value=0, mode='partial')
        catalog, segmap.data = clean_catalog(catalog, mask, segmap=segmap.data)
        mask[mask & (segmap.data>0)] = False

        # save stuff
        self.catalogs[band][imgtype] = catalog
        self.data[band]['segmap'] = segmap
        self.data[band]['weight'].data[mask] = 0 #removes buffer but keeps segment pixels
        self.data[band]['mask'].data[mask] = True # adds buffer to existing mask
        self.n_sources[band][imgtype] = len(catalog)

        # add ids
        self.catalogs[band][imgtype].add_column(self.brick_id * np.ones(self.n_sources[band][imgtype]), name='brick_id', index=0)
        self.catalogs[band][imgtype].add_column(1+np.arange(self.n_sources[band][imgtype]), name='ID', index=0)

        # add world positions
        skycoords = self.data[band][imgtype].wcs.all_pix2world(catalog['x'], catalog['y'], 0)
        self.catalogs[band][imgtype].add_column(skycoords[0]*u.deg, name=f'ra', index=1, )
        self.catalogs[band][imgtype].add_column(skycoords[1]*u.deg, name=f'dec', index=2)


    def identify_groups(self, band='detection', imgtype='science', radius=conf.DILATION_RADIUS):
        """Takes the catalog and segmap 
        """
        catalog = self.catalogs[band][imgtype]
        segmap = self.data[band]['segmap'].data
        radius = radius.to(u.arcsec)
        radius_px = radius / (self.wcs['detection'].pixel_scale_matrix[-1,-1] * u.deg).to(u.arcsec) # this won't be so great for non-aligned images...
        radius_rpx = round(radius_px.value)
        self.logger.debug(f'Dilation radius of {radius} or {radius_px:2.2f} px rounded to {radius_rpx} px')

        group_ids, group_pops, groupmap = dilate_and_group(catalog, segmap, radius=radius_rpx, fill_holes=True)

        self.catalogs[band][imgtype].add_column(group_ids, name='group_id', index=3)
        self.catalogs[band][imgtype].add_column(group_pops, name='group_pop', index=3)
        self.data[band]['groupmap'] = Cutout2D(groupmap, self.position, self.buffsize[::-1], self.wcs[band], mode='partial', fill_value = 0)
        self.group_ids[band][imgtype] = np.unique(group_ids)

    def spawn_group(self, group_id=None, imgtype='science', bands=None):
        # Instantiate brick
        group = Group(group_id, self, imgtype=imgtype)
        
        # Cut up science, weight, and mask, if available
        group.add_bands(self, bands=bands)

        # Return it
        return group

    def detect_sources(self, band='detection', imgtype='science'):

        # detection
        self.extract(band=band, imgtype=imgtype)

        # grouping
        self.identify_groups(band=band, imgtype=imgtype)

        return self.catalogs[band][imgtype]

            
