import config as conf
import os
import logging

import numpy as np
from astropy.io import fits
from astropy.coordinates import SkyCoord
from scipy.ndimage import label, binary_dilation, binary_erosion, binary_fill_holes
import astropy.units as u
from astropy.wcs import WCS

from tractor.psfex import PixelizedPsfEx, PixelizedPSF #PsfExModel
# from tractor.psf import HybridPixelizedPSF
from tractor.galaxy import ExpGalaxy
from tractor import EllipseE
from astrometry.util.util import Tan
from tractor import ConstantFitsWcs

import time
from reproject import reproject_interp
from tqdm import tqdm


def start_logger():
    print('Starting up logging system...')

    # Start the logging
    import logging.config
    logger = logging.getLogger('farmer')

    if not len(logger.handlers):
        if conf.LOGFILE_LOGGING_LEVEL is not None:
            logging_level = logging.getLevelName(conf.LOGFILE_LOGGING_LEVEL)
        else:
            logging_level = logging.DEBUG
        logger.setLevel(logging_level)
        logger.propagate = False
        formatter = logging.Formatter('[%(asctime)s] %(name)s :: %(levelname)s - %(message)s', '%H:%M:%S')

        # Logging to the console at logging level
        ch = logging.StreamHandler()
        ch.setLevel(logging.getLevelName(conf.CONSOLE_LOGGING_LEVEL))
        ch.setFormatter(formatter)
        logger.addHandler(ch)

        if (conf.LOGFILE_LOGGING_LEVEL is None) | (not os.path.exists(conf.PATH_LOGS)):
            print('Logging information wills stream only to console.\n')
            
        else:
            # create file handler which logs even debug messages
            logging_path = os.path.join(conf.PATH_LOGS, 'logfile.log')
            print(f'Logging information will stream to console and {logging_path}\n')
            # If overwrite is on, remove old logger
            if conf.OVERWRITE & os.path.exists(logging_path):
                print('WARNING -- Existing logfile will be overwritten.')
                os.remove(logging_path)

            fh = logging.FileHandler(logging_path)
            fh.setLevel(logging.getLevelName(conf.LOGFILE_LOGGING_LEVEL))
            fh.setFormatter(formatter)
            logger.addHandler(fh)

    return logger

def read_wcs(wcs):
    t = Tan()
    crpix = wcs.wcs.crpix
    crval = wcs.wcs.crval
    t.set_crpix(wcs.wcs.crpix[0], wcs.wcs.crpix[1])
    t.set_crval(wcs.wcs.crval[0], wcs.wcs.crval[1])
    cd = wcs.wcs.cd
    # assume your images have no rotation...
    t.set_cd(cd[0,0], cd[0,1], cd[1,0], cd[1,1])
    t.set_imagesize(wcs.array_shape[0], wcs.array_shape[1])
    wcs = ConstantFitsWcs(t)
    return wcs


def get_brick_position(brick_id):
    logger = logging.getLogger('farmer.get_brick_position')
    # Do this relative to the detection image
    wcs = WCS(fits.getheader(conf.DETECTION['science']))
    nx, ny = wcs.array_shape
    brick_width = nx / conf.N_BRICKS[0]
    brick_height = ny / conf.N_BRICKS[1]
    if brick_id > (nx * ny):
        raise RuntimeError(f'Cannot request brick #{brick_id} on grid {nx} X {ny}!')
    logger.debug(f'Using bricks of size ({brick_width:2.2f}, {brick_height:2.2f}) px, in grid {nx} X {ny} px')
    xc = 0.5 * brick_width + int(((brick_id - 1) * brick_height) / nx) * brick_width
    yc = 0.5 * brick_height + int(((brick_id - 1) * brick_height) % ny)
    logger.debug(f'Brick #{brick_id} found at ({xc:2.2f}, {yc:2.2f}) px with size {brick_width:2.2f} X {brick_height:2.2f} px')
    position = wcs.pixel_to_world(xc, yc)
    upper = wcs.pixel_to_world(xc+brick_width/2., yc+brick_height/2.)
    lower = wcs.pixel_to_world(xc-brick_width/2., yc-brick_height/2.)
    size = (lower.ra - upper.ra), (upper.dec - lower.dec)

    logger.debug(f'Brick #{brick_id} found at ({position.ra:2.1f}, {position.dec:2.1f}) with size {size[0]:2.1f} X {size[1]:2.1f}')
    return position, size

def clean_catalog(catalog, mask, segmap=None):
    logger = logging.getLogger('farmer.clean_catalog')
    if segmap is not None:
        assert(mask.shape == segmap.shape, f'Mask {mask.shape} is not the same shape as the segmentation map {segmap.shape}!')
    zero_seg = np.sum(segmap==0)
    logger.debug('Cleaning catalog using mask provided')
    tstart = time.time()

    # map the pixel coordinates to the map
    x, y = np.round(catalog['x']).astype(int), np.round(catalog['y']).astype(int)
    keep = ~mask[y, x]
    segmap[np.isin(segmap, np.argwhere(~keep)+1)] = 0
    cleancat = catalog[keep]

    # relabel segmentation map
    uniques = np.unique(segmap)
    uniques = uniques[uniques>0]
    ids = 1 + np.arange(len(cleancat))
    for (id, uni) in zip(ids, uniques):
        segmap[segmap == uni] = id


    pc = (np.sum(segmap==0) - zero_seg) / np.size(segmap)
    logger.debug(f'Cleaned {np.sum(~keep)} sources ({pc*100:2.2f}% by area), {np.sum(keep)} remain. ({time.time()-tstart:2.2f}s)')
    if segmap is not None:
        return cleancat, segmap
    else:
        return cleancat

def dilate_and_group(catalog, segmap, radius=0, fill_holes=False):
    logger = logging.getLogger('farmer.identify_groups')
    """Takes the catalog and segmap and performs a dilation + grouping. ASSUMES RADIUS IN PIXELS!
    """

    # segmask
    segmask = np.where(segmap==0, 0, segmap)

    # dilation
    if (radius is not None) & (radius > 0):
        logger.debug(f'Dilating segments with radius of {radius:2.2f} px')
        struct2 = create_circular_mask(2*radius, 2*radius, radius=radius)
        segmask = binary_dilation(segmask, structure=struct2).astype(int)

    if fill_holes:
        logger.debug(f'Filling holes...')
        segmask = binary_fill_holes(segmask).astype(int)

    # relabel
    groupmap, n_groups = label(segmask)
    logger.debug(f'Found {np.max(groupmap)} groups for {np.max(segmap)} sources.')
    x, y = np.round(catalog['x']).astype(int), np.round(catalog['y']).astype(int)
    group_ids = groupmap[y, x]

    group_pops = -99 * np.ones(len(catalog))
    for i, group_id in enumerate(group_ids):
        group_pops[i] =  np.sum(group_ids == group_id)  # np.unique with indices might be faster.
    
    for i in np.arange(1, 5):
        ngroup = np.sum(group_pops==i)
        pc = ngroup / n_groups
        logger.debug(f'... N  = {i}: {ngroup} ({pc*100:2.2f}%) ')
    ngroup = np.sum(group_pops>=5)
    pc = ngroup / n_groups
    logger.debug(f'... N >= {5}: {ngroup} ({pc*100:2.2f}%) ')

    return group_ids, group_pops, groupmap


def verify_psfmodel(band):
    logger = logging.getLogger('farmer.verify_psfmodel')
    psfmodel_type = conf.BANDS[band]['psfmodel_type']
    psfmodel_path = conf.BANDS[band]['psfmodel']
    if psfmodel_type == 'PSFGRID':
        # open up gridpnt file
        if os.path.exists(psfmodel_path):
            if os.path.exists(psfmodel_path):
                psftab_grid = ascii.read(psfmodel_path)
                psftab_ra = psftab_grid['RA']
                psftab_dec = psftab_grid['Dec']
                psfcoords = SkyCoord(ra=psftab_ra*u.degree, dec=psftab_dec*u.degree)
                psffname = psftab_grid['FILE_ID']
                psfmodel = (psfcoords, psffname)
                logger.debug(f'Adopted PSFGRID PSF.') 
            else:
                raise RuntimeError(f'{band} is in PRFGRID but does NOT have an gridpoint file!')
        else:
            raise RuntimeError(f'{band} is in PRFGRID but does NOT have an output directory!')

    elif psfmodel_type == 'PRFMAP':
        try:        
            # read in prfmap table
            prftab = ascii.read(psfmodel_path)
            prftab_ra = prftab[conf.PRFMAP_COLUMNS[1]]
            prftab_dec = prftab[conf.PRFMAP_COLUMNS[2]]
            prfcoords = SkyCoord(ra=prftab_ra*u.degree, dec=prftab_dec*u.degree)
            prfidx = prftab[conf.PRFMAP_COLUMNS[0]]
            psfmodel = (prfcoords, prfidx)
            logger.debug(f'Adopted PRFMap PSF.') 
        except:
            raise RuntimeError(f'{band} is in PRFMAP_PS but does NOT have a PRFMAP grid filename!')

      
    elif psfmodel_type == 'constant':
        if psfmodel_path.endswith('.psf'):
            try:
                psfmodel = PixelizedPsfEx(fn=psfmodel_path)
                logger.debug(f'PSF model for {band} adopted as PixelizedPsfEx.')

            except:
                img = fits.open(psfmodel_path)[0].data
                img = img.astype('float32')
                img[img<=0.] = 1E-31
                psfmodel = PixelizedPSF(img)
                logger.debug(f'PSF model for {band} adopted as PixelizedPSF.')
    
        elif psfmodel_path.endswith('.fits'):
            img = fits.open(psfmodel_path)[0].data
            img = img.astype('float32')
            img[img<=0.] = 1E-31
            psfmodel = PixelizedPSF(img)
            logger.debug(f'PSF model for {band} adopted as PixelizedPSF.')

    else:
        if psfmodel_type == 'gaussian':
            psfmodel = None
            logger.warning(f'PSF model not found for {band} -- using {conf.PSF_SIGMA}" gaussian!')
        else:
            raise ValueError(f'PSF model not found for {band}!')

    # normalize it
    psfmodel.img /= np.sum(psfmodel.img)


    return psfmodel


def header_from_dict(params):
    logger = logging.getLogger('farmer.header_from_dict')
    """ Take in dictionary and churn out a header. Never forget configs again. """
    hdr = fits.Header()
    total_public_entries = np.sum([ not k.startswith('__') for k in params.keys()])
    logger.debug(f'header_from_dict :: Dictionary has {total_public_entries} entires')
    tstart = time()
    for i, attr in enumerate(params.keys()):
        if not attr.startswith('__'):
            logger.debug(f'header_from_dict ::   {attr}')
            value = params[attr]
            if type(value) == str:
                # store normally
                hdr.set(f'CONF{i+1}', value, attr)
            if type(value) in (float, int):
                # store normally
                hdr.set(f'CONF{i+1}', value, attr)
            if type(value) in (list, tuple):
                # freak out.
                for j, val in enumerate(value):
                    hdr.set(f'CONF{i+1}_{j+1}', str(val), f'{attr}_{j+1}')
            
    logger.debug(f'header_from_dict :: Completed writing header ({time() - tstart:2.3f}s)')
    return hdr


def create_circular_mask(h, w, center=None, radius=None):

    if center is None: # use the middle of the image
        center = [int(w/2), int(h/2)]
    if radius is None: # use the smallest distance between the center and image walls
        radius = min(center[0], center[1], w-center[0], h-center[1])

    Y, X = np.ogrid[:h, :w]
    dist_from_center = np.sqrt((X - center[0])**2 + (Y-center[1])**2)

    mask = np.zeros((h, w), dtype=int)
    mask[dist_from_center <= radius] = 1
    return mask
    

class SimpleGalaxy(ExpGalaxy):
    '''This defines the 'SIMP' galaxy profile -- an exponential profile
    with a fixed shape of a 0.45 arcsec effective radius and spherical
    shape.  It is used to detect marginally-resolved galaxies.
    '''
    shape = EllipseE(0.45, 0., 0.)

    def __init__(self, *args):
        super(SimpleGalaxy, self).__init__(*args)

    def __str__(self):
        return (self.name + ' at ' + str(self.pos)
                + ' with ' + str(self.brightness))

    def __repr__(self):
        return (self.name + '(pos=' + repr(self.pos) +
                ', brightness=' + repr(self.brightness) + ')')

    @staticmethod
    def getNamedParams():
        return dict(pos=0, brightness=1)

    def getName(self):
        return 'SimpleGalaxy'

    ### HACK -- for Galaxy.getParamDerivatives()
    def isParamFrozen(self, pname):
        if pname == 'shape':
            return True
        return super(SimpleGalaxy, self).isParamFrozen(pname) 


def reproject_discontinuous(input, out_wcs, out_shape, thresh=0.1):
    array, in_wcs = input
    logger = logging.getLogger('farmer.reproject_discontinuous')
    outarray = np.zeros(out_shape)
    segs = np.unique(array.flatten())
    segs = segs[segs!=0]
    sizes = np.array([np.sum(array==segid) for segid in segs])
    zorder = np.argsort(sizes)[::-1]
    sizes = sizes[zorder]
    segs = segs[zorder]

    for seg in tqdm(segs):
        mask = (array==seg).astype(int)
        newmask = reproject_interp((mask, in_wcs), out_wcs, out_shape, return_footprint=False)
        newmask = newmask > thresh
        outarray[newmask] = seg

    return outarray
    
