# General imports
import os
import sys

if os.path.exists(os.path.join(os.getcwd(), 'config')): # You're 1 up from config?
    sys.path.insert(0, os.path.join(os.getcwd(), 'config'))
else: # You're working from a directory parallel with config?
    sys.path.insert(0, os.path.abspath(os.path.join(os.getcwd(), '../config')))

# Miscellaneous science imports
from astropy.io import fits, ascii
from astropy.table import Table, Column, vstack, join
from astropy.wcs import WCS
import astropy.units as u
import numpy as np
from functools import partial
import matplotlib.pyplot as plt
import weakref
from scipy import stats
import pathos as pa
from astropy.coordinates import SkyCoord
import collections
from tqdm import tqdm

# Local imports
# from .brick import Brick
from .mosaic import Mosaic
# from .group import Group
try:
    import config as conf
except:
    raise RuntimeError('Cannot find configuration file!')

# Make sure no interactive plotting is going on.
plt.ioff()
import warnings
warnings.filterwarnings("ignore")# General imports
if os.path.exists(os.path.join(os.getcwd(), 'config')): # You're 1 up from config?
    sys.path.insert(0, os.path.join(os.getcwd(), 'config'))
else: # You're working from a directory parallel with config?
    sys.path.insert(0, os.path.abspath(os.path.join(os.getcwd(), '../config')))

print(
f"""
====================================================================
T H E
 ________    _       _______     ____    ____  ________  _______        
|_   __  |  / \     |_   __ \   |_   \  /   _||_   __  ||_   __ \    
  | |_ \_| / _ \      | |__) |    |   \/   |    | |_ \_|  | |__) |   
  |  _|   / ___ \     |  __ /     | |\  /| |    |  _| _   |  __ /    
 _| |_  _/ /   \ \_  _| |  \ \_  _| |_\/_| |_  _| |__/ | _| |  \ \_ 
|_____||____| |____||____| |___||_____||_____||________||____| |___|
                                                                    
--------------------------------------------------------------------
 M O D E L   P H O T O M E T R Y   W I T H   T H E   T R A C T O R   
--------------------------------------------------------------------
    Version 2.0                               
    (C) 2018-2022 -- J. Weaver (DAWN, University of Copenhagen)          
====================================================================

CONSOLE_LOGGING_LEVEL ..... {conf.CONSOLE_LOGGING_LEVEL}			
LOGFILE_LOGGING_LEVEL ..... {conf.LOGFILE_LOGGING_LEVEL}												
PLOT ...................... {conf.PLOT}																		
NTHREADS .................. {conf.NTHREADS}																			
OVERWRITE ................. {conf.OVERWRITE} 
"""	
)

# Load logger
from .utils import start_logger
logger = start_logger()

# Look at mosaics and check they exist
logger.info('Verifying bands...')
for band in conf.BANDS.keys():
    Mosaic(band, load=False)


def get_mosaic(band, load=True):
    return Mosaic(band, load=load)

def build_bricks(brick_ids=None, include_detection=True, bands=None, write=False):
    if bands is not None: # some kind of manual job
        if np.isscalar(bands):
            bands = [bands,]
    elif bands == 'detection':
        bands = [bands,]
        include_detection = False
    else:
        bands = list(conf.BANDS.keys())

    # Check first
    for band in bands:
        if band == 'detection':
            break
        if band not in conf.BANDS.keys():
            raise RuntimeError(f'Cannot find {band} -- check your configuration file!')

    if include_detection:
        bands = ['detection'] + bands

    # Generate brick_ids
    if brick_ids is None:
        n_bricks = conf.N_BRICKS[0] * conf.N_BRICKS[1]
        brick_ids = 1 + np.arange(n_bricks)

    # Build bricks
    if np.isscalar(brick_ids): # single brick built in memory and saved
        for band in bands:
            mosaic = get_mosaic(band, load=True)
            try:
                mosaic.add_to_brick(brick)
            except:
                brick = mosaic.spawn_brick(brick_ids)
            del mosaic
        if write: brick.write(allow_update=False)
        return brick
    else: # If brick_ids is none, then we're in production. Load in mosaics, make bricks, update files.
        for band in bands:
            mosaic = get_mosaic(band, load=True)
            arr = brick_ids
            if conf.CONSOLE_LOGGING_LEVEL != 'DEBUG':
                arr = tqdm(brick_ids)
            logger.info('Spawning bricks...')
            for brick_id in arr:
                brick = mosaic.spawn_brick(brick_id)
                if write: brick.write(allow_update=True)
            del mosaic
        return

def detect_sources(brick_ids=None, band='detection', imgtype='science', mosaic=None, overwrite=False):

    # TODO his background workflow is messy... 
    # can we just force bricks to get the parent backgrounds at init if confiigured like that?

    # Deal with mosaic backgrounds first
    background = None
    if conf.DETECTION['subtract_background'] & (conf.DETECTION['backregion'] == 'mosaic'):
        if mosaic is None:
            mosaic = get_mosaic('detection')
            sepback = mosaic.estimate_background()
        else:
            try:
                sepback = mosaic.backgrounds[imgtype]
            except:
                sepback = mosaic.estimate_background()

    # Generate brick_ids
    if brick_ids is None:
        n_bricks = conf.N_BRICKS[0] * conf.N_BRICKS[1]
        brick_ids = 1 + np.arange(n_bricks)
    elif np.isscalar(brick_ids):
        brick_ids = [brick_ids,]

    # Loop over bricks
    for brick_id in brick_ids:
        
        # does the brick exist? load it.
        try:
            brick = load_brick(brick_id)
        except:
            brick = build_bricks(brick_id,  bands='detection')

        # deal with backgrounds on bricks
        if conf.DETECTION['subtract_background']:
            if conf.DETECTION['backregion'] == 'brick':
                try:
                    sepback = brick.backgrounds[band][imgtype]
                except:
                    sepback = brick.estimate_background(band=band)
            
            # Which kind?
            if conf.DETECTION['backtype'] == 'flat':
                background = sepback.globalback
            elif conf.DETECTION['backtype'] == 'variable':
                background = sepback.back()

            # Make a cutout and inheret mosaic info
            if conf.DETECTION['backregion'] == 'mosaic':
                brick.data[band]['background'].data = sepback.back()[brick.data[band][imgtype].slices_original]
                brick.data[band]['rms'].data = sepback.rms()[brick.data[band][imgtype].slices_original]
                brick.properties[band][imgtype] = sepback.globalback
                brick.properties[band][imgtype] = sepback.globalrms
                if conf.DETECTION['backtype'] == 'variable':
                    background = brick.data[band]['background'].data

        # detection
        brick.extract(band=band, imgtype=imgtype, background=background)

        # grouping
        brick.identify_groups(band=band, imgtype=imgtype)

    if len(brick_ids) == 1:
        return brick

def generate_models(brick_ids=None, bands=conf.MODEL_BANDS, imgtype='science'):
    # get bricks with 'brick_ids' for 'bands'
    if brick_ids is None:
        n_bricks = conf.N_BRICKS[0] * conf.N_BRICKS[1]
        brick_ids = 1 + np.arange(n_bricks)
    elif np.isscalar(brick_ids):
        brick_ids = [brick_ids,]

    # Loop over bricks (or just one!)
    for brick_id in brick_ids:
        # does the brick exist? load it.
        try:
            brick = load_brick(brick_id)
        except:
            brick = build_bricks(brick_id,  bands=bands)

        # check that detection exists
        assert('detection' in brick.bands, f'No detection information contained in brick #{brick.brick_id}!')

        #TODO make sure background is dealt with

        # detect sources
        brick.detect_sources()

        # loop or parallel groups
        for group_id in brick.group_ids['detection']:

            # get group
            group = brick.spawn_group(group_id)

            # transfer maps
            group.transfer_maps()

            # stage images and models
            group.stage_engine()

            # optimize
            group.optimize()

            # run model DT

            # ancillary stuff

            # cleanup

    pass

def photometer(brick_ids=None, bands=None, imgtype='science'):
    # get bricks with 'brick_ids' for 'bands'

    # make sure background is dealt with

    # check that model solutions exist

    # stage images

    # stage models

    # force models

    # ancillary stuff

    # cleanup
    
    pass


def rebuild_mosaic(brick_ids=None, bands=None, imgtype='science'):
    pass

def load_brick(brick_id=None, bands=None):
    'find them and build them up'
    raise RuntimeError('Not implelented yet!')