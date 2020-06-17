"""
Transforms data from spiral to matrix
"""

import ctapipe

import traitlets

from ctapipe_io_magic import MAGICEventSource

from ctapipe.io import event_source

from ctapipe.calib import CameraCalibrator
from enum import Enum
from ctapipe.visualization import CameraDisplay
import time
import copy
import pickle
import re
import glob
from matplotlib import pyplot as plt
#import PreprocessingFunctions

from pathlib import Path
from ctapipe.image import tailcuts_clean
from astropy import units as u
import importlib

import LoadMAGICFile as lf
import LoadCTAFile_MAGIC as lf2
from ctapipe.coordinates import (
    GroundFrame,
    TiltedGroundFrame,
    NominalFrame,
    TelescopeFrame,
    CameraFrame,
)
from astropy.coordinates import EarthLocation, SkyCoord, FK5, AltAz, Angle
from astropy.time import Time

from ctapipe.instrument import CameraGeometry

from astropy.utils import iers
iers.conf.auto_download = False  

class Magic_Coordinate_Transformation(object):
    def __init__(self, obstime=None):
        if obstime is None:
            self.obstime = Time('2013-11-01T03:00')
        else:
            self.obstime = obstime
            
        self.magic_location = EarthLocation(lat=Angle("28°45′42.462″"),
                           lon=Angle("-17°53′26.525″"),
                           height=2199.4 * u.m)
        
        self.altaz = AltAz(location=self.magic_location, obstime=self.obstime)
        
        self.focal_length = 16.97*u.m # Maybe get from file?
        self.camera = CameraGeometry.from_name('MAGICCam')

        
        
    def transform_to_telescope_camera(self, alt, az, pointing_alt, pointing_az):
        pointing_coord = SkyCoord(alt=0*u.deg+pointing_alt,
                          az=pointing_az,
                          frame=altaz,
                          unit='deg')

        #pointing = pointing_coord_m.transform_to(self.altaz)

        camera_frame_m = CameraFrame(
            telescope_pointing=pointing_coord,
            focal_length=self.focal_length,
            obstime=self.obstime,
            location=self.magic_location,
            rotation=0*u.deg
        )

        event_location_m = SkyCoord(alt=0*u.deg+alt,
                                  az=az,
                                  frame=altaz,
                                  unit='deg')
        
        event_loc_cam = event_location_m.transform_to(camera_frame_m)
        
        return event_loc_cam.x, event_loc_cam.y
        
    def transform_to_telescope_altaz(self, alt, az, pointing_alt, pointing_az):
        """
        Returns alt, az
        
        """
        
        pointing_coord = SkyCoord(alt=0*u.deg+pointing_alt,
                          az=pointing_az,
                          frame=altaz,
                          unit='deg')

        #pointing = pointing_coord_m.transform_to(self.altaz)

        camera_frame_m = CameraFrame(
            telescope_pointing=pointing_coord,
            focal_length=self.focal_length,
            obstime=self.obstime,
            location=self.magic_location,
            rotation=0*u.deg
        )

        event_location = SkyCoord(alt=0*u.deg+alt,
                                  az=az,
                                  frame=altaz,
                                  unit='deg')
        
        tf1 = TelescopeFrame(telescope_pointing = pointing_coord, obstime = obstime, location = magic_location)
        loc1 = event_location.transform_to(tf1)

        loc1.delta_az.to_value(u.deg)
        
        
        return loc1.delta_alt.to_value(u.deg), loc1.delta_az.to_value(u.deg)     
        
    def transform_to_telescope_xy_altaz_from_event(self, event):
    
        pointing_alt = event.pointing[1].altitude
        pointing_az = event.pointing[1].azimuth
        alt = event.mc.alt
        az = event.mc.az
        
        return self.transform_to_telescope_xy_altaz(alt, az, pointing_alt, pointing_az)
    
    
    def transform_to_telescope_xy_altaz(self, alt, az, pointing_alt, pointing_az):
        """
        Returns alt, az
        
        """
        # change azimuth if negative
        az2 = az.to_value(u.deg)
        while az2 > 360:
            az2 -= 360
        while az2 < 0:
            az2 += 360
        az = az2*u.deg
        
        pointing_coord = SkyCoord(alt=0*u.deg+pointing_alt,
                          az=pointing_az,
                          frame=self.altaz,
                          unit='deg')

        #pointing = pointing_coord_m.transform_to(self.altaz)

        camera_frame = CameraFrame(
            telescope_pointing=pointing_coord,
            focal_length=self.focal_length,
            obstime=self.obstime,
            location=self.magic_location,
            rotation=(-90+70.9)*u.deg
        )

        #print("A1: ",0*u.deg+alt, " V: ", az, " D: ",self.altaz)
        
        event_location = SkyCoord(alt=0*u.deg+alt,
                                  az=az,
                                  frame=self.altaz,
                                  unit='deg')
        
        event_loc_cam = event_location.transform_to(camera_frame)
        x1 = event_loc_cam.x
        y1 = event_loc_cam.y
        tf1 = TelescopeFrame(telescope_pointing = pointing_coord, obstime = self.obstime, location = self.magic_location)
        loc1 = event_location.transform_to(tf1)

        loc1.delta_az.to_value(u.deg)
        
        return x1,y1, loc1.delta_alt, loc1.delta_az
        
    def plot_event_with_shower_position_seperate_events(self, event1, event2):
        
        x, y, alt, az = self.transform_to_telescope_xy_altaz_from_event(event1)
        
        x = x.to_value(u.m)
        y = y.to_value(u.m)
        
        # Cameras
        m1_cam = copy.deepcopy(event1.inst.subarray.tel[1].camera)
        m2_cam = copy.deepcopy(event2.inst.subarray.tel[2].camera)

        m1_cam.rotate(-19.1*u.deg)#(-90+70.9)
        m2_cam.rotate((-90+70.9)*u.deg)

        # Charge images
        m1_event_image = event1.dl1.tel[1].image
        m2_event_image = event2.dl1.tel[2].image

        # Peak position maps
        #m1_event_times = b5["M1"].dl1.tel[1].peakpos
        #m2_event_times = b5["M2"].dl1.tel[2].peakpos

        fig1, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15,10))

        # M1 charge map
        disp1 = CameraDisplay(m1_cam, m1_event_image, ax=ax1, title="MAGIC 1 image")
        disp1.add_colorbar(ax=ax1)

        # M2 charge map
        disp2 = CameraDisplay(m2_cam, m2_event_image,  ax=ax2, title="MAGIC 2 image")
        disp2.add_colorbar(ax=ax2)

        ax1.plot(x, y, marker='*', color='red')
        ax1.annotate(
            s="Peter", xy=(x, y), xytext=(5, 5),
            textcoords='offset points', color='red',

        )
        #ax1.scatter(cam.pix_x, cam.pix_y, marker="o", s=10, c='red')

        ax2.plot(x, y, marker='*', color='red')
        ax2.annotate(
            s="Peter", xy=(x, y), xytext=(5, 5),
            textcoords='offset points', color='red',
        )
        
        
    


