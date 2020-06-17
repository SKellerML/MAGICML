import ctapipe

import traitlets

from ctapipe_io_magic import MAGICEventSource

from ctapipe.io import event_source

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

from pathlib import Path
from ctapipe.image import tailcuts_clean
from astropy import units as u

from ctapipe.io.containers import DataContainer

import transformcam1

data_path = "/remote/ceph/group/magic/ie.vovk/Data/CTA/MAGIC_LST/Simulations/Simtelarray/Diffuse/"


from enum import Enum
class LoadOption(Enum):
    raw_sim = 0
    raw_data = 1
    cal_sim = 2
    cal_sim_MARS = 3
    cal_data = 4
    cal_at_once = 5


def load_obj(name ):
    with open('obj/' + name + '.pkl', 'rb') as f:
        return pickle.load(f)


def apply_cut(event, value, load_option=LoadOption.raw_sim):
    """
    Takes calibrated event, returns whether it survives the cut or not

    :param event: calibrated event
    :param value: value to cut off at

    returns bool: Whether the image is above the cut True or not False
    """

    tel = event.inst.subarray.tel[1]
    if load_option == LoadOption.raw_sim or load_option == LoadOption.cal_sim:
        image = event.dl1.tel[1].image[1]
    else:
        image = event.dl1.tel[1].image
        #print("Image: ",image)

    mask = tailcuts_clean(tel.camera, image, picture_thresh=6, boundary_thresh=3.5)
    count = 0
    for i in range(len(image)):
        if mask[i]:
            count += image[i]

    #print(count)
    return count > value


class CTAFileLoader(object):
    """
    Loads CTA Files and handles them for saving events in different format
    """
    # particle ID: 0 for gamma, 1 for proton

    def __init__(self, telescopes, file_name_mask, load_option, calibrate=False, use_cut_value=-1, pedestals_only=False):
        """

        :param particleID: ID of particle to save, deprecated
        :param telescopes: set of telescopes to save events for
        :param start_at_runnum: runnum to start at
        :param end_at_runnum: runnum to stop at (file is included)
        :param load_file_name_start: name of file to load after runnum
        :param load_file_name_end: name of file to load after runnum
        :param use_cut_value: use cut and remove all events lower, -1 for no cut
        """
        self.pedestals_only = pedestals_only
        self.calibrate = calibrate
        self.eventList = []
        self.load_option = load_option
        CameraCalibrator()
        # CameraCalibrator()
        if calibrate:
            config = traitlets.config.Config()
            self.calibrator = CameraCalibrator(config=config)
            # r1_product="HESSIOR1Calibrator",extractor_product="NeighbourPeakIntegrator", 
        
        self.telescopes = telescopes

        #
        #  Mask run number with @
        #
        self.runnumset = set()

        p = file_name_mask.split(sep="/")
        path = ""
        for k in range(len(p) - 1):
            path += p[k] + "/"
        filename_mask = p[-1]
        p = None

        self.masked_file_name = filename_mask
        self.path = path

        fl = glob.glob(path + filename_mask.replace('@', '*'))
        mask = r"\w+_M\d{1}.*_(\d+).*_Y_.+.*"

        for fp in fl:
            s2 = fp.split(sep="/")[-1]
            self.runnumset.add(re.findall(mask, s2)[0])



        self.file_name_mask = filename_mask
        self.cut_value = use_cut_value
        self.use_cut = False if self.cut_value == -1 else True


        if load_option == LoadOption.cal_at_once: 
            self.loadFilesWithMask()

    def loadFileWithID(self, fid):
        filename = self.loadFileName_start + str(fid) + self.loadFileName_end
        #if self.particleID == 1:
        #    filename = "Protons/proton_20deg_174.681deg_run" + str(
        #        fid) + "___cta-prod4-lapalma-2158m--baseline_with-MAGIC.simtel.gz"

        #elif self.particleID == 0:
        #    filename = "Gamma/gamma_20deg_174.681deg_run" + str(
        #        fid) + "___cta-prod4-lapalma-2158m--baseline_with-MAGIC_cone1.5.simtel.gz"

        #    print("Invalid particle ID given, please set 0 for gamma, 1 for proton file to load")
        self.loadFile(self.filepath + filename)

    def loadFileWithID_string(self, fidstr):
        filename = self.loadFileName_start + fidstr + self.loadFileName_end
        #if self.particleID == 1:
        #    filename = "Protons/proton_20deg_174.681deg_run" + str(
        #        fid) + "___cta-prod4-lapalma-2158m--baseline_with-MAGIC.simtel.gz"

        #elif self.particleID == 0:
        #    filename = "Gamma/gamma_20deg_174.681deg_run" + str(
        #        fid) + "___cta-prod4-lapalma-2158m--baseline_with-MAGIC_cone1.5.simtel.gz"

        #    print("Invalid particle ID given, please set 0 for gamma, 1 for proton file to load")
        self.loadFile(self.filepath + filename)

    def loadNextFile(self):

        maxFilesToTest = 100
        currentFileToTest = 0
        
        # If all files were loaded at once
        if self.load_option == LoadOption.cal_at_once:
            if self.eventList <= 0:
                print("Eventlist empty - all events used")
                return False

            return True

        if len(self.runnumset) > 0:
            # runnumset
            namecopy = copy.deepcopy(self.file_name_mask)
            fnmask = self.path + namecopy.replace("@",self.runnumset.pop())
            print("QAI: ",fnmask)
            #try:
            if self.loadFile(fnmask):
                return True
            #finally:
            #    print("Error loading files, ending...")
        print("Loaded all files.")
        return False
        print("Could not find next file.")

    def loadFilesWithMask(self):
        # Use masks? path = "/remote/ceph/group/magic/MAGIC-LST/MARS/CrabNebula/CalibratedWithPointings/2018_03_09/*05070968*00[1-2]*root"
        
        mask = self.filepath + self.loadFileName_start +"*"+ self.loadFileName_end 
            # Might need to change this line to work?
        event_factory = MAGICEventSource(input_url=path)  
        event_factory.allowed_tels = self.telescopes #{1, 2}
        event_generator = event_factory._generator()
        # event = next(event_generator)

        for s_event in event_generator:
            # Use event only if M1 and M2 are triggered

            #if 1 in s_event.r0.tels_with_data and 1 in self.telescopes and 2 in s_event.r0.tels_with_data:
            if self.telescopes <= s_event.r0.tels_with_data:
                self.calibrator.calibrate(s_event)
                # print("Oder num: {:d}, event id: {:.0f}, triggered telescopes: {}".format(stereo_event.count, stereo_event.r0.event_id, stereo_event.r0.tels_with_data))
                a = copy.deepcopy(s_event)
                self.eventList.append(a)
        #event_factory.pyhessio.close_file()
        print("New File loaded, file {:d} contains {:d} events".format(14, len(self.eventList)))
        
    def loadFile(self, path):
        # Use masks? path = "/remote/ceph/group/magic/MAGIC-LST/MARS/CrabNebula/CalibratedWithPointings/2018_03_09/*05070968*00[1-2]*root"

            # might need to change this line to work?
        #try:
        if True:    
            if self.load_option == LoadOption.raw_sim or self.load_option == LoadOption.raw_data:
                event_factory = event_source(input_url=path)  
                event_factory.allowed_tels = self.telescopes #{1, 2}
                event_generator = event_factory._generator()
                if self.pedestals_only:
                    print("Warning: Pedestals only available for real data, not for simulation data.")

            elif self.load_option == LoadOption.cal_sim or self.load_option == LoadOption.cal_data:
                #event_factory.allowed_tels = self.telescopes #{1, 2}
                if self.pedestals_only:
                    event_factory = MAGICEventSource(input_url=path)
                    pedestal_event_generator1 = event_factory._pedestal_event_generator(telescope='M1')
                    pedestal_event_generator2 = event_factory._pedestal_event_generator(telescope='M2')
                else:
                    event_factory = MAGICEventSource(input_url=path)
                    event_generator = event_factory._generator()
                    
        else: #except Exception as e:
            print("Error 194: File does not exist: " + path + " or ")
            print( "Error: %s" % str(e) )
            return False
        # event = next(event_generator)
        if self.pedestals_only:
            ped_available = True
            while ped_available:
                try:
                    p1 = next(pedestal_event_generator1, None)
                    p2 = next(pedestal_event_generator2, None)
                except:
                    print("Errror:")
                    p1 = None
                    p2 = None

                
                if p1 is None or p2 is None:
                    ped_available=False
                else:
                    a1 = copy.deepcopy(p1)
                    a2 = copy.deepcopy(p2)
                    self.eventList.append({1:a1,2:a2})
        else:
            for s_event in event_generator:
                # Use event only if M1 and M2 are triggered
                #if 1 in s_event.r0.tels_with_data and 1 in self.telescopes and 2 in s_event.r0.tels_with_data:
                if self.telescopes <= s_event.r0.tels_with_data:
                    if self.calibrate:
                        self.calibrator(s_event)
                    # print("Oder num: {:d}, event id: {:.0f}, triggered telescopes: {}".format(stereo_event.count, stereo_event.r0.event_id, stereo_event.r0.tels_with_data))
                    a = copy.deepcopy(s_event)
                    if not self.use_cut or apply_cut(a, self.cut_value, self.load_option):
                        self.eventList.append(a)
                        #print("Accepted. ",len(self.eventList))



        #event_factory.pyhessio.close_file()
        print("New File loaded, file {:d} contains {:d} events".format(14, len(self.eventList)))
        return True


        
    def checkEventList(self):
        """
        Checks whether eventList is empty and loads new file if it is
        Returns False if it is empty and all files were loaded
        Return True if it is not empty
        """
        if len(self.eventList) < 1:
            print("Not enough")
            if self.loadNextFile():
                if not self.checkEventList():
                    return False
            else:
                return False
        #print("Loaded")
        return True

    def getNextEvent(self):
        """
        Returns next event and removes it from eventList
        Automatically loads next file if eventList is empty
        """
        if self.checkEventList():
            return self.eventList.pop(0)
        else:
            return None


class Telescopes(Enum):
    MAGIC1 = 1
    MAGIC2 = 2
    LST1 = 3


def matrix_to_string(matrix):
    string = ""
    for y in matrix:
        for x in y:
            string+=","+str(x)
    return string

def list_to_string(lst):
    string = ""
    for y in lst:
            string += "," + str(y)
    return string

def remap_no(image, dict_tdc, padding = -10):
    #print(image)
    return [image]

def remap(image, dict_tdc, padding = -10):
    llx = [[padding for i in range(0, dict_tdc[-1])] for n in range(0, dict_tdc[-2])]

    for i, n in enumerate(image, 0):
        ct = dict_tdc[i]

        llx[ct[1]][ct[0]] = n

    return llx


def remap_oversampling(image, dict_tdc, padding = -10):
    llx = [[padding for i in range(0, dict_tdc[-1]+1)] for n in range(0, dict_tdc[-2]+1)]
    for i, n in enumerate(image, 0):
        ct = dict_tdc[i]
        m = n
        llx[ct[1]][ct[0]] = m
        llx[ct[1]][ct[0]+1] = m
        llx[ct[1]+1][ct[0]+1] = m
        llx[ct[1]+1][ct[0]] = m
        # print("llx: ",llx[ct[1]][ct[0]]," | ",llx[ct[1]][ct[0]+1])
    return llx


def remap_linear_interpol(image, dict_tdc, padding = -10):
    maxx = 2*dict_tdc[-1] - 1
    maxy = dict_tdc[-2]
    llx = [[-1000 for i in range(0, maxx)] for n in range(0, maxy)]
    # Insert Info into list
    for i, n in enumerate(image, 0):
        ct = dict_tdc[i]
        llx[ct[1]][2 * ct[0]] = n

    # interpolate in y
    for u in range(maxx):
        if(u % 2 == 0):
            for i, n in enumerate(llx, 0): # starting at the bottom, going up
                if(n[u] > -999):
                    if not (i + 2 >= maxy): # if we are not at the top
                        if (llx[i+2][u] > -999):
                            # interpolate between both values
                            llx[i + 1][u] = (llx[i+2][u]+llx[i][u])/2
    # interpolate in x
    for u in range(0, maxx - 2, 2):
        for i, n in enumerate(llx, 0):  # starting at the bottom, going up
            if (n[u] > -999):
                if (llx[i][u + 2] > -999):
                    # interpolate between both values
                    llx[i][u + 1] = ( (llx[i][u + 2] + llx[i][u]) / 2 )
                else:   # This line is to make everything at least 2 pixel wide
                    pass#llx[i][u + 1] = llx[i][u]

    for y in range(maxy):
        for x in range(len(llx[y])):
            if (llx[y][x] < -999):
                llx[y][x] = padding




    # Remove extremely negative pixel

    return llx

import astropy as ap

from astropy.coordinates import EarthLocation, SkyCoord, FK5, AltAz, Angle
from astropy.time import Time

class PortFiles(object):

    def __init__(self, use_cut_value=-1, get_pedestals_only=False):
        '''

        :param remap_dict_type: "SPACED_GRID" or "HEXAGONAL"
        :param use_cut_value: use cut and remove all events lower, -1 for no cut
        '''
        self.pedestals_only = get_pedestals_only

        self.__load_list = [] # contains (file name beginning, file name end, run num start, run num end,

        self.use_cut_value = use_cut_value


    def add_to_load_list(self, file_name_mask, particleID=-1, load_option=False, filepath=None, calibrate = False):
        """

        :param file_name_mask:
        :param run_start_num:
        :param run_end_num:
        :param mask_split_character:
        :param particleID:
        :param filepath:
        :return:
        """

        self.file_name_mask = file_name_mask

        # load list description:
        # masked name start, masked name end, run start num, run end num, filepath, particle ID

    # debug functions
    def getll(self):
        return self.__load_list



    def save_files_to_csv(self, file_name_mask, generated_file_name, telescopes, load_option, max_events_per_file=1000, calibrate=False):

        magic_location = EarthLocation(lat=Angle("28°45′42.462″"),
                                       lon=Angle("-17°53′26.525″"),
                                       height=2199.4 * u.m)

        t1 = transformcam1.Magic_Coordinate_Transformation()
        f = ap.coordinates.SkyCoord.from_name('Crab Nebula')

        # Loop through load list
        #print("ECO: ",type(ddctlist[0][0]))
        cfileID = 0
        # max_events_per_file = 1000
        self.file_name_mask = file_name_mask
        if True:
            loader = CTAFileLoader( telescopes, file_name_mask = self.file_name_mask,
                                    use_cut_value=self.use_cut_value,
                                    pedestals_only = self.pedestals_only,
                                    calibrate=calibrate,
                                    load_option=load_option)
            # Loop until current file list is empty
            # for fnum in range(0, maxFiles):
            print("Saving...")
            loader.loadNextFile() # load files
            while loader.checkEventList():
                # Open new file to write to
                #print("E ",loader.currentFileID)
                with open(generated_file_name + str(cfileID) + ".csv", 'w') as file:
                    print("New File: ", generated_file_name + str(cfileID) + ".csv")

                    # titlestr = "MAGIC,"+str(ddct[-1])+","+str(ddct[-2])
                    # titlestr = "energy, particleID, altitude, azimuth, core_x, core_y, h_first_int, x_max"
                    # for i in range(0,pixnum):
                    #    titlestr += ",1_"+str(i)
                    # for i in range(0,pixnum):
                    #    titlestr += ",2_"+str(i)
                    # titlestr += "\n"
                    # file.write(titlestr)
                    titlestr = ""
                    runnum = 0



                    # Write 1000 images to the file
                    while runnum < max_events_per_file:
                        # get first event and remove it
                        event = loader.getNextEvent()

                        # convert it to readable csv string
                        # format: energy, particleID, altitude, azimuth, core_x, core_y, h_first_int, x_mac

                        
                        # pid = c_load[4]
                        # -----------------------------------
                        # maybe check for right unit?
                        # -----------------------------------

                        # 0.35011425614356995 TeV 	1.21102rad 	2.99732rad
                        # 78.76237487792969 m 	1.421595811843872 m 	22278.5 m 	348.3333435058594 g / cm2
                        if event is not None:
                            if True:#try:
                                if loader.load_option == LoadOption.raw_sim or loader.load_option == LoadOption.cal_sim:

                                    pid = event.mc.shower_primary_id
                                    if pid == 101: # if it is a proton
                                        pid = 0
                                    elif pid == 0: # if it is a gamma
                                        pid = 1

                                    line_str = str(pid) + "," + str(event.mc.energy.value) + "," + str(
                                        event.mc.alt.value) \
                                               + "," + str(event.mc.az.value) \
                                               + "," + str(event.pointing[1].altitude.to_value(u.deg)) \
                                               + "," + str(event.pointing[1].azimuth.to_value(u.deg))
                                               # '''+ "," + str(event.mc.core_x.value) + "," + str(
                                               # event.mc.core_y.value) + "," \
                                               #+ str(event.mc.h_first_int.value) + "," + str(event.mc.x_max.value)
                                               #'''

                                    for tel in telescopes:

                                            line_str += matrix_to_string(event.dl1.tel[tel].image[1])
                                else:
                                    pid = 0
                                    #print("E1: ", event)
                                    if self.pedestals_only:
                                        #print(event[1])
                                        line_str = "0,0,0,0,0,0,0" \
                                                   + "," + str(event[1].pointing[1].altitude.to_value(u.deg)) \
                                                   + "," + str(event[1].pointing[1].azimuth.to_value(u.deg)) \
                                                   + "0,0,0,0"

                                        for tel in telescopes:
                                            line_str += list_to_string(event[tel].dl1.tel[tel]["image"])
                                        for tel in telescopes:
                                            line_str += list_to_string(event[tel].dl1.tel[tel].pulse_time)
                                        # '''+ "," + str(0) + "," + str(0) + "," \

                                    else:


                                        f2 = f.transform_to(AltAz(obstime= Time(event.trig.gps_time.value,
                                                                                   format='unix',
                                                                                   out_subfmt='longdate'),
                                                                              location=magic_location))
                                        horizon_frame = AltAz()
                                        array_pointing = ap.coordinates.SkyCoord(
                                            az=event.pointing[1].azimuth,
                                            alt=event.pointing[1].altitude,
                                            frame=horizon_frame
                                        )

                                        emx, emy, tel_alt, tel_az = t1.transform_to_telescope_xy_altaz(f2.altaz.alt,
                                                                                                       f2.altaz.az,
                                                                                                       array_pointing.altaz.alt,
                                                                                                       array_pointing.altaz.az)
                                        # m33altaz = f.transform_to(array_pointing)


                                        line_str = "0,0,0,0,0,0" \
                                                   + "," + str(event.trig.gps_time.value)\
                                                   + "," + str(event.pointing[1].altitude.to_value(u.deg)) \
                                                   + "," + str(event.pointing[1].azimuth.to_value(u.deg)) \
                                                   + "0,0" \
                                                   + "," + str(tel_alt) + "," + str(tel_az)

                                        for tel in telescopes:
                                            line_str += list_to_string(event.dl1.tel[tel]["image"])
                                        for tel in telescopes:
                                            line_str += list_to_string(event.dl1.tel[tel].pulse_time)


                                            #for i in event.dl1.tel[dct[1]]["image"]:
                                            #    print(i)
                                            #print("")
                                            #print("")
                                            #print("")
                                # Add Image to string for each telescope dict object in ddctlist

                                # line_str += matrix_to_string(remap(event.dl1.tel[2].image[1], ddct))
                                # plotBoth(event)
                                # runnum = max_events_per_file+200
                                # fnum = maxFiles+2
                                # plotByMatrix( remap(event.dl1.tel[1].image[1],ddct) )
                                # print(line_str)
                                # for i in event.dl1.tel[1].image[1]:
                                #    line_str += ","+str(i)
                                # for i in event.dl1.tel[2].image[1]:
                                #    line_str += ","+str(i)
                                line_str += "\n"
                                file.write(line_str)
                                if not loader.checkEventList():
                                    break
                                #loader.checkEventList()
                                runnum += 1
                            else:#except BaseException:
                                print("Error")
                        else:
                            return 
                        
                    cfileID += 1


    """
    def __init__(self, telescopes, file_name_mask_list, filepath = data_path, mask_split_char = "*"):
        self.file_name_mask_list = file_name_mask_list
        self.fname_masked = file_name_mask.split(sep = mask_split_char)
        self.filepath = filepath

        self.ctaLoader = CTAFileLoader(telescopes, start_at_runnum=0, end_at_runnum=0,
                                       load_file_name_start=self.fname_masked[0],
                                       load_file_name_end=self.fname_masked[1],
                                       filepath=self.filepath)
    """




# mask split with *

'''
def preprocess_data(particle_id=1, telescopes=(1, 2), generated_file_name="MLD1/Stereo/2_stereo_hex_gamma_magic_mc_",
                    preprocess_fnc = None, preprocess_fnc_after_remap = None, remap_padding = -10,
                    append = False, fileName = ""):
    """
    Load CTA files and save as csv file

    :param particle_id: Gamma = 0, Proton = 1
    :param telescopes: Set
    :param generated_file_name: "MLD1/Stereo/..."
    :param preprocess_fnc: function with which the data is preprocessed as a list before remapping
    :param preprocess_fnc_after_remap: function with which the data is preprocessed as a list before remapping
    :return:
    """
    # Save into file
    loader = CTAFileLoader(telescopes=telescopes, )
    """
    maxFiles = 1000
    if particle_id == 0:
        cfileID = 0
        max_events_per_file = 2000
        # pixnum = 1039

        # generatedFileName = "MLD1/Stereo/stereo_hex_gamma_magic_mc_"
    if particle_id == 1:
        cfileID = 1000
        max_events_per_file = 2000
        # pixnum = 1039

        # generatedFileName = "MLD1/Stereo/stereo_hex_proton_magic_mc_"
    """
    ddctlist = []
    try:
        if 1 in telescopes:
            ddctlist.append( (load_obj("MAGICDict"), 1) )
        if 2 in telescopes:
            ddctlist.append( (load_obj("MAGICDict"), 2) )
        if 3 in telescopes:
            ddctlist.append( (load_obj("LSTDict"), 3) )
    except:
        print("Required dict not found")
        return

    if preprocess_fnc is None:
        preprocess_fnc = lambda n: n
    if preprocess_fnc_after_remap is None:
        preprocess_fnc_after_remap = lambda n: n

    #ddct = load_obj("MAGICDict")

    # check if files already exist and append further values if yes
    pass


    # Loop through all files to be created
    for fnum in range(0, maxFiles):
        # Open new file to write to
        with open(generated_file_name + str(cfileID) + ".csv", 'w') as file:
            # titlestr = "MAGIC,"+str(ddct[-1])+","+str(ddct[-2])
            # titlestr = "energy, particleID, altitude, azimuth, core_x, core_y, h_first_int, x_max"
            # for i in range(0,pixnum):
            #    titlestr += ",1_"+str(i)
            # for i in range(0,pixnum):
            #    titlestr += ",2_"+str(i)
            # titlestr += "\n"
            # file.write(titlestr)
            titlestr = ""
            runnum = 0
            # Write 1000 images to the file
            while runnum < max_events_per_file:
                # get first event and remove it
                event = loader.getNextEvent()

                # convert it to readable csv string
                # format: energy, particleID, altitude, azimuth, core_x, core_y, h_first_int, x_mac
                line_str = str(event.mc.energy) + "," + str(event.mc.shower_primary_id) + "," + str(event.mc.alt) \
                           + ","+ str(event.mc.az) + "," + str(event.mc.core_x) + "," + str(event.mc.core_y) + "," \
                           + str(event.mc.h_first_int) + "," + str(event.mc.x_max)
                # Add Image to string for each dict in ddctlist
                for dct in ddctlist:
                    line_str += matrix_to_string(preprocess_fnc_after_remap(remap(preprocess_fnc(event.dl1.tel[ dct[1] ].image[0]),
                                                       dct[0], remap_padding)))
                #line_str += matrix_to_string(remap(event.dl1.tel[2].image[0], ddct))
                # plotBoth(event)
                # runnum = max_events_per_file+200
                # fnum = maxFiles+2
                # plotByMatrix( remap(event.dl1.tel[1].image[0],ddct) )
                # print(line_str)
                # for i in event.dl1.tel[1].image[0]:
                #    line_str += ","+str(i)
                # for i in event.dl1.tel[2].image[0]:
                #    line_str += ","+str(i)
                line_str += "\n"
                file.write(line_str)

                loader.checkEventList()
                runnum += 1

        cfileID += 1
'''


def getSingleEvent(telescopes, fileID = 1, eventID = 1, particleID=0 ):
    # telescopes=set((1,2)
    if particleID==0:
        loader = CTAFileLoader(telescopes=telescopes, start_at_runnum=fileID, end_at_runnum=fileID,
                               load_file_name_start = "Gamma/gamma_20deg_174.681deg_run",
                               load_file_name_end = "___cta-prod4-lapalma-2158m--baseline_with-MAGIC_cone1.5.simtel.gz")
    else:
        loader = CTAFileLoader(telescopes=telescopes, start_at_runnum=fileID, end_at_runnum=fileID,
                               load_file_name_start="Protons/proton_20deg_174.681deg_run",
                               load_file_name_end="___cta-prod4-lapalma-2158m--baseline_with-MAGIC.simtel.gz")
    loader.loadFileWithID(fileID)
    if len(loader.eventList) < eventID:
        print("Event ID too big, file contains ", len(loader.eventList), " events.")
    return loader.eventList[eventID]