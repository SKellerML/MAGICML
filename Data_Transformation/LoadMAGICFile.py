import ctapipe

import traitlets

from ctapipe_io_magic import MAGICEventSourceMC, MAGICEventSource

from ctapipe.io import event_source

from ctapipe.io import event_source
from ctapipe.calib import CameraCalibrator
from enum import Enum
from ctapipe.visualization import CameraDisplay
import time
import copy
import pickle
import glob
import re
from matplotlib import pyplot as plt
#import PreprocessingFunctions

from pathlib import Path
from ctapipe.image import tailcuts_clean
from astropy import units as u
from ctapipe.io.containers import DataContainer

from enum import Enum

import transformcam1

class LoadOption(Enum):
    raw_sim = 0
    raw_data = 1
    cal_sim = 2
    cal_sim_MARS = 3
    cal_data = 4
    cal_at_once = 5


def load_obj(name):
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
        # print("Image: ",image)

    mask = tailcuts_clean(tel.camera, image, picture_thresh=6, boundary_thresh=3.5)
    count = 0
    for i in range(len(image)):
        if mask[i]:
            count += image[i]

    # print(count)
    return count > value


class MAGICFileLoader(object):
    """
    Loads CTA Files and handles them for saving events in different format
    """

    # particle ID: 0 for gamma, 1 for proton

    def __init__(self, load_option, file_path_mask, use_cut_value=-1):
        """

        :param particleID: ID of particle to save, deprecated
        :param telescopes: set of telescopes to save events for
        :param start_at_runnum: runnum to start at
        :param end_at_runnum: runnum to stop at (file is included)
        :param load_file_name_start: name of file to load after runnum
        :param load_file_name_end: name of file to load after runnum
        :param use_cut_value: use cut and remove all events lower, -1 for no cut
        """

        self.eventList = []
        self.load_option = load_option
        CameraCalibrator()
        # CameraCalibrator()

        self.cut_value = use_cut_value
        self.use_cut = False if self.cut_value == -1 else True
        self.currentFileID = 0

        #if load_option == LoadOption.cal_at_once:
        #    self.loadFilesWithMask()

        names_m1 = glob.glob(file_path_mask.replace('>', '1', 2))
        names_m2 = glob.glob(file_path_mask.replace('>', '2', 2))

        names_m1.sort()
        names_m2.sort()

        self.filename_list = list(zip(names_m1, names_m2))
        self.max_files = len(self.filename_list)

        print(self.filename_list[0])


    def loadFileWithID(self, fid):
        filename = self.loadFileName_start + str(fid) + self.loadFileName_end
        # if self.particleID == 1:
        #    filename = "Protons/proton_20deg_174.681deg_run" + str(
        #        fid) + "___cta-prod4-lapalma-2158m--baseline_with-MAGIC.simtel.gz"

        # elif self.particleID == 0:
        #    filename = "Gamma/gamma_20deg_174.681deg_run" + str(
        #        fid) + "___cta-prod4-lapalma-2158m--baseline_with-MAGIC_cone1.5.simtel.gz"

        #    print("Invalid particle ID given, please set 0 for gamma, 1 for proton file to load")
        self.loadFile(self.filepath + filename)

    def loadFileWithID_string(self, fidstr):
        filename = self.loadFileName_start + fidstr + self.loadFileName_end
        # if self.particleID == 1:
        #    filename = "Protons/proton_20deg_174.681deg_run" + str(
        #        fid) + "___cta-prod4-lapalma-2158m--baseline_with-MAGIC.simtel.gz"

        # elif self.particleID == 0:
        #    filename = "Gamma/gamma_20deg_174.681deg_run" + str(
        #        fid) + "___cta-prod4-lapalma-2158m--baseline_with-MAGIC_cone1.5.simtel.gz"

        #    print("Invalid particle ID given, please set 0 for gamma, 1 for proton file to load")
        self.loadFile(self.filepath + filename)

    def loadNextFile(self):

        # check if each run in m1 and m2 correspond
        #for name_tuple in self.filename_list:

        name_tuple = self.filename_list[self.currentFileID]


        mask = r"\w+_M\d{1}.*_(\d+).*_Y_.+.*"
        stereo_events = []
        parsed_info = re.findall(mask, name_tuple[0])
        parsed_info2 = re.findall(mask, name_tuple[1])

        self.data = False

        if not parsed_info[0] == parsed_info2[0]:
            print("Error: Run numbers do not match")
        else:
            if self.data:
                sl1 = self.load_files_data(name_tuple)
            else:
                sl1 = self.loadFile_stereo(name_tuple[0], name_tuple[1])
            #stereo_events = stereo_events + sl1

        print("ADF: ", self.currentFileID)
        self.currentFileID += 1
        if self.currentFileID >= self.max_files:
            print("Loaded all files.")
            return False
        return True


    def loadFilesWithMask(self):
        # Use masks? path = "/remote/ceph/group/magic/MAGIC-LST/MARS/CrabNebula/CalibratedWithPointings/2018_03_09/*05070968*00[1-2]*root"

        mask = self.filepath + self.loadFileName_start + "*" + self.loadFileName_end
        # Might need to change this line to work?
        event_factory = MAGICEventSource(input_url=path)
        event_factory.allowed_tels = self.telescopes  # {1, 2}
        event_generator = event_factory._generator()
        # event = next(event_generator)

        for s_event in event_generator:
            # Use event only if M1 and M2 are triggered

            # if 1 in s_event.r0.tels_with_data and 1 in self.telescopes and 2 in s_event.r0.tels_with_data:
            if self.telescopes <= s_event.r0.tels_with_data:
                self.calibrator.calibrate(s_event)
                # print("Oder num: {:d}, event id: {:.0f}, triggered telescopes: {}".format(stereo_event.count, stereo_event.r0.event_id, stereo_event.r0.tels_with_data))
                a = copy.deepcopy(s_event)
                self.eventList.append(a)
        # event_factory.pyhessio.close_file()
        print("New File loaded, file {:d} contains {:d} events".format(self.currentFileID, len(self.eventList)))


    def load_files_data(self, path):
        print("A: ",path)
        event_factory = MAGICEventSourceMC(input_url=path)

        event_generator = event_factory._generator()

        cnt_t = 0
        cnt_id1 = 0

        end1 = False

        while not end1:
            event = next(event_generator, None)
            if event is None:
                end1 = True
            else:
                self.eventList.append(copy.deepcopy(event))

        print(cnt_t)
        # event_factory.pyhessio.close_file()
        print("New Files loaded, file {:d} contains {:d} events".format(self.currentFileID, len(self.eventList)))
        return True


    def loadFile_stereo(self, path1, path2):

        stereo_event_list = []

        print("path1: ", path1)
        print("path2: ", path2)

        event_factory1 = MAGICEventSourceMC(input_url=path1)
        event_factory2 = MAGICEventSourceMC(input_url=path2)

        event_generator1 = event_factory1._generator()
        event_generator2 = event_factory2._generator()

        cnt_t = 0
        cnt_id1 = 0
        cnt_id2 = 0

        end1 = False
        end2 = False

        event1 = next(event_generator1, None)
        event2 = next(event_generator2, None)
        
        camera = ctapipe.instrument.CameraGeometry.from_name("MAGICCam")
        
        while end1 == False and end2 == False:
            if event2 is None:
                end2 = True
            else:
                id2 = event2.r0.event_id
            if event1 is None:
                end1 = True
            else:
                id1 = event1.r0.event_id
            if end1 == False and end2 == False:
                if id1 == id2:
                    # print("True")
                    nxt = True

                    cnt_id1 += 1
                    cnt_id2 += 1
                    cnt_t += 1
                    
                    image1 = event1.dl1.tel[1].image
                    image2 = event2.dl1.tel[2].image
                    # cut events with tailcuts_clean
                    boundary, picture, min_neighbors = (3.5,6,1)#(5, 10, 3)#cleaning_level[camera.cam_id]
                    clean1 = tailcuts_clean(
                        camera,
                        image1,
                        boundary_thresh=boundary,
                        picture_thresh=picture,
                        min_number_picture_neighbors=min_neighbors
                    )
                    clean2 = tailcuts_clean(
                        camera,
                        image2,
                        boundary_thresh=boundary,
                        picture_thresh=picture,
                        min_number_picture_neighbors=min_neighbors
                    )
                    #print("B1: ", clean1.sum(), " B2: ", clean2.sum() )
                    if clean1.sum() > 5 and clean2.sum() > 5:
                        self.eventList.append({"M1": copy.deepcopy(event1), "M2": copy.deepcopy(event2)})
                    event1 = next(event_generator1, None)
                    event2 = next(event_generator2, None)


                elif id1 > id2:
                    event2 = next(event_generator2, None)
                    cnt_id2 += 1
                elif id1 < id2:
                    event1 = next(event_generator1, None)
                    cnt_id1 += 1

        # count to the end
        '''
        if True:  # end1 == True:
            while end2 == False:
                event2 = next(event_generator2, None)
                cnt_id2 += 1
                if event2 is None:
                    end2 = True
                # print("Counting 1...")
        if True:  # end2 == True:
            while end1 == False:
                event1 = next(event_generator1, None)
                cnt_id1 += 1
                if event1 is None:
                    end1 = True
                    # print("Counting 2...")
        '''
        print(cnt_t)
        # event_factory.pyhessio.close_file()
        print("New File loaded, file {:d} contains {:d} events".format(self.currentFileID, len(self.eventList)))
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
        # print("Loaded")
        return True

    def getNextEvent(self):
        """
        Returns next event and removes it from eventList
        Automatically loads next file if eventList is empty
        """
        self.checkEventList()
        return self.eventList.pop(0)


class Telescopes(Enum):
    MAGIC1 = 1
    MAGIC2 = 2
    LST1 = 3

def list_to_string(lst):
    string = ""
    for y in lst:
            string += "," + str(y)
    return string


def matrix_to_string(matrix):
    string = ""
    for y in matrix:
        for x in y:
            string += "," + str(x)
    return string


def remap_no(image, dict_tdc, padding=-10):
    # print(image)
    return [image]


def remap(image, dict_tdc, padding=-10):
    llx = [[padding for i in range(0, dict_tdc[-1])] for n in range(0, dict_tdc[-2])]

    for i, n in enumerate(image, 0):
        ct = dict_tdc[i]

        llx[ct[1]][ct[0]] = n

    return llx


def remap_oversampling(image, dict_tdc, padding=-10):
    llx = [[padding for i in range(0, dict_tdc[-1] + 1)] for n in range(0, dict_tdc[-2] + 1)]
    for i, n in enumerate(image, 0):
        ct = dict_tdc[i]
        m = n
        llx[ct[1]][ct[0]] = m
        llx[ct[1]][ct[0] + 1] = m
        llx[ct[1] + 1][ct[0] + 1] = m
        llx[ct[1] + 1][ct[0]] = m
        # print("llx: ",llx[ct[1]][ct[0]]," | ",llx[ct[1]][ct[0]+1])
    return llx


def remap_linear_interpol(image, dict_tdc, padding=-10):
    maxx = 2 * dict_tdc[-1] - 1
    maxy = dict_tdc[-2]
    llx = [[-1000 for i in range(0, maxx)] for n in range(0, maxy)]
    # Insert Info into list
    for i, n in enumerate(image, 0):
        ct = dict_tdc[i]
        llx[ct[1]][2 * ct[0]] = n

    # interpolate in y
    for u in range(maxx):
        if (u % 2 == 0):
            for i, n in enumerate(llx, 0):  # starting at the bottom, going up
                if (n[u] > -999):
                    if not (i + 2 >= maxy):  # if we are not at the top
                        if (llx[i + 2][u] > -999):
                            # interpolate between both values
                            llx[i + 1][u] = (llx[i + 2][u] + llx[i][u]) / 2
    # interpolate in x
    for u in range(0, maxx - 2, 2):
        for i, n in enumerate(llx, 0):  # starting at the bottom, going up
            if (n[u] > -999):
                if (llx[i][u + 2] > -999):
                    # interpolate between both values
                    llx[i][u + 1] = ((llx[i][u + 2] + llx[i][u]) / 2)
                else:  # This line is to make everything at least 2 pixel wide
                    pass  # llx[i][u + 1] = llx[i][u]

    for y in range(maxy):
        for x in range(len(llx[y])):
            if (llx[y][x] < -999):
                llx[y][x] = padding

    # Remove extremely negative pixel

    return llx

class PortFiles_2(object):

    def __init__(self, use_cut_value=-1, get_pedestals_only=False):
        '''

        :param remap_dict_type: "SPACED_GRID" or "HEXAGONAL"
        :param use_cut_value: use cut and remove all events lower, -1 for no cut
        '''
        self.pedestals_only = get_pedestals_only
        self.__load_list = []  # contains (file name beginning, file name end, run num start, run num end,

        self.dict_name_string = "Dict"  # "MAGIC" is added later

        self.use_cut_value = use_cut_value

        # particleID (-1 for unknown )

    '''
    def add_to_load_list(self, file_name_mask, run_start_num=0, run_end_num=0, mask_split_character="*",
                         particleID=-1, load_option=False, filepath=None, calibrate = False):
        """

        :param file_name_mask:
        :param run_start_num:
        :param run_end_num:
        :param mask_split_character:
        :param particleID:
        :param filepath:
        :return:
        """

        file_name_masked_list = file_name_mask.split(sep=mask_split_character)
        if filepath is None:
            filepath = data_path
        self.__load_list.append((file_name_masked_list[0], file_name_masked_list[1], run_start_num, run_end_num,
                                 filepath, particleID, load_option, calibrate))


        # load list description:
        # masked name start, masked name end, run start num, run end num, filepath, particle ID

    # debug functions
    def getll(self):
        return self.__load_list
    '''

    def save_files_to_csv(self, generated_file_name, path_mask, max_events_per_file=1000):
        # load dictionaries according to telescopes

        # Could be added:
        # check if files already exist and append further values if yes

        # Loop through load list
        # print("ECO: ",type(ddctlist[0][0]))
        cfileID = 0
        # max_events_per_file = 1000
        #for c_load in self.__load_list:
        
        t1 = transformcam1.Magic_Coordinate_Transformation()
        
        if True:
            loader = MAGICFileLoader(load_option=0, file_path_mask=path_mask, use_cut_value=self.use_cut_value)
            # Loop until current file list is empty
            # for fnum in range(0, maxFiles):
            while loader.checkEventList():
                # Open new file to write to
                print("E ", loader.currentFileID)
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
                        #print(event)
                        # convert it to readable csv string
                        # format: energy, particleID, altitude, azimuth, core_x, core_y, h_first_int, x_mac

                        # pid = c_load[4]
                        # -----------------------------------
                        # maybe check for right unit?
                        # -----------------------------------

                        # 0.35011425614356995 TeV 	1.21102rad 	2.99732rad
                        # 78.76237487792969 m 	1.421595811843872 m 	22278.5 m 	348.3333435058594 g / cm2
                        if True:#loader.load_option == LoadOption.raw_sim or loader.load_option == LoadOption.cal_sim:

                            pid = event['M1'].mc.shower_primary_id
                            if pid == 101:  # if it is a proton
                                pid = 0
                            elif pid == 0:  # if it is a gamma
                                pid = 1
                            evmc = event['M1'].mc
                            
                            p_alt = event['M1'].pointing[1].altitude#.to_value(u.deg) 
                            p_az = event['M1'].pointing[1].azimuth#.to_value(u.deg)
                            
                            em_alt = evmc.alt#.to_value(u.deg)
                            em_az = evmc.az#.to_value(u.deg)
                            #print("C: ", em_alt, "C: ", em_az, "C: ", p_alt, "C: ", p_az)
                            emx, emy, tel_alt, tel_az = t1.transform_to_telescope_xy_altaz(em_alt, em_az, p_alt, p_az)
                            #print("B: ", emx, "B: ", emy, "B: ", p_alt, "B: ", p_az)
                            #"," + str(evmc.x_max)
                            line_str = str(pid) + "," + str(evmc.energy.to_value(u.gigaelectronvolt)) + \
                                        "," + str(evmc.alt.to_value(u.deg)) + \
                                        "," + str(evmc.az.to_value(u.deg)) + \
                                        "," + str(evmc.core_x.to_value(u.meter)) + \
                                        "," + str(evmc.core_y.to_value(u.meter)) + \
                                        "," + str(evmc.h_first_int.to_value(u.meter)) + \
                                        "," + str(event['M1'].pointing[1].altitude.to_value(u.deg)) + \
                                        "," + str(event['M1'].pointing[1].azimuth.to_value(u.deg)) + \
                                        "," + str(emx.to_value(u.meter)) + \
                                        "," + str(emy.to_value(u.meter)) + \
                                        "," + str(tel_alt.to_value(u.deg)) + \
                                        "," + str(tel_az.to_value(u.deg))
                            line_str += list_to_string(event['M1'].dl1.tel[1].image)
                            line_str += list_to_string(event['M2'].dl1.tel[2].image)

                            line_str += list_to_string(event['M1'].dl1.tel[1].pulse_time)
                            line_str += list_to_string(event['M2'].dl1.tel[2].pulse_time)
                        '''
                        else:
                            pid = 0
                            line_str = str(pid) + "," + str(0) + "," + str(0) \
                                       + "," + str(0) + "," + str(0) + "," + str(0) + "," \
                                       + str(0) + "," + str(0)
                        '''
                                    # for i in event.dl1.tel[dct[1]]["image"]:
                                    #    print(i)
                                    # print("")
                                    # print("")
                                    # print("")
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
                        # loader.checkEventList()
                        runnum += 1

                cfileID += 1
