import ctapipe

import traitlets
from ctapipe.io import EventSourceFactory
from ctapipe.io import event_source
from ctapipe.calib import CameraCalibrator
from enum import Enum
from ctapipe.visualization import CameraDisplay
import time
import copy
import pickle
from matplotlib import pyplot as plt
import PreprocessingFunctions

from pathlib import Path


from ctapipe.io.containers import DataContainer

data_path = "/remote/ceph/group/magic/ie.vovk/Data/CTA/MAGIC_LST/Simulations/Simtelarray/Diffuse/"


def load_obj(name ):
    with open('obj/' + name + '.pkl', 'rb') as f:
        return pickle.load(f)


class CTAFileLoader(object):
    """
    Loads CTA Files and handles them for saving events in different format
    """
    # particle ID: 0 for gamma, 1 for proton

    def __init__(self, telescopes, start_at_runnum = 0, end_at_runnum = 0,
                 load_file_name_start="Protons/proton_20deg_174.681deg_run",
                 load_file_name_end="___cta-prod4-lapalma-2158m--baseline_with-MAGIC.simtel.gz",
                 filepath=data_path):
        """

        :param particleID: ID of particle to save, deprecated
        :param telescopes: set of telescopes to save events for
        :param start_at_runnum: runnum to start at
        :param end_at_runnum: runnum to stop at (file is included)
        :param load_file_name_start: name of file to load after runnum
        :param load_file_name_end: name of file to load after runnum
        :param
        """

        self.eventList = []

        # CameraCalibrator()
        config = traitlets.config.Config()
        self.calibrator = CameraCalibrator(r1_product="HESSIOR1Calibrator",
                                           extractor_product="NeighbourPeakIntegrator", config=config)
        self.telescopes = telescopes

        self.currentFileID = start_at_runnum

        self.start_at_runnum = start_at_runnum
        self.end_at_runnum = end_at_runnum if end_at_runnum >= start_at_runnum else start_at_runnum
        self.loadFileName_start = load_file_name_start
        self.loadFileName_end = load_file_name_end
        self.filepath = filepath

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

        while currentFileToTest < maxFilesToTest:
            if self.currentFileID > self.end_at_runnum:
                print("Loaded all files.")
                return False

            filename = self.loadFileName_start + str(self.currentFileID) + self.loadFileName_end
            #if self.particleID == 1:
            #    filename = "Protons/proton_20deg_174.681deg_run" + str(
            #        self.currentFileID) + "___cta-prod4-lapalma-2158m--baseline_with-MAGIC.simtel.gz"

            #elif self.particleID == 0:
            #    filename = "Gamma/gamma_20deg_174.681deg_run" + str(
            #        self.currentFileID) + "___cta-prod4-lapalma-2158m--baseline_with-MAGIC_cone1.5.simtel.gz"
            #else:
            #    print("Invalid particle ID given, please set 0 for gamma, 1 for proton file to load")
            #    return

            # Check whether the file exists or not

            my_file = Path(self.filepath + filename)
            print(my_file)
            if not my_file.is_file():
                print("File does not exist: " + filename)
                self.currentFileID += 1
            else:
                self.loadFile(self.filepath + filename)
                self.currentFileID += 1
                return True


            currentFileToTest += 1

        print("Could not find next file.")

    def loadFile(self, path):
        event_factory = EventSourceFactory.produce(input_url=path)
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
        event_factory.pyhessio.close_file()
        print("New File loaded, file {:d} contains {:d} events".format(self.currentFileID, len(self.eventList)))

    def checkEventList(self):
        """
        Checks whether eventList is empty and loads new file if it is
        Returns False if it is empty and all files were loaded
        Return True if it is not empty
        """
        if len(self.eventList) < 1:
            if self.loadNextFile():
                if not self.checkEventList():
                    return False
            else:
                return False

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


def matrix_to_string(matrix):
    string = ""
    for y in matrix:
        for x in y:
            string+=","+str(x)
    return string


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


class PortFiles(object):

    def __init__(self, remap_dict_type):
        '''

        :param remap_dict_type: "SPACED_GRID" or "HEXAGONAL"
        '''
        self.__load_list = [] # contains (file name beginning, file name end, run num start, run num end,

        self.dict_name_string = "Dict" # "MAGIC" is added later

        if remap_dict_type == "OVERSAMPLING" or remap_dict_type == "SPACED_GRID":
            self.dict_name_string = "_SPACED_GRID"
            print("Using Oversampling remapping dict: ",self.dict_name_string)
        # particleID (-1 for unknown )

    def add_to_load_list(self, file_name_mask, run_start_num=0, run_end_num=0, mask_split_character="*",
                         particleID=-1, filepath=None):
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
        self.__load_list.append((file_name_masked_list[0], file_name_masked_list[1], run_start_num, run_end_num, filepath, particleID))


        # load list description:
        # masked name start, masked name end, run start num, run end num, filepath, particle ID

    # debug functions
    def getll(self):
        return self.__load_list

    def save_files_to_csv(self, generated_file_name, telescopes, remap_fnc, preprocess_fnc = None,
                          preprocess_fnc_after_remap=None, remap_padding=-10, max_events_per_file=1000, 
                          preprocess_fnc_multiple_return_values=False):
        # load dictionaries according to telescopes
        ddctlist = []
        try:
            if 1 in telescopes:
                ddctlist.append((load_obj("MAGIC"+self.dict_name_string), 1))
            if 2 in telescopes:
                ddctlist.append((load_obj("MAGIC"+self.dict_name_string), 2))
            if 3 in telescopes:
                ddctlist.append((load_obj("MAGIC"+self.dict_name_string), 3))
        except:
            print("Required dict not found")
            return

        # Set preprocessing functions
        if preprocess_fnc is None:
            preprocess_fnc = lambda n: n
        if preprocess_fnc_after_remap is None:
            preprocess_fnc_after_remap = lambda n: n

        # Could be added:
        # check if files already exist and append further values if yes

        # Loop through load list
        #print("ECO: ",type(ddctlist[0][0]))
        cfileID = 0
        # max_events_per_file = 1000
        for c_load in self.__load_list:
            loader = CTAFileLoader( telescopes, start_at_runnum = c_load[2], end_at_runnum = c_load[3],
                                    load_file_name_start=c_load[0],
                                    load_file_name_end=c_load[1],
                                    filepath=c_load[4])
            # Loop until current file list is empty
            # for fnum in range(0, maxFiles):
            while loader.checkEventList():
                # Open new file to write to
                print("E ",loader.currentFileID)
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
                        pid = event.mc.shower_primary_id
                        if pid == 101: # if it is a proton
                            pid = 0
                        elif pid == 0: # if it is a gamma
                            pid = 1
                        # pid = c_load[4]
                        # -----------------------------------
                        # maybe check for right unit?
                        # -----------------------------------

                        # 0.35011425614356995 TeV 	1.21102rad 	2.99732rad
                        # 78.76237487792969 m 	1.421595811843872 m 	22278.5 m 	348.3333435058594 g / cm2

                        line_str = str(pid) + "," + str(event.mc.energy.value) + "," + str(
                            event.mc.alt.value) \
                                   + "," + str(event.mc.az.value) + "," + str(event.mc.core_x.value) + "," + str(
                            event.mc.core_y.value) + "," \
                                   + str(event.mc.h_first_int.value) + "," + str(event.mc.x_max.value)
                        # Add Image to string for each telescope dict object in ddctlist
                        for dct in ddctlist:
                            if preprocess_fnc_multiple_return_values == True:
                                preprocess_result_tuple = preprocess_fnc(event.dl1.tel[dct[1]].image[0])
                                for i in range(1, len(preprocess_result_tuple)):
                                    line_str += preprocess_result_tuple[i]
                                    
                                line_str += matrix_to_string(                                
                                    preprocess_fnc_after_remap(remap_fnc(preprocess_result_tuple[0],
                                                                     dct[0], remap_padding)))
                            else: 
                                line_str += matrix_to_string(
                                    preprocess_fnc_after_remap(remap_fnc(preprocess_fnc(event.dl1.tel[dct[1]].image[0]),
                                                                         dct[0], remap_padding)))
                        # line_str += matrix_to_string(remap(event.dl1.tel[2].image[0], ddct))
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
                        if not loader.checkEventList():
                            break
                        #loader.checkEventList()
                        runnum += 1

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