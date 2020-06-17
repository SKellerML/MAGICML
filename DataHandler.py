import torch as torch
from torch.utils import data
import copy
import pandas as pd
import numpy as np
import DataProcessingFunction as DPF

import DataOption as DataOption
import MagicDataset as MD

from matplotlib import pyplot as plt
import torch.optim as optim
import time as time

from sklearn import metrics as metrics


class DataHandler(object):
    """
    Loads and Handles Magic Data

    Constructor:
    maxFilesGamma
    maxFilesProton
    gammaFileName           Gamma file name without file number
    protonFileName          Proton file name without file number
    startAtFile_Gamma=0
    startAtFile_Proton=0
    telSizeX=39:            Pixel width of telescope
    telSizeY=34             Pixel height of telescope

    """

    def __init__(self, telescope_ids, data_option=DataOption.mono,
                 regressor_columns=None, time_channel=False, pixnum=1039, timing_info=True):

        """
        :param telescope_ids: list/tuple of telescopes to use
        :param mono_in_multi_telescope_telescope: Select only telescope #mono_in_multi_telescope_telescope in data,
                                                  -1 for use all telescopes
        :param keepInfoColumns: ['particleID','energy','altitude','azimuth','core_x','core_y','h_first_int','x_max']
        """

        self.energy_filter = False
        self.energy_filter_min = 0
        self.energy_filter_max = 10000

        # maxFilesGamma, maxFilesProton, gammaFileName, protonFileName, startAtFile_Gamma=0,
        # startAtFile_Proton=0, telSizeX=39, telSizeY=34, stereo = False, mono_in_stereo_data_telescope=-1,
        # keepInfoColumns = None):
        self.timing_info = timing_info
        self.start_columns = 8
        self.mc_info_columns = self.start_columns  # len(keepInfoColumns) # might need to be changed, not sure though
        self.data_is_selected = False

        self.telescope_ids = telescope_ids
        self.data_option = data_option
        self.time_channel = time_channel

        self.colList = ['particleID', 'energy', 'altitude', 'azimuth', 'core_x', 'core_y', 'h_first_int',
                        'pointing_alt', 'pointing_az', 'pos_in_cam_x', 'pos_in_cam_y', 'event_tel_alt', 'event_tel_az']

        self.col_list_base_info = ['particleID', 'energy', 'altitude', 'azimuth', 'core_x', 'core_y', 'h_first_int',
                                   'pointing_alt', 'pointing_az', 'pos_in_cam_x', 'pos_in_cam_y', 'event_tel_alt',
                                   'event_tel_az']

        self.end_columns = ["min_val", "max_val"]

        self.info_col_num = len(self.col_list_base_info)
        self.pixnum = pixnum  # get as parameter?

        self.telescope_data_position_map = {}
        # Images
        for i in range(1, len(self.telescope_ids) + 1):
            # Add Columns to colList
            imagename = "i_M" + str(i)

            for x in range(0, self.pixnum):
                self.colList.append(imagename + "_" + str(x))

            # Add data location information to location list
            self.telescope_data_position_map[imagename] = (imagename + "_0", imagename + '_' + str(self.pixnum - 1))

        # Timing
        for i in range(1, len(self.telescope_ids) + 1):
            # Add Columns to colList
            peakname = "p_M" + str(i)

            for x in range(0, self.pixnum):
                self.colList.append(peakname + "_" + str(x))

            # Add data location information to location list
            self.telescope_data_position_map[peakname] = (peakname + "_0", peakname + '_' + str(self.pixnum - 1))

        self.colList = self.colList + self.end_columns
        self.c_dataframe = pd.DataFrame(columns=self.colList)

        print(self.c_dataframe.columns)
        print("Columns finished")

        self.regressor_columns = regressor_columns


    def set_energy_filter(self, energy_filter_min=0, energy_filter_max=10000, energy_mult_factor=1):
        """
        Sets the energy filter to only select events between energy_filter_min and energy_filter_max with the
        energy_mult_factor multiplied with the min and max value.


        :param energy_filter_min:
        :param energy_filter_max:
        :param energy_mult_factor:
        :return:
        """
        self.energy_filter = True
        self.energy_filter_min = energy_filter_min * energy_mult_factor
        self.energy_filter_max = energy_filter_max * energy_mult_factor

    def load_files_and_save(self, file_name, max_num_files=1, start_at_file=0,
                            overwrite_particle_id=-10, given_columns=None, path="", max_per_file=1000, csv_start_num=0,
                            transform=False):
        """
        Loads files and saves them after transforming/ applying other operations like energy filtering etc.

        :param file_name: file name to load without file number and .csv ending
        :param max_num_files: maximum number of files to load
        :param start_at_file: file number to start loading at
        :param overwrite_particle_id: whether to overwrite the particle ID with this number, -10 if do not overwrite
        :param given_columns: Information columns given in the file, should correspond to columns defined in this class
        :param path: Path to save new files to
        :param max_per_file: Maximum Number of Events per File
        :param csv_start_num: Start Saving at this number
        :param transform: Transform the data with 3i and 5 (see documentation for these two function)

        """

        if given_columns is None:
            colList2 = self.col_list_base_info + self.colList[self.info_col_num:]
        else:
            colList2 = given_columns + self.colList[self.info_col_num:]
            # print("12000: ",colList2)

        g_dataframe = pd.read_csv(
            file_name + str(start_at_file) + ".csv",
            sep=",",
            names=colList2,
            index_col=None,
            header=None)
        # g_dataframe['pID'] = g_dataframe['pID'].map(lambda x: 1)
        # header = None)
        # header=0)
        print("Loaded file ", file_name + str(start_at_file) + ".csv")
        if self.energy_filter:
            g_dataframe = self.clip_by_energy_df(g_dataframe, self.energy_filter_min, self.energy_filter_max)
        if transform:
            g_dataframe = g_dataframe.apply(DPF.process_data3_i, axis=1, args=(self.info_col_num, self.pixnum))
            g_dataframe = g_dataframe.apply(DPF.process_data5, axis=1, args=[self.info_col_num])

        maxFileNum = start_at_file + max_num_files
        n = start_at_file + 1

        while n < maxFileNum:
            a_dataframe = pd.read_csv(
                file_name + str(n) + ".csv",
                sep=",",
                names=colList2)
            # header=None)
            # a_dataframe['pID'] = a_dataframe['pID'].map(lambda x: 1)
            if self.energy_filter:
                a_dataframe = self.clip_by_energy_df(a_dataframe, self.energy_filter_min, self.energy_filter_max)
            if transform:
                a_dataframe = a_dataframe.apply(DPF.process_data3_i, axis=1, args=(self.info_col_num, self.pixnum))
                a_dataframe = a_dataframe.apply(DPF.process_data5, axis=1, args=[self.info_col_num])
            # print(len(g_dataframe), " EFFEL ", len(a_dataframe))
            g_dataframe = pd.concat([g_dataframe, a_dataframe], ignore_index=True, axis=0)
            print(len(g_dataframe), " R ")
            # print("Loaded file ", file_name + str(n) + ".csv")
            n += 1
            if len(g_dataframe) > max_per_file:
                g_dataframe = self.save_as_csv_part(g_dataframe, path=path, max_per_file=max_per_file,
                                                    csv_start_num=csv_start_num)
                csv_start_num += 1

        if overwrite_particle_id != -10:
            g_dataframe.loc[:, "particleID"] = overwrite_particle_id

        # Add missing columns
        for i in range(len(self.colList[:self.info_col_num])):
            if self.colList[i] in colList2:
                pass
            else:
                g_dataframe.insert(i, self.colList[i], [0 for m in range(len(g_dataframe))])

        #return g_dataframe

    def save_as_csv_part(self, g_dataframe, path, max_per_file=1000, csv_start_num=0):
        """
        Saves part of dataframe to csv file. Starts at the beginning of the file until max_per_file events are saved

        :param g_dataframe: dataframe to save from
        :param path: Path to save new files to
        :param max_per_file: Maximum Number of Events per File
        :param csv_start_num: Start Saving at this number
        :return:
        """
        print("Len1: ", len(g_dataframe))
        d_dataframe = g_dataframe.iloc[: max_per_file]
        d_dataframe.to_csv(path + str(csv_start_num) + ".csv", index=False, header=None)
        g_dataframe = g_dataframe.tail(len(g_dataframe) - max_per_file)
        print("Len2: ", len(g_dataframe))
        return g_dataframe

    def save_as_csv(self, path, max_per_file=1000, csv_start_num=0):
        """
        Saves whole dataframe to .csv files.
        :param path: Path to save new files to
        :param max_per_file: Maximum Number of Events per File
        :param csv_start_num: Start Saving at this number
        :return:
        """
        datnum = len(self.c_dataframe)

        if datnum < max_per_file:
            self.c_dataframe.to_csv(path + "0.csv", index=False, header=None)
            return True

        filenum = csv_start_num
        cdatnum = max_per_file
        while cdatnum < datnum:
            self.c_dataframe.iloc[cdatnum - max_per_file: cdatnum].to_csv(path + str(filenum) + ".csv", index=False,
                                                                          header=None)
            cdatnum += max_per_file
            filenum += 1

        self.c_dataframe.iloc[cdatnum - max_per_file:].to_csv(path + str(filenum) + ".csv", index=False, header=None)

    def clip_by_energy(self, min_energy, max_energy):
        """
        Performs the energy clipping
        :param min_energy: lower energy bound
        :param max_energy: higher energy bound
        :return:
        """
        self.c_dataframe = self.c_dataframe.loc[self.c_dataframe["energy"] > min_energy, :]
        self.c_dataframe = self.c_dataframe.loc[self.c_dataframe["energy"] < max_energy, :]
        print("Energy clipped to [", min_energy, ",", max_energy, "]")
        print("After clipping containts ", len(self.c_dataframe), " entries.")

    def __len__(self):
        """
        Returns the number of events in the DataHandler object.
        :return: number of events in the DataHandler object
        """
        return len(self.c_dataframe)

    def transform_data(self, transformation_function, extra_columns=["t_min", "t_max"]):
        """
        Transform Data to with transformation function, add min and max value columns
        :param transformation_function:
        :param extra_columns:
        :return:
        """
        self.c_dataframe.reindex(columns=[*self.c_dataframe.columns.tolist(), *extra_columns])
        self.c_dataframe = self.c_dataframe.apply(transformation_function, axis=1)

    def bin_select_energy(self, bin_start, bin_end):
        """
        Selects data into energy bin
        :param bin_start: Start of Energy Bin
        :param bin_end: End of Energy Bin
        :return:
        """
        self.c_dataframe = self.c_dataframe.loc[self.c_dataframe["energy"] > bin_start, :]
        self.c_dataframe = self.c_dataframe.loc[self.c_dataframe["energy"] < bin_end, :]
        # print(self.c_dataframe)
        print("Remaining Data: ", self.c_dataframe["particleID"].value_counts())
        plt.hist(self.c_dataframe["energy"])

    def sum_pixels(self, idx):
        """
        Sums up all pixel in event in brightness image
        :param idx:
        :return:
        """
        row = self.c_dataframe.iloc[idx, self.mc_info_columns:].to_numpy()
        return sum(row)

    def add_files(self, max_files, file_name, start_at_file=0, overwrite_particle_id=-10, given_columns=None,
                  save_while_loading=False, path='', max_per_file=1000, csv_start_num=0, transform=False):
        """
        Add files to the DataHandler Object
        :param max_files:
        :param file_name:
        :param start_at_file:
        :param overwrite_particle_id:
        :param given_columns:
        :param save_while_loading: Save to new files
        :param path: Path to save new files to
        :param max_per_file: Maximum Number of Events per File
        :param csv_start_num: Start Saving at this number
        :param transform: Whether it sould be transformed or not
        :return:
        """
        # if particle_id == 0:
        #    adatf = self.loadProtonFiles(file_name, max_files, start_at_file)
        # elif particle_id == 1:
        #    adatf = self.loadGammaFiles(file_name, max_files, start_at_file)

        if save_while_loading:
            adatf = self.load_files_and_save(file_name, max_files, start_at_file,
                                             overwrite_particle_id, given_columns=given_columns, path=path,
                                             max_per_file=max_per_file, csv_start_num=csv_start_num,
                                             transform=transform)
        else:
            adatf = self.load_files(file_name, max_files, start_at_file,
                                    overwrite_particle_id, given_columns=given_columns)

        # add to existing c_dataframe
        self.c_dataframe = pd.concat([self.c_dataframe, adatf], ignore_index=True, axis=0)

    def filter_data(self, is_simtel=False, is_pedestal=False):
        # Replace all NaN
        self.c_dataframe = self.c_dataframe.fillna(0)  # self.c_dataframe =s
        # remove negative values for magic sim only
        if not is_simtel:
            cols = self.c_dataframe.columns[self.info_col_num: self.info_col_num + 2 * self.pixnum]
            self.c_dataframe[cols] = self.c_dataframe[cols].clip(lower=0)
        # remove all pixel with time values above 50 or below 10

        # get all pixel with wrong time value
        minm_t = 11.4375
        maxm_t = 60
        mask = (self.c_dataframe.iloc[:,
                self.info_col_num + 2 * self.pixnum:self.info_col_num + 4 * self.pixnum] < minm_t) | (
                           self.c_dataframe.iloc[:,
                           self.info_col_num + 2 * self.pixnum:self.info_col_num + 4 * self.pixnum] > maxm_t)
        self.c_dataframe.loc[:, mask.columns] = self.c_dataframe.loc[:, mask.columns].mask(mask, other=minm_t)
        # remove all pixel with wrong time value in image domain
        mask.columns = self.c_dataframe.columns[self.info_col_num:self.info_col_num + 2 * self.pixnum]
        self.c_dataframe.loc[:, mask.columns] = self.c_dataframe.loc[:, mask.columns].mask(mask,
                                                                                           other=0)  # , inplace=False, axis=None, level=None, errors='raise', try_cast=False)[source]

        # remove too high values in pedestals?

    def select_data(self):
        """
        Randomizes Data - Depreceated, just call randomizeData()
        """
        self.randomizeData()

    def randomizeData(self):
        """
        Randomizes Data
        """
        self.c_dataframe = self.c_dataframe.reindex(np.random.permutation(self.c_dataframe.index))
        print("Randomized")

    def get_whole_dataset(self):
        """
        Returns whole Dataset as MagicDataset
        :return: MagicDataset containing whole data
        """
        return MD.MagicDataset(self.c_dataframe, data_option=self.data_option,
                               telescope_data_position_map=self.telescope_data_position_map,
                               regressor_columns=self.regressor_columns)

    def getTrainAndValidationDataset(self, percentTrain, percentValidation):
        """
        Returns Train and Validation Set of data with each percentage
        :param percentTrain: Not percentage but factor < 1, might not be used currently, only percentValidation works
        :param percentValidation: Not percentage but factor < 1
        :return: MagicDataset containing whole data
        """
        # Split Data
        datLength = self.c_dataframe.count()[0]
        valnum = int(round(percentValidation * datLength))
        # print(testnum)
        # Split 20% test, 80% train
        train_datf = self.c_dataframe.head(datLength - valnum)
        test_datf = self.c_dataframe.tail(valnum)

        print("Data split: Train: ", len(train_datf), " Test: ", len(test_datf))

        return MD.MagicDataset(train_datf, data_option=self.data_option,
                               telescope_data_position_map=self.telescope_data_position_map,
                               regressor_columns=self.regressor_columns), \
               MD.MagicDataset(test_datf, data_option=self.data_option,
                               telescope_data_position_map=self.telescope_data_position_map,
                               regressor_columns=self.regressor_columns)

    def get_suiting_magic_dataset_for_dataset(self, dataset):
        """
        Returns MagicDataset based on Pandas Dataframe
        :param dataset: Pandas Dataframe
        :return:
        """
        return MD.MagicDataset(dataset, self.data_option, telescope_data_position_map=self.telescope_data_position_map)

    def getTestDataset(self, percent):
        """

        Returns MagicDataset with percentage from the end
        :param dataset: <1 factor, not percentage from the end to include in MagicDataset
        :return:

        """
        return MD.MagicDataset(self.c_dataframe.tail(percent*len(self.c_dataframe)), self.data_option, telescope_data_position_map=self.telescope_data_position_map)

    def clip_by_energy_df(self, df, min_energy, max_energy):
        """
        Clip Dataframe by energy
        :param df: Pandas Dataframe
        :param min_energy: lower energy bound
        :param max_energy: higher energy bound
        :return:
        """
        lenbefore = len(df)
        df = df.loc[df["energy"] > min_energy, :]
        df = df.loc[df["energy"] < max_energy, :]
        print("Energy clipped to [", min_energy, ",", max_energy, "]")
        print("After clipping contains ", len(df), " out of ", lenbefore, " entries.")
        return df

    def load_files(self, file_name, max_num_files=1, start_at_file=0,
                   overwrite_particle_id=-10, given_columns=None):
        """
        Function that handles the loading of files.
        :param file_name:
        :param max_num_files:
        :param start_at_file:
        :param overwrite_particle_id:
        :param given_columns:
        :return:
        """

        if given_columns is None:
            colList2 = self.col_list_base_info + self.colList[self.info_col_num:]
        else:
            colList2 = given_columns + self.colList[self.info_col_num:]
            # print("12000: ",colList2)

        g_dataframe = pd.read_csv(
            file_name + str(start_at_file) + ".csv",
            sep=",",
            names=colList2,
            index_col=None,
            header=None)
        # g_dataframe['pID'] = g_dataframe['pID'].map(lambda x: 1)
        # header = None)
        # header=0)
        print("Loaded file ", file_name + str(start_at_file) + ".csv")
        if self.energy_filter:
            g_dataframe = self.clip_by_energy_df(g_dataframe, self.energy_filter_min, self.energy_filter_max)

        # energy, particleID, altitude, azimuth, core_x, core_y, h_first_int, x_max
        maxFileNum = start_at_file + max_num_files
        n = start_at_file + 1

        while n < maxFileNum:
            a_dataframe = pd.read_csv(
                file_name + str(n) + ".csv",
                sep=",",
                names=colList2)
            # header=None)
            # a_dataframe['pID'] = a_dataframe['pID'].map(lambda x: 1)

            g_dataframe = pd.concat([g_dataframe, a_dataframe], ignore_index=True, axis=0)
            print("Loaded file ", file_name + str(n) + ".csv")
            if self.energy_filter:
                g_dataframe = self.clip_by_energy_df(g_dataframe, self.energy_filter_min, self.energy_filter_max)
            n += 1
        if overwrite_particle_id != -10:
            g_dataframe.loc[:, "particleID"] = overwrite_particle_id

        # Add missing columns
        for i in range(len(self.colList[:self.info_col_num])):
            if self.colList[i] in colList2:
                pass
            else:
                g_dataframe.insert(i, self.colList[i], [0 for m in range(len(g_dataframe))])

        return g_dataframe

    def getWeights(self):
        """
        Returns a tensor of weights containing the relation between the amount of data of each class
        """

        self.gammaAmount = (self.c_dataframe["particleID"] == 1).sum()
        self.protonAmount = (self.c_dataframe["particleID"] == 0).sum()
        # print("ACC:G ", self.gammaAmount)
        # print("ACC:P ", self.protonAmount)
        if self.gammaAmount <= 0 or self.protonAmount <= 0:
            print("Warning: No data in one of the datasets, returning weights of 1.")
            return torch.tensor([1.0, 1.0])
        if self.gammaAmount > self.protonAmount:
            wlist = torch.tensor([self.gammaAmount / self.protonAmount, 1])
        else:
            wlist = torch.tensor([1, self.protonAmount / self.gammaAmount])
        print("Weights: ", wlist)

        return wlist

