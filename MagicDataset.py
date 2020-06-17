import torch as torch
from torch.utils import data
import copy
import hexagdly as hg
import pandas as pd
import numpy as np

import math as math

from matplotlib import pyplot as plt
import torch.optim as optim
import time as time

from sklearn import metrics as metrics

import DataOption as DataOption


class MagicDataset(data.Dataset):
    """ Magic dataset used for loading MAGIC Data with the PyTorch DataLoader """

    def __init__(self, datf, telescope_data_position_map, data_option=DataOption.stereo_tot_sep,
                 regressor_columns=None, pixnum=1039):  # csv_file, root_dir, transform=None):
        """
         Magic dataset used for loading MAGIC Data with the PyTorch DataLoader

        :param datf pandas dataframe: Dataframe containing MAGIC Data according to ...
        :param data_option: DataOption from DataOption.py
        :param telescope_data_position_map: Map containing the position of data inside the Dataframe (given by DataHandler)
        :param regressor_columns: List of Regressor Column Names used for regressiond tasks, None if Classification
        """

        if regressor_columns is not None:
            self.regressor_columns = regressor_columns
        else:
            self.regressor_columns = []


        self.data_option = data_option
        #print(self.data_option)
        if self.data_option == DataOption.stereo_tot_sep:
            self.itemFunction = self.item_stereo_tot_sep

        if self.data_option == DataOption.it_channel_sep:
            self.itemFunction = self.item_it_channel_sep

        if self.data_option == DataOption.channel_stereo_comb_sep:
            self.itemFunction = self.item_channel_stereo_comb_sep

        if self.data_option == DataOption.channel_tot_comb:
            self.itemFunction = self.item_channel_tot_comb

        self.dataframe = datf

        #self.tIndex = "1_" + str(self.sizeX - 1) + '|' + str(self.sizeY - 1)
        #self.tIndex2 = "2_" + str(self.sizeX - 1) + '|' + str(self.sizeY - 1)

        #if telescope_data_position_map is None:
        #    print("Error, No position List found")
        #    self.telescope_data_position_list = [("1_0|0", '1_' + str(self.sizeX-1) + '|' + str(self.sizeY-1))]
        #    self.telescope_data_position_list.append(("2_0|0",'2_' + str(self.sizeX - 1) + '|' + str(self.sizeY - 1)))
        #if telescope_data_position_list is None:

        #else:
        self.telescope_data_position_map = telescope_data_position_map

        self.dtype = "double"

    def __len__(self):
        return len(self.dataframe)

    def __getitem__(self, idx):
        return self.itemFunction(idx)

    def __get_data__(self, idx):
        row = self.dataframe.iloc[idx]

        # Read data from pandas series
        image1 = row.loc[self.telescope_data_position_map["i_M1"][0]: self.telescope_data_position_map["i_M1"][1]]
        image2 = row.loc[self.telescope_data_position_map["i_M2"][0]: self.telescope_data_position_map["i_M2"][1]]
        time_1 = row.loc[self.telescope_data_position_map["p_M1"][0]: self.telescope_data_position_map["p_M1"][1]]
        time_2 = row.loc[self.telescope_data_position_map["p_M2"][0]: self.telescope_data_position_map["p_M2"][1]]

        image1 = torch.tensor(image1.to_numpy(dtype=self.dtype))
        image2 = torch.tensor(image2.to_numpy(dtype=self.dtype))

        time_1 = torch.tensor(time_1.to_numpy(dtype=self.dtype))
        time_2 = torch.tensor(time_2.to_numpy(dtype=self.dtype))

        pID = row['particleID']
        minval = torch.tensor(row.iloc[-2])
        maxval = torch.tensor(row.iloc[-1])

        reg_col = []
        for regc in self.regressor_columns:

            reg_col.append(row.loc[regc])
        reg_col = torch.tensor(reg_col)

        return image1, image2, time_1, time_2, pID, maxval, minval, self.dataframe.index[idx], reg_col, row.loc['pointing_alt'], row.loc['pointing_az']

    def item_stereo_tot_sep(self, idx):

        image1, image2, time1, time2, pID, maxval, minval, pdRow, reg_col, palt, paz = self.__get_data__(idx)

        sample = {'image1': image1, 'image2': image2, 'time1': time1, 'time2': time2, 'particleID': pID,
                  'pdRow': pdRow, 'min_value': minval, 'max_value': maxval , 'regressor_columns':reg_col,
                  'pointing_alt': palt, 'pointing_az': paz}
        return sample

    def item_it_channel_sep(self, idx):
        image1, image2, time1, time2, pID, maxval, minval, pdRow, reg_col, palt, paz = self.__get_data__(idx)

        image1 = torch.cat([image1.unsqueeze(0), image2.unsqueeze(0)])#, dim=1)
        time1 = torch.cat([time1.unsqueeze(0), time2.unsqueeze(0)])#, dim=1)

        sample = {'image1': image1, 'time1': time1, 'particleID': pID,
                  'pdRow': pdRow, 'min_value': minval, 'max_value': maxval, 'regressor_columns': reg_col,
                  'pointing_alt': palt, 'pointing_az': paz}
        return sample

    def item_channel_stereo_comb_sep(self, idx):
        image1, image2, time1, time2, pID, maxval, minval, pdRow, reg_col, palt, paz = self.__get_data__(idx)

        image1 = torch.cat([time1.unsqueeze(0), image1.unsqueeze(0)])#, dim=1)
        image2 = torch.cat([time2.unsqueeze(0), image2.unsqueeze(0)])#, dim=1)

        sample = {'image1': image1, 'image2': image2, 'particleID': pID,
                  'pdRow': pdRow, 'min_value': minval, 'max_value': maxval , 'regressor_columns':reg_col,
                  'pointing_alt': palt, 'pointing_az': paz}
        return sample

    def item_channel_tot_comb(self, idx):
        image1, image2, time1, time2, pID, maxval, minval, pdRow, reg_col, palt, paz = self.__get_data__(idx)

        image1 = torch.cat([time1.unsqueeze(0), image1.unsqueeze(0), time2.unsqueeze(0), image2.unsqueeze(0)])#, dim=1)

        sample = {'image1': image1, 'particleID': pID,
                  'pdRow': pdRow, 'min_value': minval, 'max_value': maxval , 'regressor_columns':reg_col,
                  'pointing_alt': palt, 'pointing_az': paz}
        return sample
