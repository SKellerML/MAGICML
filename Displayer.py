import pickle

import time
from IPython.display import clear_output
from matplotlib import pyplot as plt
from ipywidgets import widgets
import Plotter as pl
import IPython.display as disp

import torch as torch
from torch.utils import data
import torch.nn as nn

import DataOption as DataOption
import MagicDataset as MD
import DataProcessingFunction as DPF

class Displayer(object):
    def __init__(self, datahandler, device="cpu", execute_net=None, regressor_column=None, restriction_list=None,
                 remap_function=None, class_names=("Proton", "Gamma"), min_col=-9):
        '''
        Displays Events, might need fixing idk

        :param dataset: pandas dataframe, not MagicDataset
        :param net:
        :param device:
        :param execute_net:
        '''

        self.min_col = min_col

        self.dataset = datahandler.c_dataframe
        if restriction_list is not None:
            self.dataset = self.dataset.loc[restriction_list, :]
            self.tmds = copy.deepcopy(datahandler)
            self.tmds.c_dataframe = self.dataset
            self.tmds = self.tmds.get_whole_dataset()
        else:
            self.tmds = datahandler.get_whole_dataset()  # MD.MagicDataset(self.dataset, regressor_column=regressor_column, data_option=data_option,
            # telescope_data_position_list=datahandler.telescope_data_position_list)
        self.currentRow = -1
        self.crow = None
        # self.crow = self.tmds[0]

        self.net = execute_net
        self.data_option = datahandler.data_option

        self.class_names = class_names

        self.mylist = []
        self.device = device

        # Next button
        self.bttn = widgets.Button(description="Next!")
        self.bttn.on_click(self.on_button_clicked_n)
        disp.display(self.bttn)
        self.bttp = widgets.Button(description="Previous!")
        self.bttp.on_click(self.on_button_clicked_p)
        disp.display(self.bttp)

        self.btts = widgets.Button(description="Skip!")
        self.btts.on_click(self.on_button_clicked_s)
        disp.display(self.btts)
        self.wtx = widgets.IntText(
            value=1,
            description='Skip:',
            disabled=False
        )

        # Save button
        self.bttn_save = widgets.Button(description="Save!")
        self.bttn_save.on_click(self.save)

        self.softm = nn.Softmax(dim=1)

        # self.stereo = False if data_option == MD.DataOption.mono or data_option == MD.DataOption.stereo_as_channel else True
        # print("Stereo: ",self.stereo)
        self.is_regressor_net = False  # datahandler
        if regressor_column is not None:
            self.is_regressor_net = True

        if remap_function is None:
            self.remap_function = lambda h: h
        else:
            self.remap_function = remap_function

    def on_button_clicked_s(self, b):
        self.currentRow += self.wtx.value
        self.create_new_image()

    def on_button_clicked_n(self, b):
        self.currentRow += 1
        self.create_new_image()

    def on_button_clicked_p(self, b):
        self.currentRow -= 1
        self.create_new_image()

    def create_new_image(self):
        self.crow = self.tmds[self.currentRow]

        # Remove Border, uncomment manually
        """
        t1 = torch.zeros(34,39)
        t2 = torch.zeros(34,39)
        print(len(self.crow["image1"][0]))
        for y in range(len(self.crow["image1"][0])):
            for x in range(len(self.crow["image1"][0][0])):
                t1[y][x] = -10 if self.crow["image1"][0][y][x] == 0 else self.crow["image1"][0][y][x]
                t2[y][x] = -10 if self.crow["image2"][0][y][x] == 0 else self.crow["image2"][0][y][x]
        """

        # print(self.crow["image1"][0])
        # print(self.crow)
        if self.data_option == DataOption.stereo_as_stereo:
            # self.crowt = self.crow["image1"].view(1, 1, 34, 39)

            t1 = self.crow["image1"][0]
            t2 = self.crow["image2"][0]
            pl.plotTensors2D([t1, t2], markerSize=65, showPadding=False)
            # [self.crow["image1"][0], self.crow["image2"][0]], markerSize=65, showPadding=False, )
        elif self.data_option == DataOption.it_channel_sep:
            print(self.crow)
            self.crowt = (self.crow["image1"])  # .view(1, 2, 40, 40))
            print(self.crowt)
            t1 = self.crow["image1"].cuda().float()
            t2 = GU.remap_44_tensor(t1[1]).cpu()
            t1 = GU.remap_44_tensor(t1[0]).cpu()
            pl.plotTensors2D([t1, t2], markerSize=65, showPadding=False, min_col=self.min_col)
        elif self.data_option == DataOption.mono:
            self.crowt = self.crow["image1"].view(1, 1, 34, 39)
            t1 = self.crow["image1"][0]
            pl.plotTensor2D(t1)
        # Datalist Options
        elif self.data_option == DataOption.datalist_stereo_as_stereo:
            # self.crowt = self.crow["image1"].view(1, 1, 34, 39)
            print(self.crow["image1"])
            t1 = self.remap_function(self.crow["image1"].float().to(self.device)).cpu()
            t2 = self.remap_function(self.crow["image2"].float().to(self.device)).cpu()
            pl.plotTensors2D([t1, t2], markerSize=65, showPadding=False, min_col=self.min_col)
            # [self.crow["image1"][0], self.crow["image2"][0]], markerSize=65, showPadding=False, )
        # NOT WORKING
        elif self.data_option == DataOption.datalist_stereo_as_channel:
            print(self.crow)
            self.crowt = self.remap_function(self.crow["image1"])  # .view(1, 2, 34, 39))
            print(self.crowt)
            t1 = self.remap_function(self.crow["image1"])
            t2 = t1[1]
            t1 = t1[0]
            pl.plotTensors2D([t1, t2], markerSize=65, showPadding=False)
        # NOT WORKING
        elif self.data_option == DataOption.datalist_mono:
            self.crowt = self.remap_function(self.crow["image1"])  # .view(1, 1, 34, 39)
            t1 = self.remap_function(self.crow["image1"][0])
            pl.plotTensor2D(t1)

        clear_output()
        # print(sum(self.hh[0].values))

        disp.display((self.bttn))
        disp.display(self.bttp)
        disp.display(self.wtx)
        disp.display(self.btts)
        print("Particle: ", self.crow["particleID"], " (", self.class_names[int(self.crow["particleID"])], ")")
        print("Max Value: ", self.crow['max_value'])
        # disp.display((self.bttn_save))
        print("Row: ", self.currentRow)
        print(self.tmds)
        print(self.dataset.loc[self.crow["pdRow"]]["energy"])

        if self.net is not None:
            if self.is_regressor_net == False:
                with torch.no_grad():
                    if self.data_option == DataOption.stereo_as_stereo or self.data_option == DataOption.datalist_stereo_as_stereo:
                        self.crowt1 = self.remap_function(self.crow["image1"].float().to(self.device)).unsqueeze(
                            0).unsqueeze(0)
                        self.crowt2 = self.remap_function(self.crow["image2"].float().to(self.device)).unsqueeze(
                            0).unsqueeze(0)  # .view(1, 1, 34, 39)
                        self.crowtm = self.crow["max_value"].float().to(self.device).unsqueeze(0).unsqueeze(0)
                        print("Maximum Value: ", self.crowtm)
                        outp = self.net(self.crowt1, self.crowt2, self.crowtm)


                    elif self.data_option == DataOption.stereo_as_channel:
                        outp = self.net(self.crowt.float().to(self.device))
                    else:
                        outp = self.net(self.crowt.float().to(self.device))
                    # outp = self.softm(outp)

                    print("Proton: {:1.3f} Gamma: {:1.3f}".format(outp[0][0], outp[0][1]))

                print("Row: ", self.currentRow)
                print("Particle: ", self.crow["particleID"], " (", self.class_names[int(self.crow["particleID"])], ")")
                isgamma = 1 if outp[0][1] > 0.5 else 0
                if isgamma == self.crow["particleID"]:
                    print(bcolors.OKGREEN + "RIGHT")
                else:
                    print(bcolors.FAIL + "WRONG")
            else:
                with torch.no_grad():
                    outp = self.net(self.crowt.float().to(self.device))
                    # outp = self.softm(outp)

                    print("Output: ", outp[0][0].item())

                print("Label: ", self.crow["label"])
                # isgamma = 1 if outp[0][1] > 0.5 else 0
                # if isgamma == self.crow["particleID"]:
                #    print(bcolors.OKGREEN + "RIGHT")
                # else:
                #    print(bcolors.FAIL + "WRONG")
        plt.show()

    def save(self, b):
        self.mylist.append(self.crow[0])
        print("saved ", self.crow[0])




class bcolors:
    HEADER = '\033[95m'
    OKBLUE = '\033[94m'
    OKGREEN = '\033[92m'
    WARNING = '\033[93m'
    FAIL = '\033[91m'
    ENDC = '\033[0m'
    BOLD = '\033[1m'
    UNDERLINE = '\033[4m'