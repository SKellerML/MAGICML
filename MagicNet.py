import torch as torch
from torch.utils import data
import copy
import numpy as np

import Displayer as Displayer
from matplotlib import pyplot as plt
import time as time
import math

from astropy.coordinates.angle_utilities import angular_separation
from astropy import units as u

import DataProcessingFunction as DPF
import DataOption as DataOption

from sklearn import metrics as metrics
import torch.nn as nn



class TrainInfoObj(object):
    """
    Object containing information about the training.
    """
    def __init__(self, lr, bs, ep, totaltime, accuracy_list, losslisttrain, losslisttest, saved_as="None"):
        self.lr = lr
        self.bs = bs
        self.ep = ep
        self.totaltime = totaltime
        self.test_loss = losslisttest
        self.train_loss = losslisttrain
        self.accuracy_list = accuracy_list



class MagicNet(object):
    """
    Class that handles the basic training and evaluation for regression and classification tasks.
    """


    def __init__(self, dHandler, useNet, trainSplit=0.8, device='cuda:0',
                 data_option=DataOption.datalist_stereo_as_stereo,
                 optimizer=None, regressor_columns=None, remap=False):

        self.net = useNet

        self.data_option = data_option
        self.dHandler = dHandler
        self.trainDF, self.testDF = dHandler.getTrainAndValidationDataset(trainSplit, 1 - trainSplit)

        # Assuming that we are on a CUDA machine, this should print a CUDA device:
        self.device = device

        if remap:
            # if self.data_option == DataOption.stereo_tot_sep:
            #    self.rmpfnc = DPF.remap_44_tensor_multiple
            # else:

            if self.data_option == DataOption.stereo_tot_sep:
                self.rmpfnc = DPF.remap_44_tensor_multiple
            else:
                self.rmpfnc = DPF.remap_44_tensor_multiple_channel_magic


        else:
            self.rmpfnc = lambda x: x

        self.optimizer = optimizer

        if regressor_columns is None:
            pass

        if self.data_option == DataOption.stereo_tot_sep:
            self.loop_Function = self.class_loop_stereo_tot_sep

        if self.data_option == DataOption.it_channel_sep:
            self.loop_Function = self.class_loop_it_channel_sep

        if self.data_option == DataOption.channel_stereo_comb_sep:
            self.loop_Function = self.class_loop_channel_stereo_comb_sep

        if self.data_option == DataOption.channel_tot_comb:
            self.loop_Function = self.class_loop_channel_tot_comb

        if regressor_columns is not None:
            self.regressor_columns = regressor_columns
            self.num_regressor_columns = len(self.regressor_columns)
        self.cnetstat = {}
        self.cnetstat['epoch'] = []
        self.cnetstat['testloss'] = []
        self.cnetstat['trainloss'] = []

        print(self.device)
        # self.roc = ROChandler()

    def to_net_stat(self, epoch, trainloss, testloss):
        self.cnetstat['epoch'] = epoch
        self.cnetstat['testloss'].append(testloss)
        self.cnetstat['trainloss'].append(trainloss)

    def load_net(self, filename="NetSD/temp_1", cuda_to_cpu=False):
        if cuda_to_cpu:
            self.net.load_state_dict(torch.load(filename, map_location='cpu'))
        else:
            self.net.load_state_dict(torch.load(filename))

    def save_net(self, filename="NetSD/temp_1"):
        torch.save(self.net.state_dict(), filename)

    def load_model(self, path, base_net, base_optimizer):
        d1 = DPF.load_model(path, base_net, base_optimizer)
        # do stuff
        self.net = d1['net']
        self.optimizer = d1['optimizer']
        self.cnetstat['epoch'] = d1['epoch']
        self.cnetstat['trainloss'] = d1['trainloss']
        self.cnetstat['testloss'] = d1['testloss']
        additional_info = d1['additional_info']
        print(additional_info)

    def save_model(self, path, additional_info=""):

        DPF.save_model(path, self.net, self.optimizer, self.cnetstat['epoch'],
                      self.cnetstat['trainloss'], self.cnetstat['testloss'], additional_info)
        print("Model saved as ", path, " at ", time.ctime())

    def getTestDataLoader(self, batch_size, shuffle=True):
        return data.DataLoader(self.testDF, batch_size=batch_size, shuffle=shuffle)

    def save_train_info_to_stat_file(self, train_info_object, saved_as="None", filename="netinfo.csv",
                                     additional_info="", max_epochs=16):
        netname = self.net.__class__.__name__

        try:
            f1 = open(filename, mode="r")
            f1.close()
        except:
            f1 = open(filename, mode="w")
            linestr = "name,learning rate,batch size,epochs, total time to train,"

            for n in range(max_epochs):
                linestr += "accuracy_epoch_" + str(n) + ","
            for n in range(max_epochs):
                linestr += "train_epoch_" + str(n) + ","
            for n in range(max_epochs):
                linestr += "test_epoch_" + str(n) + ","

            linestr += "additional info,saved as\n"
            f1.write(linestr)
            f1.close()

        with open(filename, mode="a") as file:
            linestr = netname + ","
            linestr += str(train_info_object.lr) + ","
            linestr += str(train_info_object.bs) + ","
            linestr += str(train_info_object.ep) + ","
            linestr += str(train_info_object.totaltime) + ","
            for n in range(max_epochs):
                linestr += str(train_info_object.accuracy_list[n]) + "," if n < train_info_object.ep else str(-100) + \
                                                                                                          ","
            for n in range(max_epochs):
                linestr += str(train_info_object.train_loss[n]) + "," if n < train_info_object.ep else str(-100) + ","
            for n in range(max_epochs):
                linestr += str(train_info_object.test_loss[n]) + "," if n < train_info_object.ep else str(-100) + ","

            linestr += additional_info + ","

            linestr += saved_as + "\n"
            file.write(linestr)

    def train_nn_regressor(self, learning_rate=None, batch_size=1, epochs=1, outputbatchnumber=-1, optim=None,
                           save_after_each_epoch_name=None):

        self.net.to(self.device)

        criterion = nn.MSELoss()  # weight=self.dHandler.getWeights().to(self.device))
        # criterion = nn.BCELoss()
        # optimizer = optim.Adagrad(self.net.parameters(),
        # lr=learningRate, lr_decay=0, weight_decay=0, initial_accumulator_value=0)
        if self.optimizer is None and optim is not None:
            self.optimizer = optim
        if self.optimizer is None:
            print("No optimizer defined. Exiting...")
            return False

        if learning_rate is not None:
            for param_group in self.optimizer.param_groups:
                param_group['lr'] = learning_rate

        trainloader = data.DataLoader(self.trainDF, batch_size=batch_size, shuffle=True, num_workers=4, drop_last=True)

        losslisttrain = []
        losslisttest = []
        acc_list = []
        time_before = time.time()

        if outputbatchnumber < 1:
            outputbatchnumber = math.round(len(self.trainDF) / batch_size)
            print("Ouputbatchnumber: ", outputbatchnumber)

        for epoch in range(epochs):  # loop over the dataset multiple times

            trainnum = 0
            lastloss = 0.0
            running_loss = 0.0
            totloss = 0.0
            for i, dat in enumerate(trainloader, 0):
                # print("Loading Time: ", t2-t1)
                # get the inputs
                self.net.zero_grad()
                # zero the parameter gradients
                self.optimizer.zero_grad()

                # Execute this part in extra function??

                outputs = self.loop_Function(dat)

                # labels = labels.view(-1)
                # print(outputs, " | ", labels)

                labels = dat['regressor_columns'].to(self.device)
                # print("Labels: ", labels)
                # print(outputs, " | ", labels)
                loss = criterion(outputs, labels)
                '''
                print()
                print(loss)
                print(outputs, " | ",labels)
                print()
                '''
                loss.backward()
                self.optimizer.step()

                # print statistics
                running_loss += loss.item()

                trainnum += 1  # len(inputs1)
                totloss += loss.item()
                # lastloss = running_loss / outputbatchnumber
                # get loss for test set with n mini batches
                # outputbatchnumber = 5
                if i % outputbatchnumber == outputbatchnumber - 1:  # print every 200 mini-batches
                    print('[{0}, {1}] loss: {2} Time passed: {3}'.format(epoch + 1, i + 1,
                                                                         running_loss / outputbatchnumber,
                                                                         time.time() - time_before))
                    lastloss = running_loss / outputbatchnumber
                    running_loss = 0.0

            lss = self.check_on_test_data_regressor(c_criterion=criterion,
                                                    numTestDataToCheck=outputbatchnumber * batch_size)
            print(lss)
            self.to_net_stat(epoch, lss, totloss / trainnum)

            losslisttrain.append(totloss / trainnum)  # lastloss)
            trainnum = 0
            totloss = 0.0
            losslisttest.append(lss)
            # acc_list.append(acc)

            # save model
            if save_after_each_epoch_name is not None:
                self.save_model(save_after_each_epoch_name)

        time_after = time.time()
        print('Finished Training, it took ', time_after - time_before, " seconds")

        ltr1, = plt.plot([q for q in range(epochs)], losslisttrain, label='train')
        lte1, = plt.plot([q for q in range(epochs)], losslisttest, label='test')

        print("Test Losslist: ", losslisttest)
        plt.xlabel("Epochs")
        plt.ylabel("Loss")
        plt.legend(handles=[ltr1, lte1])
        print("E1")

        tio = TrainInfoObj(learning_rate, batch_size, epochs, (time_after - time_before), acc_list, losslisttrain,
                           losslisttest)
        print("E2")
        return tio

    def plotloss(self):
        # print(self.cnetstat['testloss'])#.append(testloss)
        # plt.plot(self.cnetstat['testloss'])
        # self.cnetstat['trainloss'].append(trainloss)

        fig = plt.figure(figsize=[8, 6])
        ax1 = fig.add_subplot(111)
        ax1.plot(self.cnetstat['trainloss'], label="Train Loss")
        ax1.plot(self.cnetstat['testloss'], label="Test Loss")
        ax1.legend()

    def reg_eval(self, test_set=None, c_criterion=None):
        label_list, output_list = self.check_on_test_data_regressor(test_set=test_set, c_criterion=c_criterion,
                                                                    returnListsValues=True)
        # print("Labels: ", label_list)
        # print("Outputs: ", output_list)

        numcol = len(label_list)

        delta_list = [[] for i in range(0, numcol)]

        # calculate deltas
        for i in range(len(label_list[0])):
            for col in range(numcol):
                delta_list[col].append((label_list[col][i] - output_list[col][i]) ** 2)

        # print("Deltas: ", delta_list)

        fig = plt.figure(figsize=[12, 5])
        axl = [fig.add_subplot(1, numcol, i + 1) for i in range(0, numcol)]

        colmap = DPF.get_discrete_colormap(2)

        title_list = self.dHandler.regressor_columns
        for col in range(numcol):
            # print(DPF.get_log_bins(delta_list[col], 100))
            # axl[col].hist(delta_list[col], bins=DPF.get_log_bins(delta_list[col], 100) , color = [colmap(0)])

            # axl[col].set_xscale('log')

            axl[col].hist(delta_list[col], bins=100, color=[colmap(0)])
            # axl[col].set_xlim([0,0.02])

            axl[col].set_title(title_list[col])
            axl[col].set_xlabel("Delta value")
            axl[col].set_ylabel("Counts value")  # dN/dOmega

    def reg_eval_theta_altazflists(self, label_list, output_list):

        # print("Labels: ", label_list)
        # print("Outputs: ", output_list)

        numcol = len(label_list)

        delta_list = [[] for i in range(0, numcol)]
        off_angles = []
        thetasquared = []
        # calculate deltas
        for i in range(len(label_list[0])):
            delta_list[0].append((label_list[0][i] * 10 - output_list[0][i] * 10) ** 2)
            delta_list[1].append((label_list[1][i] * 10 - output_list[1][i] * 10) ** 2)

            off_angle = angular_separation((label_list[1][i] * 10 - 5.0) * u.deg, (label_list[0][i] * 10 - 5.0) * u.deg,
                                           (output_list[1][i] * 10 - 5.0) * u.deg,
                                           (output_list[0][i] * 10 - 5.0) * u.deg)
            # thetasquared.append(off_angle**2)
            thetasquared.append(off_angle.to_value(u.deg) ** 2)
        # print("Deltas: ", delta_list)

        fig_width_pt = 345.0 / 2  # * 0.5
        # Convert from pt to inches
        inches_per_pt = 1 / 72.27
        # Golden ratio to set aesthetic figure height
        golden_ratio = (5 ** .5 - 1) / 2

        # Figure width in inches
        fig_width_in = 4.7747 / 2  # fig_width_pt * inches_per_pt
        # Figure height in inches
        fig_height_in = fig_width_in * golden_ratio

        fig1 = plt.figure(figsize=[fig_width_in, fig_height_in])
        ax1 = fig1.add_subplot(1, 1, 1)
        fig2 = plt.figure(figsize=[fig_width_in, fig_height_in])
        ax2 = fig2.add_subplot(1, 1, 1)
        fig3 = plt.figure(figsize=[fig_width_in, fig_height_in])
        ax3 = fig3.add_subplot(1, 1, 1)

        colmap = DPF.get_discrete_colormap(12)

        from matplotlib import rc
        # rc('font', **{'family': 'sans-serif', 'sans-serif': ['Helvetica']})
        ## for Palatino and other serif fonts use:
        rc('font', **{'family': 'serif', 'serif': ['Computer Modern'], 'size': 10})
        rc('text', usetex=True)

        title_list = [r"Altitude", r"Azimuth", r"$\Theta^2$"]  # self.dHandler.regressor_columns
        col = 0
        weights = np.ones_like(delta_list[col]) / float(len(delta_list[col]))
        ax1.hist(delta_list[col], bins=100, color=[colmap(4)], weights=weights)
        ax1.set_title(title_list[col])
        ax1.set_xlabel(r"Delta value [deg]")
        ax1.set_ylabel(r"Counts")  # dN/dOmega
        ax1.set_yscale('log')
        ax1.set_xlim([0, 0.25])

        col = 1
        weights = np.ones_like(delta_list[col]) / float(len(delta_list[col]))
        ax2.hist(delta_list[col], bins=100, color=[colmap(4)], weights=weights)
        # axl[col].set_xlim([0, 0.02])

        ax2.set_title(title_list[col])
        ax2.set_xlabel(r"Delta value [deg]")
        ax2.set_ylabel(r"Counts")  # dN/dOmega
        ax2.set_yscale('log')
        ax2.set_xlim([0, 0.25])

        col = 2
        # Theta squared

        weights = np.ones_like(thetasquared) / float(len(thetasquared))
        a = ax3.hist(thetasquared, bins=300, color=[colmap(4)], weights=weights)
        # axl[col].set_xlim([0, 0.02])

        # ax3.set_title(title_list[col])
        ax3.set_xlabel(r"$\Theta^2$ [deg$^2$]")
        ax3.set_ylabel(r"Normalized Counts")  # dN/dOmega
        ax3.set_yscale('log')

        f1 = 0
        f2 = 0
        mmx = sum(a[0]) * 0.68
        mmx2 = sum(a[0]) * 0.95
        d = 0
        d2 = 0
        for i, m in enumerate(a[0]):
            f1 += 1
            f2 += m
            if f2 > mmx and d == 0:
                d = a[1][i] + (a[1][i + 1] - a[1][i]) * ((mmx + m - f2) / a[0][i])
            if f2 > mmx2:
                d2 = a[1][i] + (a[1][i + 1] - a[1][i]) * ((mmx2 + m - f2) / a[0][i])
                break

        ax3.axvline(d, color='k', linestyle='dashed', linewidth=1)
        ax3.text(d * 1.55, 0.2, "68 \% containment")
        ax3.axvline(d2, color='k', linestyle='dashed', linewidth=1)
        ax3.text(d2 * 1.15, 0.03, "95 \% containment")
        ax3.set_xlim([0, 0.25])

        fig1.savefig("alt_dr.png", dpi=300, bbox_inches='tight')
        fig2.savefig("az_dr.png", dpi=300, bbox_inches='tight')
        fig3.savefig("ts_dr.png", dpi=300, bbox_inches='tight')

        # plt.hist2d(label_list[0][0], label_list[0][1])

    def reg_eval_theta_altaz(self, test_set=None, c_criterion=None):
        label_list, output_list = self.check_on_test_data_regressor(test_set=test_set, c_criterion=c_criterion,
                                                                    returnListsValues=True)
        # print("Labels: ", label_list)
        # print("Outputs: ", output_list)

        numcol = len(label_list)

        delta_list = [[] for i in range(0, numcol)]
        off_angles = []
        thetasquared = []
        # calculate deltas
        for i in range(len(label_list[0])):
            delta_list[0].append((label_list[0][i] * 10 - output_list[0][i] * 10))
            delta_list[1].append((label_list[1][i] * 10 - output_list[1][i] * 10))

            off_angle = angular_separation((label_list[1][i] * 10 - 5.0) * u.rad, (label_list[0][i] * 10 - 5.0) * u.rad,
                                           (output_list[1][i] * 10 - 5.0) * u.rad,
                                           (output_list[0][i] * 10 - 5.0) * u.rad)
            thetasquared.append(off_angle ** 2)

        # print("Deltas: ", delta_list)

        fig_width_pt = 345.0 / 2  # * 0.5
        # Convert from pt to inches
        inches_per_pt = 1 / 72.27
        # Golden ratio to set aesthetic figure height
        golden_ratio = (5 ** .5 - 1) / 2

        # Figure width in inches
        fig_width_in = 4.7747 / 2  # fig_width_pt * inches_per_pt
        # Figure height in inches
        fig_height_in = fig_width_in * golden_ratio

        fig1 = plt.figure(figsize=[fig_width_in, fig_height_in])
        ax1 = fig1.add_subplot(1, 1, 1)
        fig2 = plt.figure(figsize=[fig_width_in, fig_height_in])
        ax2 = fig2.add_subplot(1, 1, 1)
        fig3 = plt.figure(figsize=[fig_width_in * 2, fig_height_in * 2])
        ax3 = fig3.add_subplot(1, 1, 1)

        colmap = DPF.get_discrete_colormap(12)

        title_list = [r"Altitude", r"Azimuth", r"$\Theta^2$"]  # self.dHandler.regressor_columns

        from matplotlib import rc
        # rc('font', **{'family': 'sans-serif', 'sans-serif': ['Helvetica']})
        ## for Palatino and other serif fonts use:
        rc('font', **{'family': 'serif', 'serif': ['Computer Modern'], 'size': 10})
        rc('text', usetex=True)
        print("EEEEDO")
        col = 0
        ax1.hist(delta_list[col], bins=15, color=[colmap(4)])
        ax1.set_title(title_list[col])
        ax1.set_xlabel(r"Delta value [deg]")
        ax1.set_ylabel(r"Counts")  # dN/dOmega
        ax1.set_yscale('log')
        ax1.set_xlim([0, 0.5])

        col = 1
        ax2.hist(delta_list[col], bins=15, color=[colmap(4)])
        # axl[col].set_xlim([0, 0.02])

        ax2.set_title(title_list[col])
        ax2.set_xlabel(r"Delta value [deg]")
        ax2.set_ylabel(r"Counts")  # dN/dOmega
        ax2.set_yscale('log')
        ax2.set_xlim([0, 0.5])

        col = 2
        # Theta squared
        ax3.hist(thetasquared, bins=25, color=[colmap(4)])
        # axl[col].set_xlim([0, 0.02])

        ax3.set_title(title_list[col])
        ax3.set_xlabel(r"Delta value [deg]")
        ax3.set_ylabel(r"Counts")  # dN/dOmega
        ax3.set_yscale('log')
        ax3.set_xlim([0, 0.5])

        fig1.savefig("alt_dr.png", dpi=300, bbox_inches='tight')
        fig2.savefig("az_dr.png", dpi=300, bbox_inches='tight')
        fig3.savefig("ts_dr.png", dpi=300, bbox_inches='tight')

        # plt.hist2d(label_list[0][0], label_list[0][1])

    def check_on_test_data_regressor(self, test_set=None, returnLists=False, c_criterion=None, numTestDataToCheck=-1,
                                     threshold_val=0.5, optim=None, returnListsValues=False):
        '''
        Tests on test data

        Parameters:
        - testSet: (Magic)Dataset to use, None if class test set should be used
        - returnLists: changes return values, see below

        returns: percentage of right classification if returnLists == False
                 list of labes and list of results if returnLists == True
                 percentage of right classification and loss if criterion != None
        '''

        if c_criterion is None:
            criterion = nn.MSELoss()
        else:
            criterion = c_criterion

        # if self.optimizer is None and optim is not None:
        #    self.optimizer = optim
        # if self.optimizer is None:
        #    print("No optimizer defined. Exiting...")
        #    return False

        # Check on Test Data
        correct = 0
        total = 0
        threshold = 0.5
        labelsList = []
        scoresList = []
        losslist = []
        testnum = 0
        totloss = 0.0

        # sfmx = nn.Softmax(dim=1)

        self.net.to(self.device)

        c_batch_size = 8

        if test_set is None:
            testDLoader = data.DataLoader(self.testDF, batch_size=c_batch_size, shuffle=True, num_workers=8,
                                          drop_last=True)
        else:
            testDLoader = data.DataLoader(test_set, batch_size=c_batch_size, shuffle=True, num_workers=8,
                                          drop_last=True)

        # testDLoader = data.DataLoader(self.trainDF, batch_size=c_batch_size, shuffle=True, num_workers=8, drop_last=True)

        c_running_loss = 0.0
        cTestDataNum = 0

        wrongPredictedList = []
        rightPredictedList = []

        label_list = [[] for i in range(self.num_regressor_columns)]
        output_list = [[] for i in range(self.num_regressor_columns)]

        proton_class_as_gamma = 0
        gamma_class_as_proton = 0

        close_threshold = 0.01
        closenumlist = [0 for i in range(self.num_regressor_columns)]
        closelist = [[] for i in range(self.num_regressor_columns)]
        wronglist = [[] for i in range(self.num_regressor_columns)]
        # criterion = nn.L1Loss()

        with torch.no_grad():
            for dat in testDLoader:

                outputs = self.loop_Function(dat)

                rowId = dat['pdRow']
                labels = dat['regressor_columns'].to(self.device)

                loss = criterion(outputs, labels)
                # print("ABCD: ",outputs)

                losslist.append(loss.item())
                totloss += loss.item()
                testnum += 1  # len(inputs1)

                # print("Labels: ",labels)
                # print("Outputs: ",outputs)
                for i in range(c_batch_size):
                    for n in range(len(labels[i])):
                        labelsList.append(labels[i][n])
                        scoresList.append(outputs[i][n])

                    for n in range(self.num_regressor_columns):
                        if abs(labels[i][n].item() - outputs[i][n].item()) < labels[i][n].item() * close_threshold:
                            closenumlist[n] += 1
                            closelist[n].append(rowId[i].item())
                        else:
                            wronglist[n].append(rowId[i].item())
                        label_list[n].append(labels[i][n].item())
                        output_list[n].append(outputs[i][n].item())

        print("Total Loss: ", float(sum(losslist) / len(losslist)))
        for n in range(self.num_regressor_columns):
            print(n, "Closer than ", close_threshold * 100, " %: ", closenumlist[n] / (len(losslist) * 4) * 100, "%",
                  closenumlist[n], " | ", len(losslist))
        # print("Closer than 2 10%: ", closenum2/(len(losslist)*4)*100, "%")
        # print(scoresList)

        if returnLists:
            return closelist, wronglist
        elif returnListsValues:
            return label_list, output_list
        else:
            return totloss / testnum  # closenumlist


        # criterion = nn.BCELoss()
        # optimizer = optim.Adagrad(self.net.parameters(),
        # lr=learningRate, lr_decay=0, weight_decay=0, initial_accumulator_value=0)

    def classifier_loop(self, dat):

        input_max_v = dat['max_value'].to(self.device)
        input_min_v = dat['min_value'].to(self.device)

        inputs1 = dat['image1'].float().to(self.device)
        inputs1 = DPF.remap_44_tensor_multiple(inputs1)  # _channel

        inputs1_time = dat['time1'].float().to(self.device)
        inputs1_time = DPF.remap_44_tensor_multiple(inputs1_time)

        # zero the parameter gradients

        # Execute net, depending on whether stereo or mono data is presented
        if self.data_option == DataOption.stereo_as_stereo or self.data_option == DataOption.datalist_stereo_as_stereo:
            inputs2 = dat['image2'].float().to(self.device)
            inputs2 = DPF.remap_44_tensor_multiple(inputs2)

            inputs2_time = dat['time2'].float().to(self.device)
            inputs2_time = DPF.remap_44_tensor_multiple(inputs2_time)
            # print("BB8: ", inputs1, inputs2, input_max_v)

            outputs = self.net(inputs1, inputs2, inputs1_time, inputs2_time,
                               input_max_v)  # , inputs1_time, inputs2_time, input_max_v)
        elif self.data_option == DataOption.stereo_as_channel:
            # print(inputs1.shape)
            # print("AEONS")
            outputs = self.net(inputs1)  # , inputs1_time)
        else:
            outputs = self.net(inputs1)  # , inputs1_time)

        outputs = outputs.float()
        return outputs

        # vars to add

    # pointing
    # impact parameter
    # remap function yes or no

    def class_loop_stereo_tot_sep(self, dat):

        inputs1 = self.rmpfnc(dat['image1'].float().to(self.device))
        inputs2 = self.rmpfnc(dat['image2'].float().to(self.device))

        inputs1_time = self.rmpfnc(dat['time1'].float().to(self.device))
        inputs2_time = self.rmpfnc(dat['time2'].float().to(self.device))

        vmax = dat['max_value'].unsqueeze(1).float().to(self.device)
        vmin = dat['min_value'].unsqueeze(1).float().to(self.device)

        return self.net(inputs1, inputs2, inputs1_time, inputs2_time, vmax, vmin)

    def class_loop_it_channel_sep(self, dat):
        inputs1 = self.rmpfnc(dat['image1'].float().to(self.device))

        inputs1_time = self.rmpfnc(dat['time1'].float().to(self.device))

        vmax = dat['max_value'].unsqueeze(1).float().to(self.device)
        vmin = dat['min_value'].unsqueeze(1).float().to(self.device)

        return self.net(inputs1, inputs1_time, vmax, vmin)

    def class_loop_channel_stereo_comb_sep(self, dat):
        inputs1 = self.rmpfnc(dat['image1'].float().to(self.device))
        inputs2 = self.rmpfnc(dat['image2'].float().to(self.device))

        vmax = dat['max_value'].unsqueeze(1).float().to(self.device)
        vmin = dat['min_value'].unsqueeze(1).float().to(self.device)

        return self.net(inputs1, inputs2, vmax, vmin)

    def class_loop_channel_tot_comb(self, dat):

        inputs1 = self.rmpfnc(dat['image1'].float().to(self.device))

        vmax = dat['max_value'].unsqueeze(1).float().to(self.device)
        vmin = dat['min_value'].unsqueeze(1).float().to(self.device)

        return self.net(inputs1, vmax, vmin)

    def train_nn_classifier(self, learning_rate=0.001, batch_size=1, epochs=1, outputbatchnumber=-1, optim=None,
                            save_after_each_epoch_name=None):

        '''
        # select classifier loop based on data option
        if self.data_option == :
            classifier_loop
        elif self.data_option == :

        else:
            print("Data Option not defined for function.")
            return
        '''

        self.net.to(self.device)

        # Define loss function
        criterion = nn.CrossEntropyLoss(weight=self.dHandler.getWeights().to(self.device))
        # criterion = nn.L1Loss(weight=self.dHandler.getWeights().to(self.device))
        # Define optimization function

        if optim is not None:
            self.optimizer = optim
        if self.optimizer is None:
            print("No optimizer defined. Exiting...")
            return False

        # Define data loader function
        trainloader = data.DataLoader(self.trainDF, batch_size=batch_size, shuffle=True, num_workers=16, drop_last=True)

        # Define lists to save statistics
        losslisttrain = []
        losslisttest = []
        acc_list = []

        time_before = time.time()

        # Set output information
        if outputbatchnumber < 1:
            outputbatchnumber = math.round(len(self.trainDF) / batch_size)
            print("Ouputbatchnumber: ", outputbatchnumber)

        for epoch in range(epochs):  # loop over the dataset multiple times

            lastloss = 0.0
            running_loss = 0.0
            # Loop through epochs
            for i, dat in enumerate(trainloader, 0):
                # Get data and send to GPU
                labels = dat['particleID'].long()
                labels = labels.to(self.device)

                self.optimizer.zero_grad()
                self.net.zero_grad()

                outputs = self.loop_Function(dat)

                # Calculate loss
                loss = criterion(outputs, labels)

                # Update parameters
                loss.backward()
                self.optimizer.step()

                # print statistics
                running_loss += loss.item()
                if i % outputbatchnumber == outputbatchnumber - 1:  # print every 200 mini-batches
                    print('[{0}, {1}] loss: {2} Time passed: {3}'.format(epoch + 1, i + 1,
                                                                         running_loss / outputbatchnumber,
                                                                         time.time() - time_before))
                    lastloss = running_loss / outputbatchnumber
                    running_loss = 0.0

            # Add values to statistics
            losslisttrain.append(lastloss)
            acc, lss = self.check_on_test_data_classifier(c_criterion=criterion,
                                                          numTestDataToCheck=outputbatchnumber * batch_size)
            losslisttest.append(lss)
            acc_list.append(acc)
            self.to_net_stat(epoch, lastloss, lss)

            # save model
            if save_after_each_epoch_name is not None:
                self.save_model(save_after_each_epoch_name)

        time_after = time.time()
        print('Finished Training, it took ', time_after - time_before, " seconds")

        ltr1, = plt.plot([q for q in range(epochs)], losslisttrain, label='train')
        lte1, = plt.plot([q for q in range(epochs)], losslisttest, label='test')

        print("Test Losslist: ", losslisttest)
        plt.xlabel("Epochs")
        plt.ylabel("Loss")
        plt.legend(handles=[ltr1, lte1])
        # print("E1")
        tio = TrainInfoObj(learning_rate, batch_size, epochs, (time_after - time_before), acc_list, losslisttrain,
                           losslisttest)
        # print("E2")
        return tio

    def check_on_test_data_classifier(self, test_set=None, returnLists=False, c_criterion=None, numTestDataToCheck=-1,
                                      threshold_val=0.5, return_prediction_list=False):
        '''
        Tests on test data

        :param test_set: (Magic)Dataset to use, None if class test set should be used
        :param returnLists: changes return values, see below
        :param c_criterion:
        :param numTestDataToCheck:
        :param threshold_val:
        :param stereo:
        :param return_prediction_list:
        :return: percentage of right classification if returnLists == False
                 list of labes and list of results if returnLists == True
                 percentage of right classification and loss if criterion != None
                 return_prediction_list - returns list of right classifications and wrong classifications
        '''

        # Check on Test Data
        correct = 0
        total = 0
        threshold = 0.5
        labelsList = []
        scoresList = []
        sfmx = nn.Softmax(dim=1)

        self.net.to(self.device)

        c_batch_size = 4

        if test_set is None:
            testDLoader = data.DataLoader(self.testDF, batch_size=c_batch_size, shuffle=True, drop_last=True)
        else:
            testDLoader = data.DataLoader(test_set, batch_size=c_batch_size, shuffle=True, drop_last=True)

        c_running_loss = 0.0
        cTestDataNum = 0

        wrongPredictedList = []
        rightPredictedList = []
        proton_class_as_gamma = 0
        gamma_class_as_proton = 0

        with torch.no_grad():
            for dat in testDLoader:
                labels = dat['particleID'].long()
                labels = labels.to(self.device)

                rowId = dat['pdRow']

                outputs = self.loop_Function(dat)

                outputs = sfmx(outputs)

                # print("Lab: ", labels)
                # for outp in outputs:
                # print("Out: ", outputs)
                # resvec = labels-(2*outputs)

                for i in range(0, len(labels)):
                    labelsList.append(labels[i].item())

                    scoresList.append(outputs[i][1].item())  # select only gamma value from output [proton, gamma]

                    class_res = 1 if outputs[i][1].item() > threshold_val else 0

                    if labels[i].item() == class_res:
                        correct += 1
                        rightPredictedList.append((rowId[i].item(), labels[i].item(), outputs[i][1].item()))
                    else:
                        # print(outputs[i][1].item(), "   Label: ", labels[i], " RowID: ", rowId, " Class result: ", class_res)
                        # check in which way it was classified wrong
                        if labels[i].item() == 0:  # Proton
                            proton_class_as_gamma += 1
                        elif labels[i].item() == 1:  # Proton
                            gamma_class_as_proton += 1
                        wrongPredictedList.append((rowId[i].item(), labels[i].item(), outputs[i][1].item()))
                    # print("UE22 ", i)
                    '''
                    if outputs[i][1] > threshold and labels[i] == 1:
                        correct += 1
                        rightPredictedList.append((rowId[i], labels[i], outputs[i][1]))
                    elif outputs[i][0] > threshold and labels[i] == 0:
                        correct += 1
                        rightPredictedList.append((rowId[i], labels[i], outputs[i][1]))
                    else:
                        wrongPredictedList.append((rowId[i], labels[i], outputs[i][1]))
                    '''
                    total += 1

                cTestDataNum += 1 * c_batch_size
                if (numTestDataToCheck > 0 and cTestDataNum >= numTestDataToCheck):
                    break

                if (c_criterion != None):
                    c_loss = c_criterion(outputs, labels)
                    c_running_loss += c_loss.item()

                # TP += resvec.count(-1)
                # FP += resvec.count(-2)
                # TN += resvec.count(0)
                # FN += resvec.count(1)
            # if(b_addToROC):
            #    self.roc.addToROC(TP, FP, TN, FN, Threshold)
        # print(scoresList, "Labels: \n", labelsList)
        """
        rlist = np.random.randint(0, len(wrongPredictedList), size=50)
        for h in rlist:
            print("Wrong: " + str(wrongPredictedList[h]))
        rlist = np.random.randint(0, len(rightPredictedList), size=50)
        for h in rlist:
            print("Right: " + str(rightPredictedList[h]))
        """
        print("Correct: ", correct, " Total: ", total)
        print('Accuracy of the network on the test images: {0}'.format(100 * correct / total))
        print("Gammas classified as protons: ", gamma_class_as_proton * 2 / total)
        print("Protons classified as gammas: ", proton_class_as_gamma * 2 / total)
        if return_prediction_list:
            return rightPredictedList, wrongPredictedList

        if (c_criterion != None):
            c_running_loss /= numTestDataToCheck
            c_running_loss *= c_batch_size
            print('Loss: {0}'.format(c_running_loss))
            return (correct / total), c_running_loss

        if (returnLists):
            return labelsList, scoresList, (correct / total)
        else:
            return (correct / total)

    def plot_tmva_classifier_cumulative(self, test_set=None, num_bins=100, data_option=DataOption.mono):
        # Create Zeta Score Distribution
        llist, slist, accuracy = self.check_on_test_data_classifier(test_set, returnLists=True)

        print(llist)
        print(slist)

        thr_list_signal = []
        thr_list_background = []
        xbl = []
        total = len(llist)

        cl_lst = []
        # Split labels and scores list
        for i in range(len(llist)):
            cl_lst.append(slist[i])

        # print(thr_list_signal)
        # print(thr_list_background)
        # plt.plot(xbl, thr_list_signal, label="Signal")
        # plt.plot(xbl, thr_list_background, label="Background")

        fig = plt.figure(1)
        ax1 = fig.add_subplot(111)
        ax1.hist(cl_lst, label="Cumulative", histtype=u'step', bins=num_bins, linewidth=2, density=1)
        ax1.set_yscale("log")
        ax1.set_xlabel("Threshold")
        ax1.set_ylabel("Percentage")
        plt.legend()
        # plt.show()

    def plot_tmva_classifier(self, test_set=None, num_bins=100,
                             data_option=DataOption.mono):  # , show_cumulative=False):
        '''

        :param test_set: MagicDataset
        :param num_bins:
        :param data_option:
        :return:
        '''
        # Create Zeta Score Distribution
        llist, slist, accuracy = self.check_on_test_data_classifier(test_set, returnLists=True)

        # thr_list_signal = []
        # thr_list_background = []
        # xbl = []
        # total = len(llist)
        # cumulative_lst = []
        signal_lst = []
        background_lst = []
        # Split labels and scores list
        for i in range(len(llist)):
            if llist[i] == 1:
                signal_lst.append(slist[i])
            else:
                background_lst.append(slist[i])
        #    cumulative_lst.append(slist[i])

        # print(thr_list_signal)
        # print(thr_list_background)
        # plt.plot(xbl, thr_list_signal, label="Signal")
        # plt.plot(xbl, thr_list_background, label="Background")

        fig = plt.figure(1)
        ax1 = fig.add_subplot(111)
        ax1.hist(signal_lst, label="Signal", histtype=u'step', bins=num_bins, linewidth=2, density=1)
        ax1.hist(background_lst, label="Background", histtype=u'step', bins=num_bins, linewidth=2, density=1)
        # if show_cumulative:
        #    ax1.hist(cumulative_lst, label="Cumulative", histtype=u'step', bins=num_bins, linewidth=2, density=2)
        ax1.set_yscale("log")
        ax1.set_xlabel("Threshold")
        ax1.set_ylabel("Percentage")
        plt.legend()
        # plt.show()

    def create_roc_classifierlists(self, llist, slist, accuracy,
                                   savename="p.png", sizedivider=1):
        """
        Creates ROC Curve and return accuracy on test data

        test_set: MagicDataset
        """
        fig_width_pt = 345.0 / sizedivider  # * 0.5
        # Convert from pt to inches
        inches_per_pt = 1 / 72.27
        # Golden ratio to set aesthetic figure height
        golden_ratio = (5 ** .5 - 1) / 2

        # Figure width in inches
        fig_width_in = 4.7747 / sizedivider  # fig_width_pt * inches_per_pt
        # Figure height in inches
        fig_height_in = fig_width_in * golden_ratio

        colmap = DPF.get_discrete_colormap(12)

        from matplotlib import rc
        # rc('font', **{'family': 'sans-serif', 'sans-serif': ['Helvetica']})
        ## for Palatino and other serif fonts use:
        rc('font', **{'family': 'serif', 'serif': ['Computer Modern'], 'size': 10})
        rc('text', usetex=True)

        fpr, tpr, thresholds = metrics.roc_curve(llist,
                                                 slist, )  # , pos_label=None, sample_weight=None, drop_intermediate=True)
        auc = metrics.roc_auc_score(llist, slist, average='macro', sample_weight=None, max_fpr=None)
        print("A: ", auc)
        # auc_score = metrics.roc_auc_score(llist, slist)
        # print(thresholds)
        fig = plt.figure(figsize=(fig_width_in, fig_height_in))
        ax = fig.add_subplot(111)
        ax.plot([0, 1], [0, 1])
        a1 = ax.scatter(fpr[1:], tpr[1:], c=thresholds[1:])
        # ax.plot(fpr[1:], tpr[1:])
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        ax.set_xlabel("False Positive Rate")
        ax.set_ylabel("True Positive Rate")
        a1.set_clim([min(thresholds[1:]), max(thresholds[1:])])
        fig.colorbar(a1)
        fig.savefig(savename, dpi=300, bbox_inches='tight')
        return accuracy
        # data.DataLoader(testSet, batch_size=batch_size, shuffle=True)

    def plot_tmva_classifierlists(self, llist, slist, accuracy, num_bins=100, data_option=DataOption.mono,
                                  savename="p.png", sizedivider=1):  # , show_cumulative=False):
        '''

        :param test_set: MagicDataset
        :param num_bins:
        :param data_option:
        :return:
        '''
        fig_width_pt = 345.0 / sizedivider  # * 0.5
        # Convert from pt to inches
        inches_per_pt = 1 / 72.27
        # Golden ratio to set aesthetic figure height
        golden_ratio = (5 ** .5 - 1) / 2

        # Figure width in inches
        fig_width_in = 4.7747 / sizedivider  # fig_width_pt * inches_per_pt
        # Figure height in inches
        fig_height_in = fig_width_in * golden_ratio

        colmap = DPF.get_discrete_colormap(12)

        colmap = DPF.get_discrete_colormap(12)

        from matplotlib import rc
        # rc('font', **{'family': 'sans-serif', 'sans-serif': ['Helvetica']})
        ## for Palatino and other serif fonts use:
        rc('font', **{'family': 'serif', 'serif': ['Computer Modern'], 'size': 10})
        rc('text', usetex=True)
        # thr_list_signal = []
        # thr_list_background = []
        # xbl = []
        # total = len(llist)
        # cumulative_lst = []
        signal_lst = []
        background_lst = []
        # Split labels and scores list
        for i in range(len(llist)):
            if llist[i] == 1:
                signal_lst.append(slist[i])
            else:
                background_lst.append(slist[i])
        #    cumulative_lst.append(slist[i])

        # print(thr_list_signal)
        # print(thr_list_background)
        # plt.plot(xbl, thr_list_signal, label="Signal")
        # plt.plot(xbl, thr_list_background, label="Background")

        fig = plt.figure(figsize=(fig_width_in, fig_height_in))
        ax1 = fig.add_subplot(111)
        ax1.hist(signal_lst, label="Signal", histtype=u'step', bins=num_bins, linewidth=2, density=1, color=colmap(0))
        ax1.hist(background_lst, label="Background", histtype=u'step', bins=num_bins, linewidth=2, density=1,
                 color=colmap(4))
        # if show_cumulative:
        #    ax1.hist(cumulative_lst, label="Cumulative", histtype=u'step', bins=num_bins, linewidth=2, density=2)
        ax1.set_yscale("log")
        ax1.set_xlabel("Classification Value")
        ax1.set_ylabel("Normalized Counts")
        plt.legend(loc=9)
        fig.savefig(savename, dpi=300, bbox_inches='tight')
        # plt.show()

    def create_roc_classifier(self, test_set=None, stereo=False):
        """
        Creates ROC Curve and return accuracy on test data

        test_set: MagicDataset
        """
        llist, slist, accuracy = self.check_on_test_data_classifier(test_set, returnLists=True)
        fpr, tpr, thresholds = metrics.roc_curve(llist,
                                                 slist, )  # , pos_label=None, sample_weight=None, drop_intermediate=True)
        auc = metrics.roc_auc_score(llist, slist, average='macro', sample_weight=None, max_fpr=None)
        print("A: ", auc)
        # auc_score = metrics.roc_auc_score(llist, slist)
        # print(thresholds)
        fig = plt.figure(figsize=(4, 4))
        ax = fig.add_subplot(111)
        ax.plot([0, 1], [0, 1])
        a1 = ax.scatter(fpr[1:], tpr[1:], c=thresholds[1:])
        # ax.plot(fpr[1:], tpr[1:])
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        ax.set_xlabel("False Positive Rate")
        ax.set_ylabel("True Positive Rate")
        a1.set_clim([min(thresholds[1:]), max(thresholds[1:])])
        fig.colorbar(a1)
        fig.savefig("ROC_High.png")

        return accuracy
        # data.DataLoader(testSet, batch_size=batch_size, shuffle=True)

    def show_wrong_classifications(self, data_handler, test_set=None):

        lists_pred_1 = self.check_on_test_data_classifier(test_set, return_prediction_list=True)
        # lists_pred_1
        rls1 = []
        for i in lists_pred_1[1]:
            rls1.append(i[0])
        dx = copy.deepcopy(data_handler)
        dx.c_dataframe = dx.c_dataframe.loc[rls1]

        Displayer.Displayer(dx, execute_net=self.net, data_option=self.data_option, device=self.device,
                      restriction_list=rls1)