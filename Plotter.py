import pandas as pd
import torch as torch
import numpy as np
from matplotlib import pyplot as plt
import matplotlib

# global parameters for plotting
colmap = "viridis"
# afmhot
# viridis
# _r inverts

import GAN_Utils as GU

def plot_events_as_image(event1, event2):
    img1 = torch.tensor(event1.dl1.tel[1].image)  # .to("cuda:0")
    img1 = GU.remap_44_tensor(img1)
    img2 = torch.tensor(event2.dl1.tel[2].image)  # .to("cuda:0")
    img2 = GU.remap_44_tensor(img2)

    plotTensors2D([img1,img2], markerSize=75)


def maxInTensor(tensor):
    return max(max(tensor, key=lambda p: max(p)))#.item()


def minInTensor(tensor):
    return min(min(tensor, key=lambda p: min(p)))



def plotTensor2D_specialaec(tensor, axW=None, maxvalue=None, minvalue=None, showPadding=False, move_2nd_rows_upwards=True,
                 marker="H", markerSize=-1, add_colorbar=False, log_color=False):


    #img2 = img2.cpu()

    if maxvalue is None:
        maxvalue = maxInTensor(tensor)
    if minvalue is None:
        minvalue = minInTensor(tensor)

    if axW is None:
        if add_colorbar:
            fig = plt.figure(figsize=[12, 10])
        else:
            fig = plt.figure(figsize=[10.5, 10])
        ax = fig.add_subplot(111)
    else:
        ax = axW

    if markerSize < 0:
        if marker == "H":
            markerSize = 255
        elif marker == "s":
            markerSize = 45
        else:
            markerSize = 100
    if (not showPadding):
        tensor = GU.fix_padding(tensor,10)
    
        
    tensor = tensor.cpu()
    
    marker_style = dict(color=None, linestyle=':', marker='o')
    
    for y in range(0, len(tensor)):
        # for x in range(0,len(tensor[y])):
        # cval = [tensor[y][x].item()]
        xp = []
        yp = []
        col = []
        col2 = []
        for x in range(len(tensor[y])):
            xp.append(x)
            yp.append(y)
            if tensor[y][x] > 0:
                col.append(tensor[y][x].item())
                col2.append(-10)
            else:
                col.append(-10)
                if tensor[y][x] < -5:
                    col2.append(tensor[y][x].item())
                else:
                    col2.append(10)
            #xp = [h for h in range(0, len(tensor[y]))]
            #yp = [y for h in range(0, len(tensor[y]))]
        for n, p in enumerate(yp, 0):
            if  move_2nd_rows_upwards and n % 2 == 0:
                yp[n] += 0.5
        i1 = ax.scatter(xp, yp, c=col, vmin=minvalue, cmap=colmap, vmax=maxvalue, marker=marker,
                        s=markerSize)#, norm=matplotlib.colors.LogNorm())#, linewidth=0.3 , edgecolor = "white")
        i2 = ax.scatter(xp, yp, **marker_style)#vmin=minvalue, cmap=colmap, vmax=maxvalue, marker=marker,
                        #s=markerSize, edgecolor = "face")
    if add_colorbar and axW is None:
        fig.colorbar(i1, cmap=colmap)
        #plt.colorbar()
    #ax.show()
    #ax.axis('off')
    ax.set_xticks([])
    ax.set_yticks([])
    if showPadding:
        i1.cmap.set_under(color='r', alpha=1.0)
    else:
        i1.cmap.set_under(color='w', alpha = 0.0) # removes everything lower than minvalue

    if showPadding:
        i2.cmap.set_under(color='r', alpha=1.0)
    else:
        i2.cmap.set_under(color='w', alpha = 0.0) # removes everything lower than minvalue

        
    return i1



def plotTensor2D(tensor, axW=None, maxvalue=None, minvalue=None, showPadding=False, move_2nd_rows_upwards=True,
                 marker="H", markerSize=-1, add_colorbar=False, log_color=False, title="", colormap='viridis'):


    #img2 = img2.cpu()

    if maxvalue is None:
        maxvalue = maxInTensor(tensor)
    if minvalue is None:
        minvalue = minInTensor(tensor)

    if axW is None:
        if add_colorbar:
            fig = plt.figure(figsize=[12, 10])
        else:
            fig = plt.figure(figsize=[10.5, 10])
        ax = fig.add_subplot(111)
    else:
        ax = axW

    if markerSize < 0:
        if marker == "H":
            markerSize = 255
        elif marker == "s":
            markerSize = 45
        else:
            markerSize = 100
    if (not showPadding):
        tensor = GU.fix_padding(tensor,10)
    
        
    tensor = tensor.cpu()
    for y in range(0, len(tensor)):
        # for x in range(0,len(tensor[y])):
        # cval = [tensor[y][x].item()]
        xp = [h for h in range(0, len(tensor[y]))]
        yp = [y for h in range(0, len(tensor[y]))]
        for n, p in enumerate(yp, 0):
            if  move_2nd_rows_upwards and n % 2 == 0:
                yp[n] += 0.5
        i1 = ax.scatter(xp, yp, c=tensor[y], vmin=minvalue, cmap=colormap, vmax=maxvalue, marker=marker,
                            s=markerSize)#, norm=matplotlib.colors.LogNorm())#, linewidth=0.3 , edgecolor = "white")
    if add_colorbar and axW is None:
        fig.colorbar(i1, cmap=colormap)
        #plt.colorbar()
    #ax.show()
    #ax.axis('off')
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_xlim([-1,39])
    ax.set_ylim([2,38])
    ax.set_title(title)
    if showPadding:
        i1.cmap.set_under(color='r', alpha=1.0)
    else:
        i1.cmap.set_under(color='w', alpha = 0.0) # removes everything lower than minvalue

    return i1


def plotTensors2D(tensorList, showPadding = False, marker="H", markerSize=-1,  move_2nd_rows_upwards=True, min_col=-9,
                  ax1=None, ax2=None, fig=None):
    if ax1 is None or ax2 is None or fig is None:
        fig = plt.figure(figsize=[12, 5.5])
        ax1 = fig.add_subplot(121)
        ax2 = fig.add_subplot(122)

    # Get min/max value of tensors to define color scheme
    mv1 = maxInTensor(tensorList[0])
    mv2 = max(mv1, maxInTensor(tensorList[1]))
    minv1 = minInTensor(tensorList[0])
    minv2 = min(minv1, minInTensor(tensorList[1]))
    # remove pixels under minv2, only if padding is not wanted
    #minv2 = -2
    if(not showPadding):
        minv2 = max(min_col,minv2)


    i1 = plotTensor2D(tensorList[0], axW=ax1, maxvalue=mv2, minvalue=minv2, marker=marker, markerSize=markerSize,
                      move_2nd_rows_upwards=move_2nd_rows_upwards)
    i2 = plotTensor2D(tensorList[1], axW=ax2, maxvalue=mv2, minvalue=minv2, marker=marker, markerSize=markerSize,
                      move_2nd_rows_upwards=move_2nd_rows_upwards)
    #ax1.set_facecolor((1.0, 0.47, 0.42, 0.0))
    #ax2.set_facecolor((1.0, 0.47, 0.42, 0.0))
    # create own subplot for colorbar
    fig.subplots_adjust(right=0.9)
    cbar_ax = fig.add_axes([0.95, 0.15, 0.025, 0.7])
    fig.colorbar(i1, cax=cbar_ax, cmap=colmap)

def plotTensors2D_seperate_colorbars(tensorList, showPadding = False, marker="H", markerSize=75):
    fig = plt.figure(figsize=[15, 5.5])
    ax1 = fig.add_subplot(121)
    ax2 = fig.add_subplot(122)

    # Get min/max value of tensors to define color scheme
    mv1 = maxInTensor(tensorList[0])
    mv2 = max(mv1, maxInTensor(tensorList[1]))
    minv1 = minInTensor(tensorList[0])
    minv2 = min(minv1, minInTensor(tensorList[1]))
    # remove pixels under minv2, only if padding is not wanted
    if(not showPadding):
        minv2 = max(-9,minv2)

    i1 = plotTensor2D(tensorList[0], axW=ax1, marker=marker, markerSize=markerSize)#, maxvalue=mv2, minvalue=minv2)
    i2 = plotTensor2D(tensorList[1], axW=ax2, marker=marker, markerSize=markerSize)#, maxvalue=mv2, minvalue=minv2)

    fig.colorbar(i1, ax=ax1)
    fig.colorbar(i2, ax=ax2)

    # create own subplot for colorbar
    #fig.subplots_adjust(right=0.9)
    #cbar_ax = fig.add_axes([0.95, 0.15, 0.025, 0.7])
    #fig.colorbar(i1, cax=cbar_ax, cmap=colmap)

def plotListAsTensor2D(dataList, showPadding = False, axW=None, device="cuda:0"):
    if axW is None:
        fig = plt.figure(figsize=[11.5, 10])
        axW = fig.add_subplot(111)


    image = torch.tensor(dataList)
    image = GU.remap_44_tensor(image.to(device).float()).cpu()

    #image = image.view(sizeY, sizeX)
    #return plotTensor2D(image, showPadding=showPadding, axW=axW)
    i1 = plotTensor2D(image, showPadding=showPadding, axW=axW)
    fig.colorbar(i1, ax=axW)

def plotListsAsTensor2D(dataList1, dataList2, showPadding = False, axW=None, device="cuda:0", seperate_colorbars=False):

    #image1 = torch.tensor(dataList1)
    image1 = GU.remap_44_tensor(dataList1.to(device).float()).cpu()

    #image2 = torch.tensor(dataList2)
    image2 = GU.remap_44_tensor(dataList2.to(device).float()).cpu()

    if seperate_colorbars:
        i1 = plotTensors2D_seperate_colorbars([image1, image2], showPadding=showPadding, markerSize=75)
    else:
        i1 = plotTensors2D([image1, image2], showPadding=showPadding, markerSize=75)

def plot_tensors_with_timing_dataframe(series, showPadding = False, marker="H", markerSize=75):
    
    img1 = torch.tensor(series[9:1048].to_numpy())
    img2 = torch.tensor(series[1048:2087].to_numpy())
    time1 = torch.tensor(series[2087:3126].to_numpy())
    time2 = torch.tensor(series[3126:4165].to_numpy())
                   
    #img1 = GU.remap_44_tensor(img1)
    #img2 = GU.remap_44_tensor(img2)
    #time1 = GU.remap_44_tensor(time1)
    #time2 = GU.remap_44_tensor(time2)
        
    plotListsAsTensor2D(img1, img2, showPadding = False, axW=None, device="cuda:0", seperate_colorbars=False)
    plotListsAsTensor2D(time1, time2, showPadding = False, axW=None, device="cuda:0", seperate_colorbars=False)
              
def plot_from_datahandler(dh, rowID = 0, showPadding = False, plot_time = False, marker="H", markerSize=75, ax1 = None,
                          ax2 = None, fig=None):
    
    row = dh.c_dataframe.iloc[rowID]
    
    img1 = torch.tensor(row.iloc[dh.info_col_num : dh.pixnum + dh.info_col_num]).float().to("cuda:0")
    img2 = torch.tensor(row.iloc[dh.pixnum + dh.info_col_num : 2*dh.pixnum + dh.info_col_num]).float().to("cuda:0")

    #img1 = img1 + torch.ones(img1.shape).cuda().float()#*0.5
    #img2 = img2 + torch.ones(img2.shape).cuda().float()#*0.5

    img1 = GU.remap_44_tensor(img1)
    img2 = GU.remap_44_tensor(img2)

    img1 = GU.fix_padding(img1, scale=10)
    #print("AFFEL")
    img2 = GU.fix_padding(img2, scale=10)

    #img1 = img1.cpu()
    #img2 = img2.cpu()
    #print("GAFFEL"
    plotTensors2D([img1, img2], showPadding=showPadding, markerSize=markerSize, min_col=0, marker=marker, ax1 = ax1,
                          ax2 = ax2, fig = fig)

    if plot_time:
        timg1 = torch.tensor(row.iloc[2*dh.pixnum + dh.info_col_num : 3*dh.pixnum + dh.info_col_num]).float().to("cuda:0")
        timg2 = torch.tensor(row.iloc[3*dh.pixnum + dh.info_col_num : 4*dh.pixnum + dh.info_col_num]).float().to("cuda:0")

        timg1 = GU.remap_44_tensor(timg1)
        timg2 = GU.remap_44_tensor(timg2)

        timg1 = GU.fix_padding(timg1, scale=10)
        timg2 = GU.fix_padding(timg2, scale=10)

        timg1 = timg1.cpu()
        timg2 = timg2.cpu()

        plotTensors2D([timg1, timg2], showPadding=showPadding, markerSize=markerSize, min_col=0, marker=marker)


def plot_from_datahandler_AEC(dh, rowID=0, showPadding=False, plot_time=False, marker="H", markerSize=75,
                              sim=False, added=False, ped=False):

    if sim:
        row = dh.sim_dataframe.iloc[rowID]
    elif added:
        row = dh.added_dataframe.iloc[rowID]
    elif ped:
        row = dh.ped_dataframe.iloc[rowID]
    else:
        row = dh.added_dataframe.iloc[rowID]

    img1 = torch.tensor(row.iloc[dh.info_col_num: dh.pixnum + dh.info_col_num]).float().to("cuda:0")
    img2 = torch.tensor(row.iloc[dh.pixnum + dh.info_col_num: 2 * dh.pixnum + dh.info_col_num]).float().to("cuda:0")

    img1 = GU.remap_44_tensor(img1)
    img2 = GU.remap_44_tensor(img2)

    img1 = GU.fix_padding(img1, scale=10)
    img2 = GU.fix_padding(img2, scale=10)

    img1 = img1.cpu()
    img2 = img2.cpu()

    plotTensors2D([img1, img2], showPadding=showPadding, markerSize=markerSize, min_col=0, marker=marker)



def plot_from_MagicDataset_AEC(md, rowID=0, showPadding=False, marker="H", markerSize=75,
                               sim=False, added=False, ped=False):

    if sim:
        name1 = 'image1_sim'
        name2 = 'image2_sim'
    elif added:
        name1 = 'image1_added'
        name2 = 'image2_added'
    elif ped:
        name1 = 'image1_ped'
        name2 = 'image2_ped'
    else:
        name1 = 'image1_added'
        name2 = 'image2_added'

    img1 = GU.remap_44_tensor( md[name1].float().cuda() )
    img2 = GU.remap_44_tensor( md[name2].float().cuda() )

    img1 = GU.fix_padding(img1, scale=10)
    img2 = GU.fix_padding(img2, scale=10)

    img1 = img1.cpu()
    img2 = img2.cpu()

    plotTensors2D([img1, img2], showPadding=showPadding, markerSize=markerSize, min_col=0, marker=marker)




def plotMagicTelescopeImagesFromPaddedDataLoaderTensorObject(tensor, showPadding=False):
    # "image", batch object #, telescope number, channel, rows
    plotTensors2D([tensor[0][0], tensor[1][0]], showPadding=showPadding)
