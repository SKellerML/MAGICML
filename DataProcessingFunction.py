# Torch imports
import torch as torch
from torch.utils import data
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
# HexagDLy import
import hexagdly as hg
import copy
import numpy as np
import pandas as pd
import math
import pickle

from matplotlib import pyplot as plt
import matplotlib as mpl

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


def get_log_bins(l1, nbins):
    bins = np.geomspace(min(l1), max(l1), nbins)
    return bins


def get_discrete_colormap(num, colormapstr='viridis'):
    cmap = plt.get_cmap(colormapstr)
    colco = [cmap(i) for i in range(0, cmap.N, int(cmap.N / (num - 1)))]
    colco.append(cmap(cmap.N))
    cmap2 = mpl.colors.LinearSegmentedColormap.from_list(
        'Custom cmap', colco, num)
    return cmap2


def save_model(path, net, optimizer, epoch, trainloss=0.0, testloss=0.0, additional_info=""):
    d1 = {'epoch': epoch, 'model_state_dict': net.state_dict(), 'optimizer_state_dict': optimizer.state_dict(),
          'trainloss': trainloss, 'testloss': testloss, "additional_info": additional_info}
    # print("QQ1: ",d1)
    torch.save(d1, path)


def load_model(path, base_net, base_optimizer):
    checkpoint = torch.load(path)
    base_net.load_state_dict(checkpoint['model_state_dict'])
    base_optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    epoch = checkpoint['epoch']
    additional_info = checkpoint['additional_info']
    testloss = checkpoint['testloss']
    trainloss = checkpoint['trainloss']

    # base_net.eval()
    # - or -
    base_net.train()

    return {"net": base_net, 'optimizer': base_optimizer, 'epoch': epoch, 'trainloss': trainloss,
            'testloss': testloss, "additional_info": additional_info}


def get_image_from_datahandler(h, imgid):
    return h.get_whole_dataset()[imgid]


def process_data(x):
    n2 = max(x.iloc[8:2086])
    x.iloc[8:2086] = x.iloc[8:2086].div(n2)
    return x


def process_data2(x):
    x_a = x[13:2091]

    x_array2 = x_a / max(x_a)
    x_array2 = x_array2 - 0.5 - min(x_array2) / 2
    x_array2 = x_array2 * 1 / max(x_array2)
    x_array2 = pd.concat([x[0:9], x_array2])

    return x_array2


def process_data_gauss(x, colnum, pixnum):
    x_a = x[colnum: colnum + 2 * pixnum]
    x_t = x[colnum + 2 * pixnum: colnum + 4 * pixnum]
    t_data = pd.Series([min(x_a), max(x_a)])

    # x_a = np.log10(x_a+100)

    mean = np.mean(x_a, None)
    std = np.std(x_a, None)

    x_a -= mean
    x_a /= std

    x_tarray2 = (((x_t - 10) * 1) / (60 - 0))  # (((value - old_min) * new_range) / (old_max - old_min) - new_min)
    x_array2 = pd.concat([x[:colnum], x_a, x_tarray2, t_data])

    return x_array2


def process_data3_tc(x):
    x_a = x[9:2087]
    x_t = x[2087:4165]

    t_data = pd.Series([min(x_a), max(x_a)])

    # OldRange = (max(x_a) - min(x_a))
    # NewRange = 2#(1 - -1)
    x_array2 = (((x_a - min(x_a)) * 2) / (max(x_a) - min(x_a))) - 1
    x_t2 = (((x_t - 10) * 1) / (60 - 10)) - 0

    x_array2 = pd.concat([x[0:9], x_array2])
    x_array2 = pd.concat([x_array2, x_t2])
    x_array2 = pd.concat([x_array2, t_data])
    return x_array2


def process_data3(x, info_col_num, pixnum):
    x_a = x[info_col_num:info_col_num + 2 * pixnum]

    x_t = x[info_col_num + 2 * pixnum:info_col_num + 4 * pixnum]

    t_data = pd.Series([min(x_a), max(x_a)])

    # OldRange = (max(x_a) - min(x_a))
    # NewRange = 2#(1 - -1)
    x_array2 = (((x_a - min(x_a)) * 2) / (max(x_a) - min(x_a))) - 1
    x_array2 = pd.concat([x[0:info_col_num], x_array2])
    x_array2 = pd.concat([x_array2, x_t, t_data])
    return x_array2


def process_data4(x):
    x_a = x[9:2087]
    x_t = x[info_col_num + 2 * pixnum:info_col_num + 4 * pixnum]

    t_data = pd.Series([min(x_a), max(x_a)])

    # OldRange = (max(x_a) - min(x_a))
    # NewRange = 2#(1 - -1)
    x_array2 = (((x_a - -1) * 2) / (2500 - -1)) - 1
    x_tarray2 = (((x_a - -1) * 2) / (2500 - -1)) - 1
    x_array2 = pd.concat([x[0:9], x_array2])
    x_array2 = pd.concat([x_array2, x_tarray2, t_data])
    return x_array2


def process_data3_i_image(x_a, colnum, pixnum):
    # x_a = x[9:2087]
    # xs = x_a.shape
    # print(x_a.shape)
    # x_a=x_a.view(-1,40*40)

    # t_data = [min(x_a),max(x_a)]

    # OldRange = (max(x_a) - min(x_a))
    # NewRange = 2#(1 - -1)
    for i, x_loop in enumerate(x_a):
        x_a[i] = (((x_loop - min(x_loop)) * 1) / (max(x_loop) - min(x_loop)))
    # x_a=x_a.view(xs)
    return x_a


'''
def process_data3_i(x, info_col_num, pixnum):
    x_a = x[info_col_num:info_col_num + 2 * pixnum]
    x_t = x[info_col_num + 2 * pixnum:info_col_num + 4 * pixnum]

    t_data = pd.Series([min(x_a),max(x_a)])

    #OldRange = (max(x_a) - min(x_a))
    #NewRange = 2#(1 - -1)
    x_array2 = (((x_a - min(x_a)) * 1) / (max(x_a) - min(x_a)))
    x_array2 = pd.concat([x[0:info_col_num], x_array2])
    x_array2 = pd.concat([x_array2, x_t, t_data])
    return x_array2
'''


def process_data3_i(x, colnum, pixnum):
    x_a = x[colnum: colnum + 2 * pixnum]
    x_t = x[colnum + 2 * pixnum: colnum + 4 * pixnum]
    t_data = pd.Series([min(x_a), max(x_a)])

    # OldRange = (max(x_a) - min(x_a))
    # NewRange = 2#(1 - -1)
    x_array2 = (((x_a - min(x_a)) * 1) / (max(x_a) - min(x_a)))
    x_tarray2 = (((x_t - 10) * 1) / (60 - 0))  # (((value - old_min) * new_range) / (old_max - old_min) - new_min)
    x_array2 = pd.concat([x[:colnum], x_array2, x_tarray2, t_data])

    return x_array2


def process_data4_i(x, colnum, pixnum):
    x_a = x[colnum: colnum + 2 * pixnum]
    x_t = x[colnum + 2 * pixnum: colnum + 4 * pixnum]
    t_data = pd.Series([min(x_a), max(x_a)])

    # OldRange = (max(x_a) - min(x_a))
    # NewRange = 2#(1 - -1)
    x_array2 = (((x_a - -1) * 1) / (2500))
    x_tarray2 = (((x_t - 10) * 1) / (60 - 0))  # (((value - old_min) * new_range) / (old_max - old_min) - new_min)
    x_array2 = pd.concat([x[:colnum], x_array2, x_tarray2, t_data])

    return x_array2


def process_data4_nt(x):
    x_a = x[9:2087]
    x_t = x[info_col_num + 2 * pixnum:info_col_num + 4 * pixnum]

    t_data = pd.Series([min(x_a), max(x_a)])

    # OldRange = (max(x_a) - min(x_a))
    # NewRange = 2#(1 - -1)
    x_array2 = (((x_a - -1) * 2) / (2500 - -1)) - 1
    x_array2 = pd.concat([x[0:9], x_array2])
    x_array2 = pd.concat([x_array2, t_data])
    return x_array2


def process_data4_i_nt(x, colnum, pixnum):
    x_a = x[colnum:colnum + 2 * pixnum]
    t_data = pd.Series([min(x_a), max(x_a)])

    # OldRange = (max(x_a) - min(x_a))
    # NewRange = 2#(1 - -1)
    x_array2 = (((x_a - -1) * 1) / (2500))
    x_array2 = pd.concat([x[0:colnum], x_array2, t_data])
    return x_array2


def process_data4_timing(x, colnum, pixnum):
    x_a = x[colnum + 2 * pixnum: colnum + 4 * pixnum]

    t_data = pd.Series([min(x_a), max(x_a)])

    # OldRange = (max(x_a) - min(x_a))
    # NewRange = 2#(1 - -1)
    x_array2 = (((x_a - -1) * 1) / (2500 - -1))
    x_array2 = pd.concat([x[0:colnum], x_array2])
    x_array2 = pd.concat([x_array2, t_data, x[colnum + 2 * pixnum:]])
    return x_array2


def transform_angle_deg(y):
    return y / 360.0


def transform_angle_deg_90(y):
    # y = y*360/(2*math.pi)

    return y / 35


def transform_angle_deg_delta(y):
    # max/min of 5 degree
    # y = y*360/(2*math.pi)
    # (((value - old_min) * new_range) / (old_max - old_min) - new_min)
    y = (((y + 5) * 1) / (5 + 5) - 0)
    return y


def transform_energy(y):
    old_min = 10
    old_max = 10000
    new_min = 0
    return ((y - old_min) / (old_max - old_min))

    # OldRange = (max(x_a) - min(x_a))
    # NewRange = 2#(1 - -1)
    # old_min = 0
    # new_range = 1
    # old_max = 360
    # new_min = 0
    # (((value - old_min) * new_range) / (old_max - old_min) - new_min)


def transform_distance(y):
    old_min = -1000
    old_max = 1000
    new_min = 0
    return ((y - old_min) / (old_max - old_min))


def transform_campos(y):
    old_min = -0.5
    old_max = 0.5
    new_min = 0
    return (((y - old_min) * 1) / (old_max - old_min) - new_min)


def transform_distance_long(y):
    old_min = 0
    old_max = 100000
    new_min = 0
    return ((y - old_min) / (old_max - old_min))


def process_data5(x, colnum):
    x_a = x[0:colnum]

    for a in range(0, colnum):
        if a in (3, 8):  # 3: Event Azimuth, Pointing Azimuth
            x_a.iloc[a] = transform_angle_deg(x_a.iloc[a])
        elif a == 1:  # Energy
            x_a.iloc[a] = x_a.iloc[a]  # transform_energy(x_a.iloc[a])
        elif a == 0:  # particle ID
            if x_a.iloc[a] == -13:
                x_a.iloc[a] = 0
        elif a in (4, 5):  # Core_x, Core_y
            x_a.iloc[a] = transform_distance(x_a.iloc[a])
        elif a == 6:  # h first interaction
            x_a.iloc[a] = transform_distance_long(x_a.iloc[a])
        elif a in (9, 10):  # Pos in Camera, x/y
            x_a.iloc[a] = transform_campos(x_a.iloc[a])
        elif a in (2, 7):  # 2: event altitude, 7: pointing altitutude
            x_a.iloc[a] = transform_angle_deg_90(x_a.iloc[a])
        elif a in (11, 12):  # delta alt, delta az
            x_a.iloc[a] = transform_angle_deg_delta(x_a.iloc[a])

    """
    0 'particleID', 
    1 'energy', 
    2 'altitude', 
    3 'azimuth', 
    4 'core_x', 
    5 'core_y', 
    6 'h_first_int', 
    7 'pointing_alt', 
    8 'pointing_az', 
    9 'pos_in_cam_x', 
    10 'pos_in_cam_y', 
    11 'event_tel_alt', 
    12 'event_tel_az'
    """

    x_array2 = pd.concat([x_a, x[colnum:]])

    return x_array2


def reverse_process_data3(x_a, t_min, t_max):
    x_array2 = (x_a + 1) * (t_max - t_min) / 2 + t_min
    return x_array2


def noise_input(size, scale=1.0):
    ten = torch.rand(size).to(device)
    return ten * scale


def remap(image, dict_tdc, padding=-10):
    llx = [[padding for i in range(0, dict_tdc[-1])] for n in range(0, dict_tdc[-2])]

    for i, n in enumerate(image, 0):
        ct = dict_tdc[i]

        llx[ct[1]][ct[0]] = n

    return llx


from os.path import expanduser

home = expanduser("~")


def load_magic_hex_dict():
    return load_obj("MAGICDict1_changed")


def load_obj(name):
    with open(home + '/CTASoftware/PT1/obj/' + name + '.pkl', 'rb') as f:
        return pickle.load(f)


nid2 = 0


def add_to(f, background_scale, df, nid):
    global nid2
    # print(background_scale, nid+nid2)
    f[9:2087] += df.iloc[nid + nid2, 9:2087] * background_scale
    nid2 += 1
    if nid2 + nid >= len(df):
        nid2 = 0
    return f


def add_to_dataframe(base_df, additional_df, background_scale=1, start_id_in_additional_df=0):
    """
    Adds additional_df to base_df, but only in range 8:2086 (2 magic images, spiral pixel)
    :param base_df:
    :param additional_df:
    :param background_scale:
    :param start_id_in_additional_df:
    :return:
    """
    print("Adding Dataframes")
    global nid2
    nid2 = 0
    return base_df.apply(add_to, axis=1, args=(background_scale, additional_df, start_id_in_additional_df))


ddct = load_magic_hex_dict()

unmap_tensor = torch.zeros([34 * 39, 1039])
for i in range(1039):
    l1 = ddct[i]
    unmap_tensor[l1[1] * 39 + l1[0]][i] = 1
unmap_tensor = unmap_tensor.to(device)

unmap_tensor44 = torch.zeros([40 * 40, 1039])
for i in range(1039):
    l1 = ddct[i]
    unmap_tensor44[l1[1] * (39 + 1) + l1[0] + 3][i] = 1
unmap_tensor44 = unmap_tensor44.to(device)

map_tensor44 = torch.zeros([1039, 40 * 40])
for i in range(1039):
    l1 = ddct[i]
    map_tensor44[i][(l1[1] + 3) * (39 + 1) + l1[0]] = 1
map_tensor44 = map_tensor44.to(device)

padding_minus_1_tensor_44 = torch.mm(torch.ones((1, 1039)).to(device), map_tensor44).view(40, 40) - 1
padding_minus_1_tensor_44 = padding_minus_1_tensor_44.to(device)


def fix_padding(image, scale=1):
    return image + padding_minus_1_tensor_44 * scale


def remap44(image, dict_tdc, padding=-10):
    # print(image.shape)
    lllx = []
    for k in range(0, len(image)):
        llx = [[padding for i in range(0, 40)] for n in range(0, 40)]

        for i, n in enumerate(image[k], 0):
            ct = dict_tdc[i]

            llx[ct[1] + 3][ct[0]] = n
        lllx.append(llx)
    return lllx


def remap_magic44(x):
    '''

    :param x:
    :return: remapped tensor
    '''
    a1 = remap44(x, ddct, padding=-1)
    # print(a1)
    return torch.tensor(a1).view([len(a1), 1, len(a1[0]), len(a1[0][0])])


def remap_44_tensor(image):
    return torch.mm(image.view(1, -1), map_tensor44).view(40, 40).to(device)


# for channel
map_tensor44_channel = torch.zeros(2078, 40 * 40 * 2)
map_tensor44_channel[0:1039, 0:40 * 40] = map_tensor44
map_tensor44_channel[1039:2078, 40 * 40:40 * 40 * 2] = map_tensor44
map_tensor44_channel = map_tensor44_channel.to(device)


# torch.cat((map_tensor44,map_tensor44))

def remap_44_tensor_channel(image):
    return torch.mm(image.view(1, -1), map_tensor44_channel).view(2, 40, 40).to(device)


def remap_44_tensor_channel_magic(image):
    return torch.mm(image.view(1, -1), map_tensor44).view(1, 40, 40).to(device)


def remap_44_tensor_multiple_channel_magic(t):
    llx = torch.zeros([len(t), t.size(1), 40, 40]).to(device)
    for n, te in enumerate(t):
        # loop through channels
        c = 0  # Channel
        # print("Te: ",te.shape)
        for c in range(te.shape[0]):
            llx[n][c] = remap_44_tensor(te[c])

    # print(llx)
    return llx


def remap_44_tensor_multiple(t):
    llx = torch.zeros([len(t), 1, 40, 40]).to(device)
    for n, te in enumerate(t):
        llx[n][0] = remap_44_tensor(te)
    # print(llx)
    return llx


def remap_44_tensor_multiple_channel(t):
    llx = torch.zeros([len(t), t.size(1), 40, 40]).to(device)
    for n, te in enumerate(t):
        llx[n] = remap_44_tensor_channel(te)
    # print(llx)
    return llx


def remap_44_tensor_multiple_dlobj(t):
    num_channel = t.size(1)
    llx = torch.zeros([len(t), num_channel, 40, 40]).to(device)
    for o in range(len(t)):
        for n in range(num_channel):
            print(t[o][n].shape)
            llx[n] = remap_44_tensor_channel(t[o][n])
    # print(llx)
    llx = torch.tensor(llx)
    print(llx.shape)
    return llx


def to_one_hot(x, cond_size=10):
    resten = torch.zeros([len(x), cond_size])
    for i, obj in enumerate(x):
        resten[i][obj.item()] = 1
    return resten


def unmap_to_list_tensor(ex):
    return torch.mm(ex.view(1, -1), unmap_tensor44)


def unmap_to_list(ex):
    resl = [0 for i in range(0, 1039)]
    for i in range(1039):
        l1 = ddct[i]
        # print(ex)
        resl[i] = copy.copy(ex[l1[1]][l1[0]])
    return torch.tensor(resl)


def get_rand_int_vector(x, maxv=10):
    return torch.tensor([[np.random.randint(10)] for i in range(x)])


def get_rand_noise_vector(x, y, channel=1, batch_size=4):
    return torch.rand(batch_size, channel, y, x)


def get_rand_noise_vector_tuple(noise_size_tuple, batch_size=4):
    """
    :param noise_size_tuple: x, y, channels
    :param batch_size:
    :return:
    """
    return torch.rand(batch_size, noise_size_tuple[2], noise_size_tuple[1], noise_size_tuple[0])

