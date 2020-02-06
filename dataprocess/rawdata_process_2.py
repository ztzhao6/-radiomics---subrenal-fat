import pydicom
import tifffile
import numpy as np
import nrrd
import pandas as pd


data_root = 'E:/to_path/'
save_root = 'E:/process/'

_, head = nrrd.read('D:/liver_CT/liver_label_CT/A_001_liver.nrrd')
datainformation = pd.read_csv('D:/1.CSV')

for i in range(243):
    data_name = datainformation.iloc[i]['DataName'][:-4]
    data = tifffile.imread(data_root + data_name + '.tif')
    data = data.transpose(2, 1, 0)

    if datainformation.iloc[i]['num'] > 1:
        data = data[:, :, :datainformation.iloc[i]['z']]

    head['space directions'][0, 0] = float(datainformation.iloc[i]['PixelSpacing'])
    head['space directions'][1, 1] = float(datainformation.iloc[i]['PixelSpacing'])
    head['space directions'][2, 2] = float(datainformation.iloc[i]['SliceThickness'])

    nrrd.write(save_root + data_name + '.nrrd', data, head)

