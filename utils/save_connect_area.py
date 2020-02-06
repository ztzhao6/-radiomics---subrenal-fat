import SimpleITK as sitk
import numpy as np
import os
from skimage import morphology
import nrrd


def delete_useless_area(data_array, save_num=1):
    connect_areas = morphology.label(data_array, connectivity=3)

    sum_tmp = []
    j = 1
    while True:
        tmp = np.sum(connect_areas == j)
        if tmp == 0:
            break
        else:
            sum_tmp.append(tmp)
            j = j + 1

    label = np.zeros(data_array.shape, dtype=np.int16)
    if save_num == 1:
        num = np.argmax(sum_tmp) + 1
        label[connect_areas == num] = 1
        return label
    elif save_num == 2:
        num = np.argmax(sum_tmp) + 1
        sum_tmp[num - 1] = 0
        num_2 = np.argmax(sum_tmp) + 1
        label[connect_areas == num] = 1
        label[connect_areas == num_2] = 1
        return label

nrrd_path = 'D:/liver_ct_2/data/'
process_path = 'D:/liver_ct_2/segment/result_3d/'
save_path = 'D:/liver_ct_2/segment/result/'

# process_data_names = [name for name in os.listdir(process_path) if name.split('.')[1] == 'mhd']

# process_data_names = [name for name in os.listdir(process_path)]
#
# for i in range(0, len(process_data_names)):
#     _, head = nrrd.read(nrrd_path + process_data_names[i].split('..')[0] + '.nrrd')
#     data_mhd = sitk.ReadImage(process_path + process_data_names[i])
#     data_array = sitk.GetArrayFromImage(data_mhd)
#     data_array_1 = sitk.GetArrayFromImage(data_mhd)
#     data_array_1[data_array_1 > 1] = 1
#
#     maxconnect_label = delete_useless_area(data_array_1, save_num=2)
#     maxconnect_label = maxconnect_label * data_array
#     maxconnect_label = maxconnect_label.transpose((2, 1, 0))
#     nrrd.write(save_path + process_data_names[i].split('..')[0] + '.nrrd', maxconnect_label, head)
    # maxconnect_label = sitk.GetImageFromArray(maxconnect_label)
    # maxconnect_label.SetSpacing(data_mhd.GetSpacing())
    # sitk.WriteImage(maxconnect_label, str(save_path + process_data_names[i]))

process_data_names = os.listdir(nrrd_path)
for name in process_data_names:
    data, temp = nrrd.read(nrrd_path + name)
    label, _ = nrrd.read(process_path + name[:5] + '_liver.nrrd')
    temp['type'] = 'uint32'
    temp['endian'] = 'little'
    temp['encoding'] = 'gzip'
    maxconnect_label = delete_useless_area(label, save_num=1)

    nrrd.write(save_path + name[:5] + '_liver.nrrd', maxconnect_label, temp)