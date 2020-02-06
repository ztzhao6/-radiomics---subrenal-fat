import SimpleITK as sitk
import numpy as np
import os
from skimage import morphology
import nrrd


def change_spacing(process_path, save_path, process_data_names):
    for k in range(0, len(process_data_names)):
        data_mhd = sitk.ReadImage(process_path + process_data_names[k])
        a, b, c = data_mhd.GetSpacing()
        data_mhd.SetSpacing((a, b, a))
        sitk.WriteImage(data_mhd, str(save_path + process_data_names[k]))


def check_null_label(data_array):
    if np.sum(data_array) == 0:
        return 0
    else:
        return 1


def check_label_num(data_array):
    # data_array has been max_connect processed.
    labels, labels_num = morphology.label(data_array, return_num=True, connectivity=3)
    return labels_num


def check_height(data_array, height_threshold=20):
    # 若两边均小于20层，则返回0
    data_array_1 = data_array[0:256, :, :]
    data_array_2 = data_array[256:, :, :]

    x1, y1, z1 = np.where(data_array_1 > 0)
    if len(z1) == 0:
        flag_1 = 0
    else:
        height_1 = max(z1) - min(z1)
        if height_1 < height_threshold:
            flag_1 = 0
        else:
            flag_1 = 1

    x2, y2, z2 = np.where(data_array_2 > 0)
    if len(z2) == 0:
        flag_2 = 0
    else:
        height_2 = max(z2) - min(z2)
        if height_2 < height_threshold:
            flag_2 = 0
        else:
            flag_2 = 1

    if flag_1 == 0 and flag_2 == 0:
        return 0
    else:
        return 1


def check_height_2(data_array):
    # 若两边全无，返回-1；若一边无，则返回0；两边全有，返回1
    flag_1 = 1
    flag_2 = 1
    data_array_1 = data_array[0:256, :, :]
    data_array_2 = data_array[256:, :, :]

    x1, y1, z1 = np.where(data_array_1 > 0)
    if len(z1) == 0:
        flag_1 = 0

    x2, y2, z2 = np.where(data_array_2 > 0)
    if len(z2) == 0:
        flag_2 = 0

    if flag_1 == 0 and flag_2 == 0:
        return -1
    elif flag_1 == 0 or flag_2 == 0:
        return 0
    else:
        return 1


def check_last_area_size(data_array, area_threshold=1200):
    # the num of connect areas should be 2
    data_array_1 = data_array[0:256, :, :]
    data_array_2 = data_array[256:, :, :]

    for k in range(data_array_1.shape[2]):
        if np.sum(data_array_1[:, :, k]) > 0:
            sum_1 = np.sum(data_array_1[:, :, k])
            break

    for k in range(data_array_2.shape[2]):
        if np.sum(data_array_2[:, :, k]) > 0:
            sum_2 = np.sum(data_array_2[:, :, k])
            break

    sum = sum_1 + sum_2
    if sum >= area_threshold:
        return 0
    else:
        return 1


def check_last_area_tumor(data_array):
    flag_1 = -1
    flag_2 = -1
    data_array_1 = data_array[0:256, :, :]
    data_array_2 = data_array[256:, :, :]

    for k in range(data_array_1.shape[2]):
        if np.sum(data_array_1[:, :, k]) > 0:
            if 2 in data_array_1[:, :, k:k + 35]:
                flag_1 = 0
                data_array[0:256, :, :] = 0
            else:
                flag_1 = 1
            break

    for k in range(data_array_2.shape[2]):
        if np.sum(data_array_2[:, :, k]) > 0:
            if 2 in data_array_2[:, :, k:k + 35]:
                flag_2 = 0
                data_array[256:, :, :] = 0
            else:
                flag_2 = 1
            break

    return data_array, flag_1, flag_2


# process_path = 'E:/hypertension/2.processdata/zztgo/'
#
# process_data_names = [name for name in os.listdir(process_path) if name.split('.')[1] == 'mhd']
#
# for i in range(len(process_data_names)):
#     data = sitk.ReadImage(process_path + process_data_names[i])
#     data_array = sitk.GetArrayFromImage(data)
#     data_array = data_array.transpose(2, 1, 0)
#
#     if check_height(data_array) == 0:
#         print(process_data_names[i])

    # check_last_area_tumor
    # data_array, flag_1, flag_2 = check_last_area_tumor(data_array)
    #
    # data_array = data_array.transpose(2, 1, 0)
    # data_array = sitk.GetImageFromArray(data_array.astype(np.uint8))
    # data_array.SetSpacing(data_array.GetSpacing())
    # sitk.WriteImage(data_array, process_path + process_data_names[i])
    # print(process_data_names[i], flag_1, flag_2)


data_path = 'E:/hypertension/3.kidneynrrd(new_data)/process_data/fat_predict/'
data_names = os.listdir(data_path)

for data_name in data_names:
    data, _ = nrrd.read(data_path + data_name)
    # if check_last_area_size(data) == 0:
    #     os.rename('E:/hypertension/3.kidneynrrd(new_data)/process_data/kidney_predict/' + data_name,
    #               'E:/hypertension/3.kidneynrrd(new_data)/process_data/' + data_name)
    #     os.rename('E:/hypertension/3.kidneynrrd(new_data)/process_data/kidney_data/' + data_name,
    #               'E:/hypertension/3.kidneynrrd(new_data)/process_data/a/' + data_name)

    # data_array, flag_1, flag_2 = check_last_area_tumor(data)
    # if flag_1 == 1 or flag_2 == 1:
    #     nrrd.write(data_path + data_name, data_array, _)
    # else:
    #     os.rename('E:/hypertension/3.kidneynrrd(new_data)/process_data/kidney_predict/' + data_name,
    #               'E:/hypertension/3.kidneynrrd(new_data)/process_data/' + data_name)
    #     os.rename('E:/hypertension/3.kidneynrrd(new_data)/process_data/kidney_data/' + data_name,
    #               'E:/hypertension/3.kidneynrrd(new_data)/process_data/a/' + data_name)

    if check_height(data) == 0:
        os.rename('E:/hypertension/3.kidneynrrd(new_data)/process_data/kidney_predict/' + data_name,
                  'E:/hypertension/3.kidneynrrd(new_data)/process_data/b/' + data_name)
        os.rename('E:/hypertension/3.kidneynrrd(new_data)/process_data/kidney_data/' + data_name,
                  'E:/hypertension/3.kidneynrrd(new_data)/process_data/a/' + data_name)
        os.rename('E:/hypertension/3.kidneynrrd(new_data)/process_data/fat_predict/' + data_name,
                  'E:/hypertension/3.kidneynrrd(new_data)/process_data/' + data_name)