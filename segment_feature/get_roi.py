import SimpleITK as sitk
import os
import numpy as np
from skimage import morphology
import nrrd


def get_roi_box(label, box_x=40, box_y=50, box_z=40):
    roi_label_box = np.zeros(label.shape, dtype=np.int16)
    x, y, z = np.where(label > 0)
    if len(z) == 0:
        return roi_label_box
    a, b = np.where(label[:, :, min(z)])
    center_xy = [(min(a) + max(a)) // 2, (min(b) + max(b)) // 2]
    # to data, the lowest middle of the kidney is center_xy[0], center_xy[1], min(z)

    roi_box_top = min(z) - 6  # interval of five layers
    if roi_box_top < 0:
        return roi_label_box

    x_begin = center_xy[0] - box_x // 2
    y_begin = center_xy[1] - box_y // 2
    if x_begin < 0:
        x_begin = 0
    if y_begin < 0:
        y_begin = 0
    x_end = center_xy[0] + box_x // 2
    y_end = center_xy[1] + box_y // 2
    if x_end > 256:
        x_end = 256
    if y_end > 512:
        y_end = 512

    if roi_box_top - box_z < 0:
        roi_box_bottom = 0
    else:
        roi_box_bottom = roi_box_top - box_z

    roi_label_box[x_begin:x_end, y_begin:y_end, roi_box_bottom:roi_box_top + 1] = 1
    return roi_label_box


def delete_useless_area(data_array, connectivity=3, save_num=1):
    if np.sum(data_array) == 0:
        return data_array

    connect_areas = morphology.label(data_array, connectivity=connectivity)

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


# data_path = 'E:/hypertension/2.processdata/select_data/'
# kidney_label_path = 'E:/hypertension/2.processdata/select_kidney_label_012/'
# roi_save_path = 'E:/hypertension/2.processdata/zztgo/'
#
# data_names = [name for name in os.listdir(data_path) if name.split('.')[1] == 'mhd']
# kidney_label_names = [name.split('.')[0] + '_l.mhd' for name in data_names]
# save_roi_names = [name.split('.')[0] + '_roi.mhd' for name in data_names]
#
# for i in range(0, len(data_names)):
#     data_mhd = sitk.ReadImage(data_path + data_names[i])
#     data_array = sitk.GetArrayFromImage(data_mhd)
#     data_array = data_array.transpose(2, 1, 0)  # change to shape (512, 512, n)
#
#     kidney_label_mhd = sitk.ReadImage(kidney_label_path + kidney_label_names[i])
#     kidney_label_array = sitk.GetArrayFromImage(kidney_label_mhd)
#     kidney_label_array = kidney_label_array.transpose(2, 1, 0)
#
#     roi_label_box1 = get_roi_box(kidney_label_array[0:256, :, :])
#     roi_label_box2 = get_roi_box(kidney_label_array[256:, :, :])
#     roi_label_box = np.concatenate((roi_label_box1, roi_label_box2), axis=0)
#
#     roi_label = np.zeros(roi_label_box.shape, dtype=np.int16)
#     # threshold
#     roi_label[(data_array < 980) * (data_array > 880) * (roi_label_box > 0)] = 1
#
#     # max connect area
#     roi_label = np.concatenate((delete_useless_area(roi_label[0:256, :, :]),
#                                 delete_useless_area(roi_label[256:, :, :])), axis=0)
#
#     roi_label = roi_label.astype(np.bool)
#     # fill hole
#     for height in range(0, roi_label.shape[2]):
#         if np.sum(roi_label[:, :, height]) > 0:
#             roi_label[:, :, height] = \
#                 morphology.remove_small_holes(roi_label[:, :, height], min_size=230, connectivity=1)
#
#     # remove small objects
#     for height in range(0, roi_label.shape[2]):
#         if np.sum(roi_label[:, :, height]) > 0:
#             roi_label[:, :, height] = \
#                 morphology.remove_small_objects(roi_label[:, :, height], min_size=230, connectivity=1)
#
#     # transpose and save
#     roi_label = roi_label.transpose(2, 1, 0)
#
#     roi_label = sitk.GetImageFromArray(roi_label.astype(np.uint8))
#     roi_label.SetSpacing(data_mhd.GetSpacing())
#     sitk.WriteImage(roi_label, roi_save_path + save_roi_names[i])


data_path = 'E:/hypertension/3.kidneynrrd(new_data)/process_data/kidney_data/'
kidney_label_path = 'E:/hypertension/3.kidneynrrd(new_data)/process_data/kidney_predict/'
roi_save_path = 'E:/hypertension/3.kidneynrrd(new_data)/process_data/fat_predict/'

data_names = os.listdir(data_path)

for i in range(0, len(data_names)):
    data_array, _ = nrrd.read(data_path + data_names[i])
    kidney_label_array, head = nrrd.read(kidney_label_path + data_names[i])

    roi_label_box1 = get_roi_box(kidney_label_array[0:256, :, :])
    roi_label_box2 = get_roi_box(kidney_label_array[256:, :, :])
    roi_label_box = np.concatenate((roi_label_box1, roi_label_box2), axis=0)

    roi_label = np.zeros(roi_label_box.shape, dtype=np.int16)
    # threshold
    roi_label[(data_array < 980) * (data_array > 880) * (roi_label_box > 0)] = 1

    # max connect area
    roi_label = np.concatenate((delete_useless_area(roi_label[0:256, :, :]),
                                delete_useless_area(roi_label[256:, :, :])), axis=0)

    roi_label = roi_label.astype(np.bool)
    # fill hole
    for height in range(0, roi_label.shape[2]):
        if np.sum(roi_label[:, :, height]) > 0:
            roi_label[:, :, height] = \
                morphology.remove_small_holes(roi_label[:, :, height], min_size=230, connectivity=1)

    # remove small objects
    for height in range(0, roi_label.shape[2]):
        if np.sum(roi_label[:, :, height]) > 0:
            roi_label[:, :, height] = \
                morphology.remove_small_objects(roi_label[:, :, height], min_size=230, connectivity=1)

    # save
    roi_label = roi_label.astype(np.uint8)
    nrrd.write(roi_save_path + data_names[i], roi_label, _)