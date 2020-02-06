import pydicom
import pandas as pd
import os
import collections


aimdata = pd.DataFrame()
data_root = 'D:/kindey_CT_hospital/dicom11/'
data_paths = os.listdir(data_root)
for data_path in data_paths:
    data_names = [dataname for dataname in os.listdir(data_root + data_path) if dataname[-3:] == 'dcm']
    for data_name in data_names:
        with pydicom.dcmread(data_root + data_path + '/' + data_name) as ds:
           data = collections.OrderedDict([
               ('DataName', data_name),
               ('PatientID', ds.PatientID),
               ('PatientBirthDate', ds.PatientBirthDate),
               ('PatientSex', ds.PatientSex),
               ('PatientAge', ds.PatientAge),
               ('InstitutionName', ds.InstitutionName),
               ('AcquisitionDate', ds.AcquisitionDate),
               ('AcquisitionTime', ds.AcquisitionTime),
               ('StudyDescription', ds.StudyDescription),
               ('SeriesDescription', ds.SeriesDescription),
               ('ProtocolName', ds.ProtocolName),
               ('PixelSpacing', str(ds.PixelSpacing)),
               ('SliceThickness', ds.SliceThickness),
               ('Rows', ds.Rows),
               ('Columns', ds.Columns),
               ('NumberOfFrames', ds.NumberOfFrames),
           ])
           aimdata_new = pd.DataFrame(data, index=[0])
        aimdata = pd.concat([aimdata, aimdata_new], ignore_index=True)

aimdata.to_csv('D:/1.CSV', index=False)

