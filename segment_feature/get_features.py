import os
import radiomics
from radiomics import featureextractor
import collections
import csv


def get_features(outputFilePath, binwidth):
    settingPath = 'E:/hypertension/hypertension_code/segment_feature/setting.yaml'

    imagePath = 'E:/hypertension/3.kidneynrrd(new_data)/process_data/kidney_data/'
    maskPath = 'E:/hypertension/3.kidneynrrd(new_data)/process_data/fat_predict/'

    # imageNames = [name for name in os.listdir(imagePath) if name.split('.')[1] == 'mhd']
    # maskNames = [name.split('.')[0] + '_roi.mhd' for name in imageNames]

    imageNames = os.listdir(imagePath)
    maskNames = imageNames

    # Initialize feature extractor
    extractor = featureextractor.RadiomicsFeatureExtractor(settingPath, binWidth=binwidth)

    headers = None

    for i in range(len(imageNames)):
        featureVector = collections.OrderedDict()
        featureVector['Image'] = imageNames[i].split('_')[4]

        imageFilePath = imagePath + imageNames[i]
        maskFilePath = maskPath + maskNames[i]

        # Calculating features
        featureVector.update(extractor.execute(imageFilePath, maskFilePath))

        with open(outputFilePath, 'a') as outputFile:
            writer = csv.writer(outputFile, lineterminator='\n')
            if headers is None:
                headers = list(featureVector.keys())
                writer.writerow(headers)

            row = []
            for h in headers:
                row.append(featureVector.get(h, "N/A"))
            writer.writerow(row)


outputPath = 'E:/hypertension/3.kidneynrrd(new_data)/origin_features/'
# binwidth = [4, 12, 20, 28, 36]
binwidth = [4]

for i in range(len(binwidth)):
    outputFilePath = outputPath + 'features_' + str(binwidth[i]) + '.CSV'
    get_features(outputFilePath, binwidth=binwidth[i])


