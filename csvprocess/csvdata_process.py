import pandas as pd
import os

# csv_frame = pd.read_csv(csv_path, encoding='gbk')

all_select = pd.read_csv('E:/hypertension/3.kidneynrrd(new_data)/origin_select_data/fat.CSV')
data_names = os.listdir('E:/hypertension/3.kidneynrrd(new_data)/process_data/fat_predict/')
data_names = [i.split('_')[4] for i in data_names]
drop_list = []

for j in range(0, len(all_select)):
    if str(all_select.iloc[j]['id']) not in data_names:
        drop_list.append(j)
save_data = all_select.drop(drop_list)
save_data.to_csv('E:/hypertension/3.kidneynrrd(new_data)/process_data/1.CSV', index=False)
