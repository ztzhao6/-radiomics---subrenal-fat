import pandas as pd


standard_select = pd.read_csv('E:/hypertension/process&result/select_test.CSV')
select_list = standard_select['PatID'].tolist()
from_data_root = 'E:/hypertension/process&result/results_1.0/process_features/'
for i in [4]:
    drop_list = []
    from_data_path = from_data_root + 'features_' + str(i) + '.CSV'
    from_data = pd.read_csv(from_data_path)
    for j in range(0, len(from_data)):
        if from_data.iloc[j]['Image'] not in select_list:
            drop_list.append(j)
    save_data = from_data.drop(drop_list)
    save_data.to_csv(from_data_root + 'test/test_' + str(i) + '.CSV', index=False)