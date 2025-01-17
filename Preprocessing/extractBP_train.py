import pandas as pd
import os
from tqdm import tqdm
import numpy as np

trainset_path = '/home/ubuntu/djboo/FactorECG/TMT/Datasets/Downstream/Revascularization/240503_downstream_train_whole.csv'
df_train = pd.read_csv(trainset_path)
df_train['Time']=df_train["TestID"].str.split('_').str[2].astype('int')
print(f'Overall set (P_CNT): {len(df_train["PseudoID"].unique())}')
df_train['Trend_fname'] = df_train['TMT_fname'].str.split('/').str[-1].str.split('.').str[0]+'_trend.csv'

def format_values(value):
    # 콜론으로 구분
    parts = value.split(':')
    if len(parts[1]) < 2:  # 콜론 뒤의 숫자가 2개 미만인 경우
        return parts[0] + '0' + parts[1]
    else:
        return value.replace(":", "")

patient_list=df_train['Trend_fname'].drop_duplicates().reset_index(drop=True)
save_dir="/home/ubuntu/dr-you-ecg-20220420_mount/Users/dachungBoo/DachungBoo_TMT/240501_TMT_Trend/"
dataframes = []

for row in tqdm(patient_list):

    master_df = df_train[df_train["Trend_fname"]==row]
    bp=pd.read_csv(save_dir+row)
    bp=bp[["pn", "sn", "time", "sys", "dias"]].drop_duplicates().reset_index(drop=True)
    bp["sn"]=bp["sn"].astype('str')
    bp['stage']=np.where(bp["pn"]=="PRETEST", "SITTING", np.where(bp['pn'] == 'EXERCISE', 'STAGE', "#"))
    bp['stage']=np.where(bp["pn"]=="EXERCISE", bp['stage']+" "+bp["sn"],
                             np.where(bp["pn"]=="RECOVERY",bp['stage']+bp["sn"], bp['stage']))
    bp["time"]=bp["time"].apply(format_values).astype("int")
    merged_data=pd.merge(master_df, bp, on = ["stage"], how = "left")
    merged_data["gap_time"]=merged_data["Time"]-merged_data["time"]
    merged_data=merged_data[['PseudoID', 'AcqDate', 'TestID', 'shape', 'stage',
       'SampleBase', 'Gain', 'length', 'CAD', 'Revascularization', 'Time', 'time','sys', 'dias',
       'gap_time']].drop_duplicates().dropna().reset_index(drop=True)
    
    merged_data=merged_data.groupby(['PseudoID', 'AcqDate', 'TestID', 'shape', 'stage', 'SampleBase', 'Gain', 'length', 'Time']).apply(lambda x: x.loc[abs(x['gap_time']).idxmin()]).reset_index(drop=True)

    dataframes.append(merged_data)
    
df_train2=pd.concat(dataframes, ignore_index=True)
df_train2.to_csv("/home/ubuntu/djboo/FactorECG/TMT/Datasets/Downstream/Revascularization/240505_downstream_train_whole_bp_origin.csv")