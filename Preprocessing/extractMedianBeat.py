
import os
from tqdm import tqdm

import xmltodict
import time, os
import csv
import pandas as pd
import numpy as np
import array

from multiprocessing import Pool
import multiprocessing as mp
import parmap
from ast import literal_eval



folder_path = '/home/ubuntu/dr-you-ecg-20220420_mount/Users/dachungBoo/DachungBoo_TMT/240425_TMT/case2/'
stage_path = '/home/ubuntu/dr-you-ecg-20220420_mount/Users/dachungBoo/DachungBoo_TMT/dhkim3'
file_list = os.listdir(folder_path)
file_list=[file for file in file_list if (file.startswith('#')!=True) & ('Test' not in file)]
save_dir = '/home/ubuntu/dr-you-ecg-20220420_mount/Users/dachungBoo/DachungBoo_TMT/240116_TMT_medianWaveform/'



for file in tqdm(file_list):
    try :
        tmt_fname_split = file.split("#")
        pt_id=tmt_fname_split[0]
        acq_date=''.join(tmt_fname_split[2].split('_'))
        
        tmt_ecg_list=pd.read_csv(os.path.join(stage_path,file.replace('.XML', '_t_list.csv')))
        stage_time_list=[]
        for stage in tmt_ecg_list['StageName'].unique():
            entry_start_time=min(tmt_ecg_list[tmt_ecg_list['StageName']==stage]['EntryTime'])
            dict_1={'StageName':stage,
                   'EntryTime':entry_start_time}
            stage_time_list.append(dict_1)
        stage_time_df=pd.DataFrame(stage_time_list)

        with open(folder_path+file, 'r') as xml_original:
            xml_dic = xmltodict.parse(xml_original.read())['CardiologyXML']
            MedianData=xml_dic.get('MedianData')
            sample_rate=MedianData.get('SampleRate').get('#text')
            resolution=MedianData.get('Resolution').get('#text')
            Median=MedianData.get('Median')
            
            waveform={}
            master_list=[]
            
            for raw in Median:
                idx = raw.get('@Idx')
                Time = str(raw.get('Time').get('Minute'))+str(raw.get('Time').get('Second').rjust(2,'0'))                
                fname = pt_id+'_'+acq_date+'_'+Time  
                waveform = {}
                for lead_ecg in raw.get('WaveformData'):
                    lead = lead_ecg.get('@lead')
                    trace = literal_eval(lead_ecg.get('#text'))
                    waveform[lead]=trace
                waveform_df=pd.DataFrame.from_dict(waveform)
                waveform_df.to_csv(os.path.join(save_dir,fname+".csv"), index=False)

                shape=waveform_df.shape

                max_range = len(stage_time_df)
                for i in range(1, max_range):
                    if (int(Time) >=stage_time_df['EntryTime'].loc[i-1])&((int(Time) <stage_time_df['EntryTime'].loc[i])) :
                        stage=stage_time_df['StageName'].loc[i-1]
                        break
                    elif i == max_range-1:
                        stage=stage_time_df['StageName'].loc[max_range-1]
                        break

                master = {'AlsUnitNo':pt_id,
                          'AcqDate':acq_date,
                          'fname':fname,
                          'shape':shape,
                          'stage':stage,
                          'SampleRate':sample_rate,
                          'gain':resolution
                         }
                master_list.append(master)
            master_df=pd.DataFrame(master_list)
            master_df.to_csv(os.path.join(save_dir,fname+"_Master.csv"), index=False)
    except Exception as Argument:
        f = open('bug.txt', 'a')
        try:
            if type(fname) == str:
                f.write(f'{i} / {fname} : {str(Argument)} \n')
                print(f'{i} / {fname} raised error : {Argument}')
            else:
                f.write(f'{i} : {str(Argument)} \n')
                print(f'{i} raised error : {Argument}')    
        except:
            f.write(f'{i} : {str(Argument)} \n')
            print(f'{i} raised error : {Argument}')
        f.close()
