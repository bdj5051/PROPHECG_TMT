import pandas as pd
import xmltodict
import xml.etree.ElementTree as ET
import glob
import os
from tqdm import tqdm
import numpy as np



folder_path = '/home/ubuntu/dr-you-ecg-20220420_mount/Users/dachungBoo/DachungBoo_TMT/240425_TMT/case2/'
file_list = os.listdir(folder_path)
file_list=[file for file in file_list if (file.startswith('#')!=True) & ('Test' not in file)]
save_path = "/home/ubuntu/djboo/FactorECG/TMT/Datasets/Downstream/Revascularization/240930_duke_sinchon2.csv"
error_log_path = os.path.join("/home/ubuntu/djboo/FactorECG/TMT/Datasets/Downstream/Revascularization/", 'errors_duke_sinchon2_.txt')  # 에러 로그 파일 경로


def extract_text_value(row):
    return float(row['#text'])

dataframes = []
with open(error_log_path, 'w') as error_file:
    for file in tqdm(file_list):
        file_path = f'/home/ubuntu/dr-you-ecg-20220420_mount/Users/dachungBoo/DachungBoo_TMT/240425_TMT/case2/{file}'
        try:
            with open(file_path, 'r', encoding='cp949') as xml_original:
                xml_dic = xmltodict.parse(xml_original.read())['CardiologyXML']
                data_list = pd.DataFrame()
                Interpretation=xml_dic.get('Interpretation').get('ReasonForTermination') if xml_dic.get('Interpretation') is not None else 'default value'
                ExerciseMeasurements=xml_dic.get('ExerciseMeasurements')
                restST=pd.DataFrame.from_dict(ExerciseMeasurements.get("RestingStats").get("RestST").get("Measurements"))
                maxST = pd.DataFrame.from_dict(ExerciseMeasurements.get("MaxSTStats").get("MaxST").get("Measurements"))
                restST['Rest_STAmplitude'] = restST['STAmplitude'].apply(extract_text_value)
                restST = restST[['@lead', 'Rest_STAmplitude']]
                maxST['Max_STAmplitude'] = maxST['STAmplitude'].apply(extract_text_value)
                maxST = maxST[['@lead', 'Max_STAmplitude']]
                subset = pd.merge(restST, maxST, on='@lead', how='left')
                subset["STD"] = abs(subset["Max_STAmplitude"]-subset["Rest_STAmplitude"])
                subset=subset[subset["@lead"]!="aVR"]
                maxSTD = subset["STD"].abs().max()
                dic={"Trend_fname":file,
                      "Duration":int(ExerciseMeasurements.get("ExercisePhaseTime").get("Minute")),
                      "STD":maxSTD,
                      "Angina":Interpretation
                     }
                dataframes.append(dic)
        except Exception as e:
            error_file.write(f"{file}\n")  # 에러 파일에 실패한 파일 경로 작성
            print(f"Error logged for {file}")
            pass

duke_dataset=pd.DataFrame(dataframes)
duke_dataset['score'] = duke_dataset['Angina'].apply(lambda x: 2 if isinstance(x, str) and 'Chest' in x else 0)
duke_dataset["duke_score"]=duke_dataset["Duration"]-(5*duke_dataset["STD"])-(4*duke_dataset["score"])
duke_dataset['risk group'] = np.where(duke_dataset['duke_score'] >= 5, 'low', 
                                      np.where(duke_dataset['duke_score'] < -10, 'high', 'middle'))
duke_dataset.to_csv(save_path)