import pandas as pd
import xmltodict
import xml.etree.ElementTree as ET
import glob
import os
from tqdm import tqdm

path_to_xml_files1 = '/home/ubuntu/dr-you-ecg-20220420_mount/Users/dachungBoo/DachungBoo_TMT/230208_TMT/Case2/'
path_to_xml_files2 = '/home/ubuntu/dr-you-ecg-20220420_mount/Users/dachungBoo/DachungBoo_TMT/240425_TMT/case2/'
output_directory = '/home/ubuntu/dr-you-ecg-20220420_mount/Users/dachungBoo/DachungBoo_TMT/240501_TMT_Trend/'
error_log_path = os.path.join("/home/ubuntu/djboo/FactorECG/TMT/", 'errors.txt')  # 에러 로그 파일 경로

def extract_text(data_dict):
    return data_dict.get('#text', None) if isinstance(data_dict, dict) else ""

def process_trends_from_xml(file_path):
    with open(file_path, 'r', encoding='cp949') as xml_original:
        xml_dic = xmltodict.parse(xml_original.read())['CardiologyXML']
    data_list = pd.DataFrame()
    Interpretation=xml_dic.get('Interpretation').get('ReasonForTermination') if xml_dic.get('Interpretation') is not None else 'default value'
    trend_list=xml_dic.get('TrendData').get('TrendEntry')
    for trend in trend_list:  # XML 구조에 따라 적절한 경로 설정
        try:
            test1=pd.DataFrame(trend["LeadMeasurements"])
            test1["ind"]=trend.get("@Idx")
            test1["time"]=f'{trend.get("EntryTime").get("Minute")}:{trend.get("EntryTime").get("Second")}'
            test1["mets"]=trend.get("Mets")
            test1['ve']=trend.get("VECount")
            test1['pace']=trend.get("PaceCount")
            test1['artifact']=trend.get("Artifact")
            test1['speed']=trend.get("Speed").get("#text")
            test1['grade']=trend.get("Grade").get("#text")
            test1['sys']=trend.get("SystolicBP").get("#text")
            test1['dias']=trend.get("DiastolicBP").get("#text")
            test1['pt']=f'{trend.get("PhaseTime").get("Minute")}:{trend.get("PhaseTime").get("Second")}'
            test1['pn']=trend.get("PhaseName")
            test1['st']=f'{trend.get("StageTime").get("Minute")}:{trend.get("StageTime").get("Second")}'
            test1['sn']=trend.get("StageNumber")
            data_list = pd.concat([data_list, test1], ignore_index=True)
        except Exception as e:
            pass
    data_list["JPointAmplitude"]=data_list["JPointAmplitude"].apply(extract_text)
    data_list["STAmplitude20ms"]=data_list["STAmplitude20ms"].apply(extract_text)
    data_list["STAmplitude"]=data_list["STAmplitude"].apply(extract_text)
    data_list["RAmplitude"]=data_list["RAmplitude"].apply(extract_text)
    data_list["R1Amplitude"]=data_list["R1Amplitude"].apply(extract_text)
    data_list["STSlope"]=data_list["STSlope"].apply(extract_text)
    data_list["Interpretation"]=', '.join(Interpretation) if isinstance(Interpretation, list) else Interpretation
    return data_list

with open(error_log_path, 'w') as error_file:
    file_path_list1 = glob.glob(path_to_xml_files1 + '*.XML')
    file_path_list2 = glob.glob(path_to_xml_files2 + '*.XML')
    file_path_list1.extend(file_path_list2)
    for file_path in tqdm(file_path_list1, desc="Processing XML files"):
        try: 
            data = process_trends_from_xml(file_path)
            # print(data)
            file_name = os.path.basename(file_path).replace('.XML', '_trend.csv')  # 파일 이름 추출 및 확장자 변경
            output_filename = os.path.join(output_directory, file_name)  # 새로운 경로 생성
            data.to_csv(output_filename, index=False)
            # print(f"Data saved to {output_filename}")
        except Exception as e:
            error_file.write(f"{file_path}\n")  # 에러 파일에 실패한 파일 경로 작성
            print(f"Error logged for {file_path}")