import pandas as pd
import os


def clean(string):
    return string.lower().replace("\"", "")

def load_icd():
    if os.path.exists("ICD_10v601.csv"):
        df = pd.read_csv("ICD_10v601.csv", header=None)
    else:
        df = pd.read_csv("../ICD_10v601.csv", header=None)
    icd = df[0].tolist()
    des = [clean(d) for d in df[1].tolist()]
    icd2str = {}
    for idx, code in enumerate(icd):
        if len(code) < 7:
            code = code + "0" * (7 - len(code))
        icd2str[code] = des[idx]
    return icd2str, list(set(des))

def load_icd_for_train():
    icd2str, des_list = load_icd()
    return des_list, [[des] for des in des_list]

def load_additional_icd_for_train():
    if os.path.exists("icd.csv"):
        df = pd.read_csv("icd.csv")
    else:
        df = pd.read_csv("../icd.csv")
    icd_modified = df["医院项目编码"]
    des_modified = df["医院项目名称"]
    icd = df["国标编码"]
    des = df["国标名称"]

    x_list = []
    y_list = []
    for idx in range(len(icd)):
        if des_modified[idx] != des[idx]:
            x_list.append(des_modified[idx])
            y_list.append([des[idx]])
    return x_list, y_list

def load_additional_icd():
    icd2str, str_list = load_icd()

    if os.path.exists("icd.csv"):
        df = pd.read_csv("icd.csv")
    else:
        df = pd.read_csv("../icd.csv")
    icd_modified = df["医院项目编码"]
    des_modified = df["医院项目名称"]
    icd = df["国标编码"]
    des = df["国标名称"]

    str2standard = {}
    for idx in range(len(icd)):
        if icd[idx] in icd2str:
            standard = icd2str[icd[idx]]
            if des_modified[idx] != standard:
                str2standard[des_modified[idx]] = standard
            if des[idx] != standard:
                str2standard[des[idx]] = standard
    print(f"Additional matching term: {len(str2standard)}")
    
    return str2standard

if __name__ == "__main__":
    icd2str, str_list = load_icd()
    print(len(icd2str), len(str_list))
    print(str_list[0:5])
    str2standard = load_additional_icd()
    print(list(str2standard.items())[0:5])

