import csv
import pandas as pd
import rarfile
import zipfile
import os


def detect_delimiter(file_name: str, n=2):
    sample_lines = _head(file_name, n)
    common_delimiters = [',', ';', '\t', ' ', '|', ':', ' ']
    for d in common_delimiters:
        ref = sample_lines[0].count(d)
        if ref > 0:
            if all([ref == sample_lines[i].count(d) for i in range(1, n)]):
               # print(d, "Delimiter")
                return d
    return ','


def _head(file_name: str, n: int):
    try:
        with open(file_name) as f:
            head_lines = [next(f).rstrip() for x in range(n)]
    except StopIteration:
        with open(file_name) as f:
            head_lines = f.read().splitlines()
    return head_lines



def read_csv_file(file_path, d):
    missing_values = ["?"]
    dataset = pd.read_csv(file_path, delimiter=d, na_values=missing_values,  error_bad_lines=False)
    #dataset = pd.DataFrame(reader)
    return dataset
        #print(filePath)

def identify_header(path, n=5, th=0.9):
    df1 = pd.read_csv(path, header='infer', nrows=n)
    df2 = pd.read_csv(path, header=None, nrows=n)
    sim = (df1.dtypes.values == df2.dtypes.values).mean()
    return 'infer' if sim < th else None

def read_excel(filePath):
    dataset = pd.read_excel(filePath)
    return dataset


def read_data(filePath):
    dataset = pd.read_table(filePath, sep="\s+")
    return dataset

def un_zip_files(abs_path):
    for path, subdirs, files in os.walk(abs_path):
        for name in files:
            if name.endswith((".zip")):
                filePath = path+'/'+name
                zip_file = zipfile.ZipFile(filePath)
                for names in zip_file.namelist():
                    zip_file.extract(names, path)
                zip_file.close()
            if name.endswith(".rar"):
                filePath = path+'/'+name
                rar_file = rarfile.RarFile(filePath)
                for names in rar_file.infolist():
                    rar_file.extractall(path)


def custom_csv(fname):
    if fname.endswith((".data", ".csv", ".trn", ".asc")):
        d = detect_delimiter(fname)
        if(d == " "):
            with open(fname) as infile, open('temp.csv', 'w') as outfile:
                outfile.write(infile.read().replace("   ", ",").replace("  ", ","))
            df = pd.read_csv("temp.csv")
            df.to_csv(fname)
        return read_csv_file(fname, d)
    elif fname.endswith(".dat"):
        return read_data(fname)
    elif fname.endswith((".xlsx", ".xls")):
        return read_excel(fname)

