
#########################################
# Project   : ARFF to CSV converter     #
# Created   : 10/01/17 11:08:06         #
# Author    : haloboy777                #
# Licence   : MIT                       #
#https://github.com/haloboy777/arfftocsv/blob/master/arffToCsv.py#
#########################################

# Importing library
import os

def arffTocsv(abs_path):
    # Getting all the arff files from the current directory

     for path, subdirs, files in os.walk(abs_path):
       for name in files:
           if name.endswith((".arff")):
                with open(os.path.join(path, name), "r") as inFile:
                    content = inFile.readlines()
                    name, ext = os.path.splitext(inFile.name)
                    new = toCsv(content)
                    with open(name + ".csv", "w") as outFile:
                        outFile.writelines(new)
# Function for converting arff list to csv list


def toCsv(text):
    data = False
    header = ""
    new_content = []
    for line in text:
        if not data:
            if "@ATTRIBUTE" in line or "@attribute" in line:
                attributes = line.split()
                if("@attribute" in line):
                    attri_case = "@attribute"
                else:
                    attri_case = "@ATTRIBUTE"
                column_name = attributes[attributes.index(attri_case) + 1]
                header = header + column_name + ","
            elif "@DATA" in line or "@data" in line:
                data = True
                header = header[:-1]
                header += '\n'
                new_content.append(header)
        else:
            new_content.append(line)
    return new_content
