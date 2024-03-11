from Annotation import do_annotation
import Annotation_forced
from os import listdir
import pandas as pd

for result_file in [file for file in listdir("../humans") if file.startswith("exp_1") and file.endswith(".csv")]:
    result_file = "../humans/" + result_file
    print(f" Now annotating {result_file}")
    df = pd.read_csv(result_file, sep=";")
    do_annotation(df, True).to_csv(result_file[:-4] + "_annotated.csv", index=False, sep=";") 
    
for result_file in [file for file in listdir("../humans") if file.startswith("exp_2") and file.endswith(".csv")]:
    result_file = "../humans/" + result_file
    print(f" Now annotating {result_file}")
    df = pd.read_csv(result_file, sep=";")
    do_annotation(df, False).to_csv(result_file[:-4] + "_annotated.csv", index=False, sep=";") 
    
for result_file in [file for file in listdir("../humans") if file.startswith("exp_3") and file.endswith(".csv")]:
    result_file = "../humans/" + result_file
    print(f" Now annotating {result_file}")
    df = pd.read_csv(result_file, sep=";")
    Annotation_forced.do_annotation(df, True).to_csv(result_file[:-4] + "_annotated.csv", index=False, sep=";") 