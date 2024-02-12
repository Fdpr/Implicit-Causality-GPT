from Annotation import do_annotation
from os import listdir
import pandas as pd

for result_file in [file for file in listdir("../data") if file.startswith("coherence") and file.endswith(".csv")]:
    result_file = "../data/" + result_file
    print(f" Now annotating {result_file}")
    df = pd.read_csv(result_file, sep=";")
    df = df[["con", "NP1", "NP2", "female", "verbclass", "verb", "prompt", "cont", "type", "NP1gender"]]
    do_annotation(df, False).to_csv(result_file[:-4] + "_annotated.csv", index=False, sep=";") 

for result_file in [file for file in listdir("../data") if file.startswith("coreference") and file.endswith(".csv")]:
    result_file = "../data/" + result_file
    print(f" Now annotating {result_file}")
    df = pd.read_csv(result_file, sep=";")
    df = df[["con", "NP1", "NP2", "female", "verbclass", "verb", "prompt", "cont", "type", "NP1gender"]]
    do_annotation(df, True).to_csv(result_file[:-4] + "_annotated.csv", index=False, sep=";") 
    