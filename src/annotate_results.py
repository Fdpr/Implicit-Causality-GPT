from Annotation import do_annotate
import pandas as pd

for result_file in [file for file in listdir("../data") if file.startswith("coreference") and file.endswith(".csv")]:
    print(f" Now annotating {result_file}")
    df = pd.read_csv(result_file, sep=";")
    df = df.rename(columns={"np1":"NP1", "np2":"NP2", "cat":"verbclass", "continuation":"cont"})
    df["type"] = ["Experiment"] * len(df)
    df["cont"] = df.apply(lambda row: row["cont"][len(row["prompt"]) + 1:], axis=1)
    df["NP1gender"] = df["female"].apply(lambda b: "f" if b else "m")
    do_annotation(df, True).to_csv(result_file[:-4] + "_annotated.csv", index=False, sep=";") 
    
for result_file in [file for file in listdir("../data") if file.startswith("coherence") and file.endswith(".csv")]:
    print(f" Now annotating {result_file}")
    df = pd.read_csv(result_file, sep=";")
    df = df.rename(columns={"np1":"NP1", "np2":"NP2", "cat":"verbclass", "continuation":"cont"})
    df["type"] = ["Experiment"] * len(df)
    df["cont"] = df.apply(lambda row: row["cont"][len(row["prompt"]) + 1:], axis=1)
    df["NP1gender"] = df["female"].apply(lambda b: "f" if b else "m")
    do_annotation(df, False).to_csv(result_file[:-4] + "_annotated.csv", index=False, sep=";") 