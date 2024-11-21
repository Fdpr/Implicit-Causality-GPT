from Annotation_Schema import do_annotation
from os import listdir
import pandas as pd

renamer = {"NP1_gender": "NP1gender", "NP2_gender", "NP2gender"}

df_1 = pd.read_csv("../data/Judge_1_not_corrected_ChatGPT_Bloom.csv", sep=";").rename(columns=renamer)
df_1["cont"] = df_1["Human"]
df_1 = do_annotation(df_1, prefix="Human")
df_1["cont"] = df_1["ChatGPT"]
df_1 = do_annotation(df_1, prefix="ChatGPT")
df_1["cont"] = df_1["Bloom"]
df_1 = do_annotation(df_1, prefix="Bloom")
df_1.drop(columns="cont").to_csv("../data/Judge_1_not_corrected_annotated.csv", sep=";", index=False)

"""
df_2 = pd.read_csv("../data/Judge_2_not_corrected_ChatGPT_Bloom.csv", sep=";").rename(columns=renamer)
df_2["cont"] = df_2["Human"]
df_2 = do_annotation(df_2, prefix="Human")
df_2["cont"] = df_2["ChatGPT"]
df_2 = do_annotation(df_2, prefix="ChatGPT")
df_2["cont"] = df_2["Bloom"]
df_2 = do_annotation(df_2, prefix="Bloom")
df_2.drop(columns="cont").to_csv("../data/Judge_2_not_corrected_annotated.csv", sep=";", index=False)
"""

df_3 = pd.read_csv("../data/Judge_3_not_corrected_ChatGPT.csv", sep=";").rename(columns=renamer)
df_3["cont"] = df_3["Human"]
df_3 = do_annotation(df_3, prefix="Human")
df_3["cont"] = df_3["ChatGPT"]
df_3 = do_annotation(df_3, prefix="ChatGPT")
df_3.drop(columns="cont").to_csv("../data/Judge_3_not_corrected_annotated.csv", sep=";", index=False)

