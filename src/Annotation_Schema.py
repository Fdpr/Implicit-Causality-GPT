#!/usr/bin/env python
# coding: utf-8
# ANNOTATION NACH ANNOTATIONSSCHEMA FÜR KOMMA/PUNKT IC

import pandas as pd
import spacy
import urllib
from tqdm.auto import tqdm
import warnings

warnings.filterwarnings('ignore')
tqdm.pandas()
nlp = spacy.load("de_dep_news_trf")


# ### Felder für Annotation
# - *text*: Von spacy analysiertes NLP-Objekt von Prompt + Fortsetzung
# - *prompt*: Slice von *text*, der nur den Prompt umfasst
# - *cont*: Slice von *text*, der nur die Fortsetzung umfasst
# - *verbclass*: Wenn vorhanden, Klasse des Verbs: es (experiencer-stimulus), se (stimulus-experiencer)
# - *NP1*: Name von NP1
# - *NP2*: Name von NP2
# - *NP1gender*: Geschlecht von NP1: m oder f

# #### Vollständig

def is_complete(item):
    if len(item["cont"]) > 2:
        return 9
    return 0
    
# #### Koreferenz

def coreference(item):
    # Get all anaphora candidates
    candidates = [token for token in item["cont"] if token.dep_ in ["oa", "sb", "da", "og", "nk"]]
    for candidate in candidates:
        if candidate.dep_ == "nk" and (candidate.pos_ == "DET" or next(candidate.ancestors).text == "von"):
            candidates.remove(candidate)
    male = 0
    female = 0
    plural = 0
    
    # Find out how many singular coreferences there are
    male_forms = ["er", "ihn", "ihm", "seiner", "dieser", "diesen", "dieses", "diesem", "der", "den", "dem", "jener", "jenen", "jenes", item["NP1"].lower()]
    female_forms = ["sie", "ihr", "ihrer", "diese", "dieser", "die", "der", "jene", "jener", item["NP2"].lower()]
    plural_forms = ["sie", "ihnen", "ihrer", "diese", "diesen", "dieser", "jene", "jenen", "jener", "beide", "beiden", "beider", "die", "denen", "derer"]
    if item["NP1gender"] == "m":
        male_forms.append(item["NP1"].lower())
        female_forms.append(item["NP2"].lower())
    else:
        male_forms.append(item["NP2"].lower())
        female_forms.append(item["NP1"].lower())
    for candidate in candidates:
        if candidate.text.lower() in male_forms and candidate.morph.get("Gender") == "Masc":
            male += 1
        elif candidate.text.lower() in female_forms and candidate.morph.get("Gender") == "Fem":
            female += 1
        elif candidate.text.lower() in plural_forms and candidate.morph.get("Number") == "Plur":
            plural += 1
        else:
            candidates.remove(candidate)
    if len(candidates) == 0:
        return 0, None, None, None
    if male == 0 and female == 0:
        anzahl = "pl"
    elif male > 0 and female > 0:
        anzahl = 2
    else:
        anzahl = 1
    # Find out which referent the first anaphora refers to
    first = candidates[0]
    if anzahl == "pl":
        bezug = "pl"
    else:
        first_gender = "m" if candidates[0].morph.get("Gender") == "Masc" else "f"
        bezug = 1 if first_gender == item["NP1gender"] else 2
    # Find out if first anaphora is first token in completion
    position = 1 if first == item["cont"][1] else 2
    # Find out form of first anaphora
    form = "andere"
    if first.text.lower() in ["er", "ihn", "ihm", "seiner", "sie", "ihr", "ihrer"]:
        form = "pers"
    elif first.text.lower() in ["der", "den", "dem", "die", "der"]:
        form = "dpron"
    elif first.text.lower() in ["dieser", "diesen", "diesem", "dieses", "diese"]:
        form = "prox_dem"
    elif first.text.lower() in ["jener", "jenen", "jenem", "jenes", "jene"]:
        form = "dist_dem"
    elif first.text.lower() in [item["NP1"].lower(), item["NP2"].lower()]:
        form = "eigen"
    
    return anzahl, bezug, position, form
        


# #### SUBORDINATION

def subordination(item):
    try:
        root = next(token for token in item["text"] if token.dep_ == "ROOT")
        verb = next(token for token in root.children if token.dep_ == "mo")
        res = 1 if any(child for child in verb.children if child.dep_ in ["oc", "mo"] and child.pos_ in ["VERB", "AUX"]) else 0
    except StopIteration:
        res = 0
    return res

# #### TOKENS

def num_tokens(item):
    return len(item["cont"]) - 1

# #### CHARACTERS

def num_characters(item):
    return len(str(item["cont"][1:]))


# # Annotation
# 
# Folgendes wird annotiert:
# 
# - **Vollständig**: Ist die Annotation vollständig? Vollständig heißt: erfolgreicher Parse und länger als 2 Ausgabetokens
# - **Anzahl**: Anzahl der Anaphern im Singular
# - **Bezug**: Koreferenz der ersten Anapher
# - **Position**: Ist die Anapher das erste Token in der Fortsetzung?
# - **Form**: Form der Anapher
# - **Subordination**: Existiert Subordination in der Fortsetzung?
# - **Tokens**: Anzahl Tokens in der Fortsetzung
# - **Buchstaben**: Anzahl Buchstaben in der Fortsetzung

def annotate(row, prefix):
    text = nlp((row["prompt"] + " " + row["cont"]).split(".")[0])
    comma = next(token for token in text if token.text == ",").i
    prompt = text[:comma+1]
    cont = text[comma+1:]
    item = {"text": text, "prompt": prompt, "cont":cont, "NP1": row["NP1"], "NP2": row["NP2"],"NP1gender": row["NP1gender"]}
    if "verbclass" in row.keys():
        item["verbclass"] = row["verbclass"]
    result = dict()
    result[prefix + "analysierbar"] = is_complete(item)
    result[prefix + "Anzahl_Ref"], result[prefix + "Koref_Ana1"], result[prefix + "erste_Stelle"], result[prefix + "Form_Ana1"] = coreference(item)
    result[prefix + "Einbettung"] = subordination(item)
    result[prefix + "Anzahl_Tokens"] = num_tokens(item)
    result[prefix + "Anzahl_Zeichen"] = num_characters(item)
    return result



def do_annotation(df, prefix=""):
    """
    Annotiert ein Dataframe mit Textfortsetzungen.
    - **df**: Das zu annotierende Dataframe
    - **prefix**: Wenn erwünscht, können die annotierten Spalten ein Prefix bekommen. Das ist sinnvoll, wenn zwei oder mehr Fortsetzungen in der gleichen Zeile stehen
    Datenschema
    
    Folgende Spalten muss die zu annotierende Datei umfassen:
    
    - *prompt*: Enthält Prompt bis einschließlich Komma / Konnektor (ohne Leerzeichen). Wenn Kontext vorhanden, muss der vorher entfernt werden. Prompt ist also nur der Prompt-Satz ohne vorhergehende Kontextsätze
    - *cont*: Enthält Fortsetzung nach dem Komma (ohne Leerzeichen vor dem ersten Wort)
    - *NP1*: Name von NP1
    - *NP2*: Name von NP2
    - *NP1gender*: Geschlecht von NP1: m oder f
    
    """
    
    index = df.index
    res = pd.DataFrame.from_records(list(df.progress_apply(lambda row: annotate(row, prefix), axis=1)))
    res.index = df.index
    return pd.concat([df, res], axis=1)


