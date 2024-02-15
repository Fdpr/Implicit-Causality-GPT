#!/usr/bin/env python
# coding: utf-8
# AKTUELLE VERSION STAND 2024/02/12

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

# #### Anaphorische Koreferenz

def coreference(item):
    if len(item["cont"]) == 0:
        return ("", "")
    if item["cont"][0].tag_ == "KOUI":
        return ("NP1", "elliptisch")
    try:
        subject = next(token for token in item["cont"] if token.dep_ == "sb")
        form = ""
        coreference = ""
        if subject.text in ["er", "sie"]:
            form = "personal"
        elif subject.text in ["dieser", "diese"]:
            form = "demonstrativ"
        elif subject.text in [item["NP1"], item["NP2"]]:
            form = "Eigenname"
        elif subject.text in ["der", "welcher", "die", "welche"]:
            form = "relativ"
        if form != "":
            if "Plur" in subject.morph.get("Number"):
                coreference = "NP1+NP2"
            else:
                if "Masc" in subject.morph.get("Gender"):
                    if item["NP1gender"] == "m":
                        coreference = "NP1"
                    else:
                        coreference = "NP2"
                elif "Fem" in subject.morph.get("Gender"):
                    if item["NP1gender"] == "f":
                        coreference = "NP1"
                    else:
                        coreference = "NP2"
        return (coreference, form)
    except StopIteration:
        return ("", "")



def relative_coreference(item):
    if len(item["cont"]) == 0:
        return ""
    reltest = item["cont"][0]
    if reltest.tag_ == "PRELS" or reltest.tag_ == "PRELAT":
        if "Plur" in reltest.morph.get("Number"):
            return "NP1+NP2"
        if "Masc" in reltest.morph.get("Gender"):
            if item["NP1gender"] == "f":
                return "NP2"
            else:
                return "NP1"
        elif "Fem" in reltest.morph.get("Gender"):
            if item["NP1gender"] == "f":
                 return "NP1"
            else:
                return "NP2"
    return ""
    
    
# #### Diskursmarker

def discourse_marker(item):
    if len(item["cont"]) == 0:
        return ""
    return str(item["cont"][0]


# #### Fortsetzungstyp

# In[4]:


def cont_type(item):
    if len(item["cont"]) == 0:
        return ""
    cont = item["cont"][0]
    tag = cont.tag_
    if tag == "PRELS" or tag == "PRELAT":
        return "Relativsatz"
    elif tag == "APPR" or tag == "APPRART":
        return "PP"
    elif tag == "KOUS" or tag == "KOUI":
        return "subordinierend"
    elif tag == "KON":
        return "koordinierend"
    else:
        try:
            subject_index = next(token for token in item["cont"] if token.dep_ == "sb").i
            verb_index = next(token for token in item["cont"] if "Fin" in token.morph.get("VerbForm")).i
            return "Hauptsatz"
        except StopIteration:
            pass
    return ""


# #### Diskursrelation

# In[5]:


discourse_markers = {}
explanation = ("weil ", "aufgrund dessen ", "da ", "dadurch, dass ", "denn ", "ergo ", "in Anbetracht dessen ", "insofern, als ", "zumal ") 
discourse_markers[explanation] = "Contingency.Cause.Reason"
consequence = ("sodass ", "also ", "daher ", "darum ", "dementsprechend ", "demgemäß ", "demnach ", "demzufolge ", "deshalb ", "deswegen ", "drum ", "folgerichtig ", "folglich ", "infolgedessen ", "sonach ", "weshalb ", "weswegen ", "wodurch ") 
discourse_markers[consequence] = "Contingency.Cause.Result"
occasion = ("bevor ", "ehe ", "wonach ", "worauf ", "woraufhin ") 
discourse_markers[occasion] = "Temporal.Asynchronous.Precedence"
elaboration = ("und zwar ", "beziehungsweise ", "genauer gesagt ", "nämlich ") 
discourse_markers[elaboration] = "Expansion.Level-of-Detail.Arg2-as-detail"
contrast = ("aber ", "allerdings ", "andererseits ", "dafür ", "dagegen ", "dahingegen ", "demgegenüber ", "dem gegenüber ", "doch ", "hingegen ", "jedoch ", "nur dass ", "wogegen ", "wohingegen ", "wohin gegen ") 
discourse_markers[contrast] = "Comparison.Contrast"
synchronous = ("als ", "derweilen ", "gleichzeitig ", "nebenbei ", "nebenher ", "nebenbei ", "während ", "währenddessen ", "unterdessen ", "zugleich ")
discourse_markers[synchronous] = "Temporal.Synchronous"
succession = ("davor ", "kaum dass ", "nachdem ", "seitdem ", "vorher ", "zuerst ", "zunächst ", "zuvor ")
discourse_markers[succession] = "Temporal.Asynchronous.Succession"
concession = ("obwohl ", "gleichwohl ", "obgleich ", "obschon ", "obzwar ", "trotzdem ", "wenngleich ", "zwar ")
discourse_markers[concession] = "Comparison.Concession"
manner = ("als wenn ", "dadurch, dass ", "indem ")
discourse_markers[manner] = "Expansion.Manner.Arg2-as-manner"
goal = ("damit ", "um ")
discourse_markers[goal] = "Contingency.Purpose.Arg2-as-goal"

def discourse_relation_explicit(item):
    if len(item["cont"]) == 0:
        return ""
    marker = item["cont"][0]
    if marker.tag_ in ["KOUI", "KON", "KOUS"]:
        cont_l = item["cont"].text.lower()
        for marker_list in discourse_markers.keys():
            if any(cont_l.startswith(marker) for marker in marker_list):
                return discourse_markers[marker_list]
    return ""

def discourse_relation_explicit_corrected(item):
    if len(item["cont"]) == 0:
        return ""
    marker = item["cont"][0].text.lower()
    if "verbclass" in item.keys():
        cat = item["verbclass"]
        if (marker == "indem") and (cat in ["se", "stim-exp"]):
            return "Contingency.Cause.Reason"
        # Keine "als"-Korrektur
        # if (marker == "als") and (cat in ["se", "stim-exp"]):
        #     return "Contingency.Cause.Reason"
        elif (cat in ["es", "exp-stim"]) and (marker in ["der", "welcher", "die", "welche"]) and (item["Koreferenz"] == "NP2"):
            return "Contingency.Cause.Reason"
    return discourse_relation_explicit(item)

def discourse_relation_implicit(item):
    pass


# # Annotation
# 
# Folgendes wird annotiert:
# 
# - **Diskursmarker**: Das erste Token der Fortsetzung, was üblicherweise der Diskursmarker ist. Es wird nicht geprüft, ob es tatsächlich einer ist.
# - **Diskursrelation Explizit**: Erkennt ausgewählte PDTB-Relationen anhand des Konnektors nach dem Komma.
# - **Diskursrelation Explizit SE/ES-Korrektur**: Wenn Informationen über ES/SE-Kategorie des Verbs vorhanden sind, wird die explizite Annotation für manche Konnektoren angepasst. Zum Beispiel: als -> Synchronous bei ES-Verben, aber als -> Reason bei SE-Verben
# - **Diskursrelation Implizit**: TODO, Erkennung von Diskursrelationen mittels BERT-Klassifizierung
# - **Fortsetzungstyp**: Die syntaktische Art der Fortsetzung: PP, Rel-Satz, subordinierend, koordinierend, Hauptsatz
# - **Relativkoreferenz**: Bei Relativsätzen: NP1- oder NP2-Koreferenz des Relativpronomens
# - **Koreferenz**: Markiert anaphorische Koreferenz zu NP1, NP2 oder NP1+NP2, so vorhanden
# - **Anaphorische Form**: Wenn Koreferenz: Form der Anapher: personal, demonstrativ, relativ, Eigenname, elliptisch

# In[6]:


# has_connector: Enthält der Prompt einen Konnektor? -> Nur Koreferenz und anaphorische Form annotieren
def annotate(row, has_connector = False):
    if row["type"] != "Experiment":
        return {}
    text = nlp((row["prompt"] + " " + row["cont"]).split(".")[0])
    comma = next(token for token in text if token.text == ",").i
    prompt = text[:comma+1]
    cont = text[comma+1:]
    item = {"text": text, "prompt": prompt, "cont":cont, "NP1": row["NP1"], "NP2": row["NP2"],"NP1gender": row["NP1gender"]}
    if "verbclass" in row.keys():
        item["verbclass"] = row["verbclass"]
    result = dict()
    result["Koreferenz"], result["Anaphorische Form"] = coreference(item)
    if not has_connector:
        results["Diskursmarker"] = discourse_marker(item)
        result["Diskursrelation Explizit"] = discourse_relation_explicit(item)
        if "verbclass" in row.keys():
            item["verbclass"] = row["verbclass"]
            result["Diskursrelation Explizit SE/ES-Korrektur"] = discourse_relation_explicit_corrected(item)
        result["Diskursrelation Implizit"] = discourse_relation_implicit(item)
        result["Fortsetzungstyp"] = cont_type(item)
        result["Relativkoreferenz"] = relative_coreference(item)
    return result



def do_annotation(df, has_connector = False):
    """
    Datenschema
    
    Folgende Spalten muss die zu annotierende Datei umfassen:
    
    - *type*: Nur wenn type = "Experiment" ist, wird die Zeile annotiert.
    - *prompt*: Enthält Prompt bis einschließlich Komma / Konnektor (ohne Leerzeichen). Wenn Kontext vorhanden, muss der vorher entfernt werden. Prompt ist also nur der Prompt-Satz ohne vorhergehende Kontextsätze
    - *cont*: Enthält Fortsetzung nach dem Komma (ohne Leerzeichen vor dem ersten Wort)
    - *verbclass*: Wenn vorhanden, Klasse des Verbs: es (experiencer-stimulus), se (stimulus-experiencer)
    - *NP1*: Name von NP1
    - *NP2*: Name von NP2
    - *NP1gender*: Geschlecht von NP1: m oder f
    
    has_connector: Enthält der Prompt einen Konnektor? -> Nur Koreferenz und anaphorische Form annotieren
    """
    
    index = df.index
    res = pd.DataFrame.from_records(list(df.progress_apply(lambda row: annotate(row, has_connector), axis=1)))
    res.index = df.index
    return pd.concat([df, res], axis=1)


