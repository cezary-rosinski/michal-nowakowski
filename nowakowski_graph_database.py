import sys
# sys.path.insert(1, 'D:\IBL\Documents\IBL-PAN-Python')
sys.path.insert(1, 'C:/Users/Cezary/Documents/IBL-PAN-Python')
import pandas as pd
from rdflib import Graph, Namespace, URIRef, Literal, BNode
from rdflib.namespace import RDF, RDFS, XSD, FOAF, OWL
import datetime
import regex as re
from my_functions import gsheet_to_df
from ast import literal_eval

#%% def

def get_birthyear(birthdate):
    try:
        if birthdate[0].isnumeric():
            b = int(birthdate.split('-')[0])
        else: b = -int(birthdate.split('-')[1])
    except TypeError:
        b = None
    return b

def get_deathyear(deathdate):
    try:
        if deathdate[0].isnumeric():
            d = int(deathdate.split('-')[0])
        else: d = -int(deathdate.split('-')[1])
    except TypeError:
        d = None
    return d

#%%
# --- CONFIG ---
JECAL = Namespace('https://example.org/jesuit_calvinist/')
dcterms = Namespace("http://purl.org/dc/terms/")
rdf = Namespace("http://www.w3.org/1999/02/22-rdf-syntax-ns#")
FABIO = Namespace("http://purl.org/spar/fabio/")
BIRO = Namespace("http://purl.org/spar/biro/")
VIAF = Namespace("http://viaf.org/viaf/")
geo = Namespace("http://www.w3.org/2003/01/geo/wgs84_pos#")
bibo = Namespace("http://purl.org/ontology/bibo/")
schema = Namespace("http://schema.org/")
WDT = Namespace("http://www.wikidata.org/entity/")
BF = Namespace("http://id.loc.gov/ontologies/bibframe/")
OUTPUT_TTL = "jecal.ttl"

#%% --- LOAD ---
df_texts = gsheet_to_df('1M2gc-8cGZ8gh8TTnm4jl430bL4tccdri9nw-frUWYLQ', 'texts')
texts_ids = df_texts['Work ID'].to_list()
df_people = gsheet_to_df('1M2gc-8cGZ8gh8TTnm4jl430bL4tccdri9nw-frUWYLQ', 'people')
df_places = gsheet_to_df('1M2gc-8cGZ8gh8TTnm4jl430bL4tccdri9nw-frUWYLQ', 'places')
df_events = gsheet_to_df('1M2gc-8cGZ8gh8TTnm4jl430bL4tccdri9nw-frUWYLQ', 'events')
df_genres = gsheet_to_df('1M2gc-8cGZ8gh8TTnm4jl430bL4tccdri9nw-frUWYLQ', 'genres')
#%%
# genres_dict = dict(zip(df_genres['genre_label'].to_list(), df_genres['genre_id'].to_list()))
# genres = [[el.strip() for el in e.split(';')] for e in df_texts['Genre'].to_list()]
# genres = [';'.join([genres_dict.get(el) for el in e]) for e in genres]
# df_test = pd.DataFrame(genres)
#%% --- GRAPH ---
g = Graph()

g.bind("jesuit_calvinist", JECAL)
g.bind("dcterms", dcterms)
g.bind("fabio", FABIO)
g.bind("geo", geo)
g.bind("bibo", bibo)
g.bind("sch", schema)
g.bind("biro", BIRO)
g.bind("foaf", FOAF)
g.bind("wdt", WDT)
g.bind("owl", OWL)
g.bind("bf", BF)

# 1) Genre
def add_genre(row):
    gid = str(row['genre_id'])
    genre = JECAL[f"Genre/{gid}"]
    g.add((genre, RDF.type, BF.GenreForm))
    g.add((genre, schema.name, Literal(row["genre_label"])))
    g.add((genre, JECAL.genreType, Literal(row["genre_type"])))
    
for _, r in df_genres.iterrows():
    add_genre(r)

# 2) Place
def add_place(row):
    pid = str(row["place_id"])
    place = JECAL[f"Place/{pid}"]
    g.add((place, RDF.type, schema.Place))
    g.add((place, schema.name, Literal(row["name"])))
    if pd.notnull(row["wikidata_id"]):
        g.add((place, OWL.sameAs, WDT[row["wikidata_id"]]))
    if pd.notnull(row['latitude']):
        latitude = Literal(row['latitude'], datatype=XSD.float)
        g.add((place, geo.lat, latitude))
        longitude = Literal(row['longitude'], datatype=XSD.float)
        g.add((place, geo.long, longitude))

for _, r in df_places.iterrows():
    add_place(r)

# 3) Event

def add_event(row):
    pid = str(row['event_id'])
    event = JECAL[f"Event/{pid}"]
    g.add((event, RDF.type, schema.Event))
    g.add((event, schema.name, Literal(row['name'])))
    if pd.notnull(row["place_id"]):
        g.add((event, schema.location, JECAL[f"Place/{row['place_id']}"]))
    g.add((event, schema.startDate, Literal(row['year'], datatype=XSD.gYear)))
    g.add((event, schema.endDate, Literal(row['year'], datatype=XSD.gYear)))
    
    Literal('0410', datatype=XSD.gYear)
    g.add((event, schema.additionalType, Literal(row['type'])))

for _, r in df_events.iterrows():
    add_event(r)

# 4) Person
def add_person(row):
    pid = str(row["person_id"])
    person = JECAL[f"Person/{pid}"]
    g.add((person, RDF.type, schema.Person))
    g.add((person, schema.name, Literal(row["person_name"])))
    if pd.notnull(row["person_wikidata_id"]):
        g.add((person, OWL.sameAs, WDT[row["person_wikidata_id"]]))
    if pd.notnull(row['birthdate']):
        year = get_birthyear(row['birthdate'])
        g.add((person, schema.birthDate, Literal(year, datatype=XSD.gYear)))
    if pd.notnull(row['deathdate']):
        year = get_deathyear(row['deathdate'])
        g.add((person, schema.deathDate, Literal(year, datatype=XSD.gYear)))
    if pd.notnull(row['productivity_century']):
        g.add((person, JECAL.productivityCentury, Literal(row['productivity_century'])))
    if pd.notnull(row['productivity_epoch']):
        g.add((person, JECAL.productivityEpoch, Literal(row['productivity_epoch'])))
    if pd.notnull(row["birthplace_id"]):
        place_id = str(row["birthplace_id"])
        g.add((person, schema.birthPlace, JECAL[f"Place/{place_id}"]))
    if pd.notnull(row["deathplace_id"]):
        place_id = str(row["deathplace_id"])
        g.add((person, schema.deathPlace, JECAL[f"Place/{place_id}"]))
    if pd.notnull(row["confession"]):
        g.add((person, JECAL.confession, Literal(row['confession'])))

for _, r in df_people.iterrows():
    add_person(r)

# 5) Text 
def add_text(row):
# for _, row in df_novels.iterrows():
    tid = str(row["Work ID"])
    text = JECAL[f"Text/{tid}"]
    g.add((text, RDF.type, schema.Text))
    for a in row['Author ID'].split(';'):
        g.add((text, schema.author, JECAL[f"Person/{a.strip()}"]))
    if pd.notnull(row['Additional Authors ID']):
        for a in row['Additional Authors ID'].split(';'):
            if a.strip() != 'no id':
                g.add((text, JECAL.additionalAuthor, JECAL[f"Person/{a.strip()}"]))  
    g.add((text, schema.title, Literal(row["Title"])))
    g.add((text, schema.datePublished, Literal(row['Date'], datatype=XSD.gYear)))
    if pd.notnull(row['Date Additional Info']):
        g.add((text, JECAL.dateAdditionalInfo, Literal(row['Date Additional Info'])))
    if pd.notnull(row["place_id"]):
        g.add((text, FABIO.hasPlaceOfPublication, JECAL[f"Place/{row['place_id']}"]))
    if pd.notnull(row['Place Additional Info']):
        g.add((text, JECAL.placeAdditionalInfo, Literal(row['Place Additional Info'])))
    g.add((text, dcterms.publisher, Literal(row['Publisher'])))
    g.add((text, schema.inLanguage, Literal(row['Languages'])))
    g.add((text, JECAL.documentType, Literal(row['Document Type'])))
    # if pd.notnull(row['Genre']):
    #     for genre in row['Genre'].split(';'):
    #         g.add((text, schema.genre, Literal(genre.strip())))
    if pd.notnull(row['Genre ID']):
        for a in row['Genre ID'].split(';'):
            g.add((text, schema.genre, JECAL[f"Genre/{a}"]))
    if row['Number of Pages'] != 'No Data':
        g.add((text, FABIO.hasPageCount, Literal(row['Number of Pages'])))
    for p in row['Digitized Copy'].split(';'):
        g.add((text, JECAL.digitizedCopy, Literal(p.strip())))
    for p in row['Existing Physical Copy'].split(';'):
        g.add((text, JECAL.existingPhysicalCopy, Literal(p.strip())))
    g.add((text, JECAL.confessionalProfile, Literal(row['Confessional Profile'])))
    for a in row['Targeted Confession'].split(';'):
        g.add((text, JECAL.targetedConfession, Literal(a.strip())))
    if row['Preface'] != 'No Data':
        g.add((text, JECAL.preface, Literal(row['Preface'])))   
    if pd.notnull(row['Dedicated to ID']):
        for a in row['Dedicated to ID'].split(';'):
            if a.strip() != 'no id':
                g.add((text, JECAL.dedicatedTo, JECAL[f"Person/{a.strip()}"]))
    if pd.notnull(row['Key Historical Figures Mentioned ID']):
        for a in row['Key Historical Figures Mentioned ID'].split(';'):
            if a.strip() != 'no id':
                g.add((text, JECAL.keyHistoricalFiguresMentioned, JECAL[f"Person/{a.strip()}"]))
    if pd.notnull(row['Key Authors Cited ID']):
        for a in row['Key Authors Cited ID'].split(';'):
            if a.strip() != 'no id':
                g.add((text, JECAL.keyAuthorsCited, JECAL[f"Person/{a.strip()}"]))
    if pd.notnull(row['Tumults and Mentioned']):
        for p in row['Tumults and Mentioned'].split(';'):
            g.add((text, JECAL.tumultMentioned, JECAL[f"Event/{p.strip()}"]))
    if row['Polemical Themes (Random Order)'] != 'No Data':
        for p in row['Polemical Themes (Random Order)'].split(';'):
            g.add((text, JECAL.polemicalTheme, Literal(p.strip())))
    if row['Categorized Polemical Themes (Random Order)'] != 'No Data':
        for p in row['Categorized Polemical Themes (Random Order)'].split(';'):
            g.add((text, JECAL.categorizedPolemicalTheme, Literal(p.strip())))
    if pd.notnull(row['Discussed Issues (Random Order)']) and row['Discussed Issues (Random Order)'] != 'No Data':
        for p in row['Discussed Issues (Random Order)'].split(';'):
            g.add((text, JECAL.discussedIssue, Literal(p.strip())))
    if pd.notnull(row['Response to Author ID']):
        for a in row['Response to Author ID'].split(';'):
            if a.strip() != 'no id':
                g.add((text, JECAL.responseToAuthor, JECAL[f"Person/{a.strip()}"]))
    if pd.notnull(row['Response to Work']):
        for p in row['Response to Work'].split(';'):
            if p.strip() in texts_ids:
                g.add((text, JECAL.responseToWork, JECAL[f"Text/{p.strip()}"]))
            elif p.strip() != 'No Data':
                g.add((text, JECAL.responseToWork, Literal(p.strip())))
    if row['Format'] != "No Data":
        g.add((text, FABIO.hasFormat, Literal(row['Format'])))
    if pd.notnull(row['Notes']):
        g.add((text, JECAL.notes, Literal(row['Notes'])))
    for p in row['Edition Type'].split(';'):
        g.add((text, JECAL.editionType, Literal(p.strip())))
           
for _, r in df_texts.iterrows():
    add_text(r)

# --- EXPORT ---
g.serialize(destination=OUTPUT_TTL, format="turtle")
print(f"RDF triples written to {OUTPUT_TTL}")

#%% RDF filtering for the article

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Filtr RDF: wybiera schema:Text z editionType zawierającym 'Original' (case-insensitive),
buduje pełny podgraf (ABox) z wszystkimi atrybutami bytów użytych w tych tekstach,
oraz graf klas (TBox-like) obserwowanych na krawędziach w podgrafie.

Gotowy do uruchomienia w Spyderze – wystarczy nacisnąć F5.
"""

from rdflib import Graph, Namespace, RDF, RDFS, URIRef, BNode, Literal
from pathlib import Path
from collections import deque

# ------------------------------------------------------------
# 🔧 USTAWIENIA ŚCIEŻEK – ZMODYFIKUJ WG SWOJEGO KATALOGU
# ------------------------------------------------------------
INPUT_PATH = Path(r"jecal.ttl")  # <- zmień ścieżkę, jeśli trzeba
OUT_SUBGRAPH_PATH = Path(r"jecal_original_texts_subgraph.ttl")
OUT_CLASSES_PATH = Path(r"jecal_original_texts_classes.ttl")
INPUT_FORMAT = "turtle"  # zmień na np. "xml" jeśli plik RDF jest w RDF/XML

# ------------------------------------------------------------
# 🔧 NAZWY PRZESTRZENi
# ------------------------------------------------------------
SCHEMA = Namespace("http://schema.org/")
JC = Namespace("https://example.org/jesuit_calvinist/")

# ------------------------------------------------------------
# 🔍 FUNKCJE POMOCNICZE
# ------------------------------------------------------------
def literal_contains_original(lit: Literal) -> bool:
    """Sprawdza, czy literal zawiera słowo 'Original' (bez rozróżniania wielkości liter)."""
    try:
        return "original" in str(lit).lower()
    except Exception:
        return False

def classes_of(g: Graph, node) -> set:
    """Zwraca zbiór klas (rdf:type) dla danego węzła."""
    return set(g.objects(node, RDF.type))

def add_all_triples_of_subject(g_src: Graph, g_dst: Graph, s) -> set:
    """
    Dodaje do g_dst wszystkie trójki, gdzie 's' jest podmiotem, i zwraca zbiór nowych węzłów
    (obiektów), które są IRI lub BNode – do dalszej eksploracji.
    """
    new_nodes = set()
    for p, o in g_src.predicate_objects(s):
        g_dst.add((s, p, o))
        if isinstance(o, (URIRef, BNode)):
            new_nodes.add(o)
    # Zawsze dodajemy typy danego węzła
    for c in g_src.objects(s, RDF.type):
        g_dst.add((s, RDF.type, c))
    return new_nodes

# ------------------------------------------------------------
# 🚀 GŁÓWNA LOGIKA
# ------------------------------------------------------------
def main():
    print("📘 Wczytywanie pliku RDF...")
    g = Graph()
    g.parse(INPUT_PATH, format=INPUT_FORMAT)
    print(f"✅ Załadowano {len(g)} trójek RDF z pliku {INPUT_PATH}")

    # 1) Znajdź wszystkie schema:Text, które mają editionType zawierające 'Original'
    texts = set(
        s for s in g.subjects(RDF.type, SCHEMA.Text)
        if any(literal_contains_original(o) for o in g.objects(s, JC.editionType))
    )
    print(f"🔎 Znaleziono {len(texts)} obiektów schema:Text z editionType zawierającym 'Original'")

    # 2) Zbuduj pełny subgraf (ABox) – domknięcie po krawędziach „do przodu”
    subg = Graph()
    # skopiuj przestrzenie nazw
    for prefix, ns in g.namespaces():
        subg.bind(prefix, ns)

    visited = set()               # odwiedzone węzły (podmioty)
    q = deque(texts)              # zaczynamy od wybranych tekstów

    while q:
        s = q.popleft()
        if s in visited:
            continue
        visited.add(s)
        # dodaj wszystkie trójki, gdzie s jest podmiotem; zbierz nowe węzły do eksploracji
        new_nodes = add_all_triples_of_subject(g, subg, s)
        # kontynuuj eksplorację po IRI/BNode w obiektach
        for n in new_nodes:
            # Jeśli node ma również własne atrybuty jako podmiot, też je dociągamy
            q.append(n)

    print(f"💾 Zbudowano subgraf (ABox) z {len(subg)} trójkami dla {len(visited)} węzłów")

    subg.serialize(destination=OUT_SUBGRAPH_PATH, format="turtle")
    print(f"💾 Zapisano subgraf → {OUT_SUBGRAPH_PATH}")

    # 3) Zbuduj graf klas (TBox-like) na podstawie subgrafu
    class_graph = Graph()
    for prefix, ns in g.namespaces():
        class_graph.bind(prefix, ns)

    observed_edges = set()
    for s, p, o in subg:
        # ignorujemy trójki typujące i trójki, gdzie obiekt nie jest IRI (literal/BNode nie tworzy łuku między klasami)
        if p == RDF.type or not isinstance(o, URIRef):
            continue
        s_classes = classes_of(g, s)
        o_classes = classes_of(g, o)
        for cs in s_classes:
            for co in o_classes:
                observed_edges.add((cs, p, co))

    for cs, p, co in observed_edges:
        class_graph.add((cs, p, co))
        class_graph.add((cs, RDF.type, RDFS.Class))
        class_graph.add((co, RDF.type, RDFS.Class))

    class_graph.serialize(destination=OUT_CLASSES_PATH, format="turtle")
    print(f"💾 Zapisano graf klas (TBox-like) → {OUT_CLASSES_PATH} ({len(class_graph)} trójek)")

    print("\n✅ Operacja zakończona pomyślnie!")

# ------------------------------------------------------------
# ▶️ URUCHOMIENIE
# ------------------------------------------------------------
if __name__ == "__main__":
    main()

























