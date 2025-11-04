from rdflib import Graph, Namespace, URIRef
from rdflib.namespace import RDF
import logging
logging.getLogger("rdflib.term").setLevel(logging.ERROR)
import pandas as pd
import networkx as nx
from tqdm import tqdm
from ipysigma import Sigma

#%% statystyki do artykułu
##cała baza
g = Graph()
g.parse("jecal.ttl", format="turtle")
SCHEMA = Namespace("http://schema.org/")

texts = [str(s) for s in g.subjects(RDF.type, SCHEMA.Text)]
print(len(texts), "schema:Text objects found")

authors = set()
for text in g.subjects(RDF.type, SCHEMA.Text):
    for person in g.objects(text, SCHEMA.author):
        if (person, RDF.type, SCHEMA.Person) in g:
            authors.add(str(person))

response_to_authors = URIRef("https://example.org/jesuit_calvinist/responseToAuthor")

response_authors = set()
for text in g.subjects(RDF.type, SCHEMA.Text):
    for person in g.objects(text, response_to_authors):
        if (person, RDF.type, SCHEMA.Person) in g:
            response_authors.add(str(person))

authors = authors | response_authors

print(len(authors), "Persons are authors and response authors of schema:Text")

KEY_AUTHORS_CITED = URIRef("https://example.org/jesuit_calvinist/keyAuthorsCited")

cited_authors = set()
for text in g.subjects(RDF.type, SCHEMA.Text):
    for person in g.objects(text, KEY_AUTHORS_CITED):
        if (person, RDF.type, SCHEMA.Person) in g:
            cited_authors.add(str(person))

print(len(cited_authors), "Persons are cited as keyAuthorsCited")

KEY_HISTORICAL_FIGURES = URIRef("https://example.org/jesuit_calvinist/keyHistoricalFiguresMentioned")

historical_figures = set()
for text in g.subjects(RDF.type, SCHEMA.Text):
    for person in g.objects(text, KEY_HISTORICAL_FIGURES):
        if (person, RDF.type, SCHEMA.Person) in g:
            historical_figures.add(str(person))

print(len(historical_figures), "Persons are mentioned as keyHistoricalFiguresMentioned")

KEY_POLEMICAL_THEME = URIRef("https://example.org/jesuit_calvinist/categorizedPolemicalTheme")

themes = set()
for text in g.subjects(RDF.type, SCHEMA.Text):
    for theme in g.objects(text, KEY_POLEMICAL_THEME):
        themes.add(str(theme))

print(len(themes), "categorizedPolemicalTheme objects found")

POLEMICAL_THEME = URIRef("https://example.org/jesuit_calvinist/polemicalTheme")

themes = set()
for text in g.subjects(RDF.type, SCHEMA.Text):
    for theme in g.objects(text, POLEMICAL_THEME):
        themes.add(str(theme))

print(len(themes), "polemicalTheme objects found")

DISCUSSED_ISSUE = URIRef("https://example.org/jesuit_calvinist/discussedIssue")

issues = set()
for text in g.subjects(RDF.type, SCHEMA.Text):
    for issue in g.objects(text, DISCUSSED_ISSUE):
        issues.add(str(issue))

print(len(issues), "discussedIssue objects found")


SCHEMA = Namespace("http://schema.org/")
BASE = "https://example.org/jesuit_calvinist/"

# Definicje właściwości
PROPERTIES = {
    "schema:author": SCHEMA.author,
    "discussedIssue": URIRef(BASE + "discussedIssue"),
    "polemicalTheme": URIRef(BASE + "polemicalTheme"),
    "keyHistoricalFiguresMentioned": URIRef(BASE + "keyHistoricalFiguresMentioned"),
    "keyAuthorsCited": URIRef(BASE + "keyAuthorsCited"),
}

# Liczenie połączeń (krawędzi)
connections = {}

for label, prop in PROPERTIES.items():
    count = 0
    for text in g.subjects(RDF.type, SCHEMA.Text):
        for _ in g.objects(text, prop):
            count += 1
    connections[label] = count

# Wyniki
total_count = 0
for label, count in connections.items():
    total_count += count
    print(f"{label}: {count} connections")
    
print(f"All connections in the graph: {total_count}")

print('________________________________')

##wyfiltrowana baza
out = Graph()
out.parse("jecal_original_texts_subgraph.ttl", format="turtle")
SCHEMA = Namespace("http://schema.org/")

texts = [str(s) for s in out.subjects(RDF.type, SCHEMA.Text)]
print(len(texts), "schema:Text objects found")

authors = set()
for text in out.subjects(RDF.type, SCHEMA.Text):
    for person in out.objects(text, SCHEMA.author):
        if (person, RDF.type, SCHEMA.Person) in out:
            authors.add(str(person))
            
response_to_authors = URIRef("https://example.org/jesuit_calvinist/responseToAuthor")

response_authors = set()
for text in out.subjects(RDF.type, SCHEMA.Text):
    for person in out.objects(text, response_to_authors):
        if (person, RDF.type, SCHEMA.Person) in out:
            response_authors.add(str(person))

authors = authors | response_authors

print(len(authors), "Persons are authors and response to authors of schema:Text")

KEY_AUTHORS_CITED = URIRef("https://example.org/jesuit_calvinist/keyAuthorsCited")

cited_authors = set()
for text in out.subjects(RDF.type, SCHEMA.Text):
    for person in out.objects(text, KEY_AUTHORS_CITED):
        if (person, RDF.type, SCHEMA.Person) in out:
            cited_authors.add(str(person))

print(len(cited_authors), "Persons are cited as keyAuthorsCited")

KEY_HISTORICAL_FIGURES = URIRef("https://example.org/jesuit_calvinist/keyHistoricalFiguresMentioned")

historical_figures = set()
for text in out.subjects(RDF.type, SCHEMA.Text):
    for person in out.objects(text, KEY_HISTORICAL_FIGURES):
        if (person, RDF.type, SCHEMA.Person) in out:
            historical_figures.add(str(person))

print(len(historical_figures), "Persons are mentioned as keyHistoricalFiguresMentioned")

KEY_POLEMICAL_THEME = URIRef("https://example.org/jesuit_calvinist/categorizedPolemicalTheme")

themes = set()
for text in out.subjects(RDF.type, SCHEMA.Text):
    for theme in out.objects(text, KEY_POLEMICAL_THEME):
        themes.add(str(theme))

print(len(themes), "categorizedPolemicalTheme objects found")

POLEMICAL_THEME = URIRef("https://example.org/jesuit_calvinist/polemicalTheme")

themes = set()
for text in out.subjects(RDF.type, SCHEMA.Text):
    for theme in out.objects(text, POLEMICAL_THEME):
        themes.add(str(theme))

print(len(themes), "polemicalTheme objects found")

DISCUSSED_ISSUE = URIRef("https://example.org/jesuit_calvinist/discussedIssue")

issues = set()
for text in out.subjects(RDF.type, SCHEMA.Text):
    for issue in out.objects(text, DISCUSSED_ISSUE):
        issues.add(str(issue))

print(len(issues), "discussedIssue objects found")


SCHEMA = Namespace("http://schema.org/")
BASE = "https://example.org/jesuit_calvinist/"

# Definicje właściwości
PROPERTIES = {
    "schema:author": SCHEMA.author,
    "responseToAuthor": URIRef(BASE + "responseToAuthor"),
    "discussedIssue": URIRef(BASE + "discussedIssue"),
    "polemicalTheme": URIRef(BASE + "polemicalTheme"),
    "keyHistoricalFiguresMentioned": URIRef(BASE + "keyHistoricalFiguresMentioned"),
    "keyAuthorsCited": URIRef(BASE + "keyAuthorsCited"),
}

# Liczenie połączeń (krawędzi)
connections = {}

for label, prop in PROPERTIES.items():
    count = 0
    for text in out.subjects(RDF.type, SCHEMA.Text):
        for _ in out.objects(text, prop):
            count += 1
    connections[label] = count

# Wyniki
total_count = 0
for label, count in connections.items():
    total_count += count
    print(f"{label}: {count} connections")

print(f"All connections in the graph: {total_count}")


## predykaty całej bazy
ps = set()
for s, p, o in g:
    ps.add(str(p))

#%% 1 autorzy i odbiorcy polemik

SCHEMA = Namespace("http://schema.org/")
# Tworzenie słownika: {osoba: name}
people_names = {}
for subject, name in g.subject_objects(predicate=SCHEMA.name):
    people_names[subject] = str(name)

text_author = dict()
text_response = dict()
# for s, p, o in g:
for s, p, o in out:
    if str(p) == str(SCHEMA.author):
        if str(s) not in text_author:
            text_author[str(s)] = [people_names.get(o)]
        else: text_author[str(s)].append(people_names.get(o))
    elif str(p) == 'https://example.org/jesuit_calvinist/responseToAuthor':
        if str(s) not in text_response:
            text_response[str(s)] = [people_names.get(o)]
        else: text_response[str(s)].append(people_names.get(o))

result_29a = []
for k, v in text_author.items():
    for ka, va in text_response.items():
        if k == ka:
            for a in v:
                for t in va:
                    result_29a.append([a, t])

df = pd.DataFrame(result_29a, columns= ['person', 'response_person'])

df = df.loc[~df['person'].isin(['Anonymous', 'Various']) &
            ~df['response_person'].isin(['Anonymous', 'Various'])]

co_occurrence = df.groupby(['person', 'response_person']).size().reset_index(name='weight')
co_occurrence.to_excel('data/article_lpg_1.xlsx', index=False)

# Tworzymy graf
G = nx.Graph()

# Dodajemy węzły i krawędzie z wagami
for _, row in tqdm(co_occurrence.iterrows(), total = len(co_occurrence)):
    person = row['person']
    response_person = row['response_person']
    weight = row['weight']

    # Węzły
    G.add_node(person, type='person', color='#e41a1c', size=10)
    G.add_node(response_person, type='response_person', color='#377eb8', size=7)

    # Krawędź z wagą
    G.add_edge(person, response_person, weight=weight)


# Dodajemy stopnie jako metrykę (do rozmiarów i etykiet)
degree_dict = dict(G.degree())
nx.set_node_attributes(G, degree_dict, name='degree')

# ✅ Interaktywny podgląd w Jupyterze (opcjonalnie)
Sigma(
    G,
    node_color="tag",
    node_label_size="degree",
    node_size="degree"
)

# ✅ Eksport do HTML
Sigma.write_html(
    G,
    'data/article_lpg_1.html',
    fullscreen=True,
    node_metrics=['louvain'],
    node_color='louvain',
    node_size='degree',
    node_size_range=(3, 20),
    max_categorical_colors=30,
    default_edge_type='curve',
    default_node_label_size=14,
    node_border_color_from='node'
)

#%% 1a. autorzy i odbiorcy polemik + parametr czasu

from rdflib import Namespace, Literal
import pandas as pd

SCHEMA = Namespace("http://schema.org/")
JECAL  = Namespace("https://example.org/jesuit_calvinist/")

# 1) słownik osób: URI -> nazwa
people_names = {}
for subject, name in g.subject_objects(predicate=SCHEMA.name):
    people_names[subject] = str(name)

# 2) słowniki dla autora, adresata odpowiedzi i daty wydania
text_author   = {}   # {str(text_uri): [author_name, ...]}
text_response = {}   # {str(text_uri): [target_name, ...]}
text_date     = {}   # {str(text_uri): "YYYY" lub pełna data}

# jeśli masz iterator `out`, użyj go; w przeciwnym razie iteruj po całym grafie
triple_iter = out if 'out' in globals() and out is not None else g

for s, p, o in triple_iter:
    s_key = str(s)

    # autor(zy)
    if str(p) == str(SCHEMA.author):
        name = people_names.get(o) or str(o)
        text_author.setdefault(s_key, []).append(name)

    # adresat(ka) polemiki (responseToAuthor)
    elif str(p) == str(JECAL['responseToAuthor']):
        name = people_names.get(o) or str(o)
        text_response.setdefault(s_key, []).append(name)

    # data publikacji
    elif str(p) == str(SCHEMA.datePublished):
        # weź pierwszą napotkaną sensowną wartość
        if s_key not in text_date:
            # Literal -> str; zostaw jak jest (rok lub pełna data)
            text_date[s_key] = str(o) if isinstance(o, Literal) else str(o)

# 3) składanie wyników: (autor, adresat, data_tekstu)
rows = []
for k, authors in text_author.items():
    targets = text_response.get(k, [])
    if not targets:
        continue
    pub_date = text_date.get(k)  # może być None, jeśli brak w danych
    for a in authors:
        for t in targets:
            rows.append([a, t, pub_date])

df = pd.DataFrame(rows, columns=['person', 'response_person', 'date_published'])
#
import pandas as pd
import numpy as np
import re

# Zakładamy, że masz DataFrame df z kolumnami:
# ['person', 'response_person', 'date_published']

# --- 1) Ekstrakcja roku z daty ---
def extract_year(x):
    if pd.isna(x):
        return np.nan
    s = str(x)
    m = re.search(r'(\d{4})', s)
    return int(m.group(1)) if m else np.nan

df = df.copy()
df['year'] = df['date_published'].apply(extract_year)
df = df.dropna(subset=['year']).assign(year=lambda d: d['year'].astype(int))

# --- 2) Definicja okresów ---
bins = [
    (1574, 1583),
    (1584, 1593),
    (1594, 1603),
    (1604, 1613),
    (1614, 1623),
    (1624, 1633),
    (1634, 1647),
]
labels = [f"{a}–{b}" for a,b in bins]

def bin_label(y):
    for (a,b), lab in zip(bins, labels):
        if a <= y <= b:
            return lab
    return None

df['period'] = df['year'].apply(bin_label)
df = df.dropna(subset=['period'])

# --- 3) Filtr: usuwamy osoby "Various" i "Anonymous" ---
exclude = {"Various", "Anonymous"}
df = df[~df['person'].isin(exclude)]
df = df[~df['response_person'].isin(exclude)]

# --- 4) Liczenie relacji ---
relation_counts = (
    df.groupby('period')
      .size()
      .reindex(labels, fill_value=0)
      .reset_index(name='count_relations')
)

# --- 5) Liczenie unikalnych osób (autorów + adresatów) ---
persons = df[['person', 'period']].rename(columns={'person': 'person_name'})
targets = df[['response_person', 'period']].rename(columns={'response_person': 'person_name'})

participants = pd.concat([persons, targets], ignore_index=True)
participants = participants.dropna(subset=['person_name'])

# usuń "Various"/"Anonymous" także tutaj, na wszelki wypadek
participants = participants[~participants['person_name'].isin(exclude)]

people_counts = (
    participants.groupby('period')['person_name']
               .nunique()
               .reindex(labels, fill_value=0)
               .reset_index(name='unique_people')
)

# --- 6) Łączymy wyniki ---
result = relation_counts.merge(people_counts, on='period')
result["people_per_relation"] = (result["unique_people"] / result["count_relations"]).round(2)

print(result)

#%% 2. współcytowanie autorów cytowanych

from rdflib import Namespace, Literal
from itertools import combinations
import pandas as pd

SCHEMA = Namespace("http://schema.org/")
JECAL  = Namespace("https://example.org/jesuit_calvinist/")

# 1) słownik osób: URI -> nazwa (schema:name); fallback: URI jako tekst
people_names = {}
for subj, name in g.subject_objects(predicate=SCHEMA.name):
    people_names[subj] = str(name)

# 2) zbieramy cytowanych autorów w obrębie każdego tekstu
#    oraz (opcjonalnie) datę publikacji
text_cited = {}   # {str(text_uri): [name1, name2, ...]}
text_date  = {}   # {str(text_uri): "YYYY" lub pełna data}
exclude = {"Various", "Anonymous"}

triple_iter = out if 'out' in globals() and out is not None else g

for s, p, o in triple_iter:
    s_key = str(s)

    # keyAuthorsCited -> osoba cytowana
    if str(p) == str(JECAL['keyAuthorsCited']):
        name = people_names.get(o) or str(o)
        if name not in exclude:
            text_cited.setdefault(s_key, []).append(name)

    # (opcjonalnie) data publikacji
    elif str(p) == str(SCHEMA.datePublished) and s_key not in text_date:
        text_date[s_key] = str(o)

# 3) w obrębie każdego tekstu: unikalna lista osób i wszystkie pary (combos 2)
rows = []
for text_uri, names in text_cited.items():
    unique_names = sorted(set(n for n in names if n and n not in exclude))
    if len(unique_names) < 2:
        continue
    for a, b in combinations(unique_names, 2):
        rows.append({
            "text_uri": text_uri,
            "person_a": a,
            "person_b": b,
            "date_published": text_date.get(text_uri)
        })

pairs_by_text = pd.DataFrame(rows).drop(columns=['date_published'])
pairs_by_text.to_excel('data/article_lpg_2_with_texts.xlsx', index=False)

# 4) sieć współ-cytowań (agregacja po parach, niezależnie od tekstu)
if not pairs_by_text.empty:
    cocitation_edges = (
        pairs_by_text
        .groupby(["person_a", "person_b"], as_index=False)
        .size()
        .rename(columns={"size": "weight"})  # waga = liczba tekstów, w których para współ-występuje
    )
else:
    cocitation_edges = pd.DataFrame(columns=["person_a","person_b","weight"])

# (opcjonalnie) wyciągnij rok jako int
import re, numpy as np
def extract_year(x):
    if pd.isna(x): return np.nan
    m = re.search(r'(\d{4})', str(x))
    return int(m.group(1)) if m else np.nan

if "date_published" in pairs_by_text.columns:
    pairs_by_text["year"] = pairs_by_text["date_published"].apply(extract_year)

co_occurrence = pairs_by_text[['person_a', 'person_b']]
co_occurrence = pairs_by_text.groupby(['person_a', 'person_b']).size().reset_index(name='weight')
co_occurrence.to_excel('data/article_lpg_2.xlsx', index=False)

# Tworzymy graf
G = nx.Graph()

# Dodajemy węzły i krawędzie z wagami
for _, row in tqdm(co_occurrence.iterrows(), total = len(co_occurrence)):
    person = row['person_a']
    response_person = row['person_b']
    weight = row['weight']

    # Węzły
    G.add_node(person, type='person_a', color='#e41a1c', size=10)
    G.add_node(response_person, type='person_b', color='#377eb8', size=7)

    # Krawędź z wagą
    G.add_edge(person, response_person, weight=weight)


# Dodajemy stopnie jako metrykę (do rozmiarów i etykiet)
degree_dict = dict(G.degree())
nx.set_node_attributes(G, degree_dict, name='degree')

# ✅ Interaktywny podgląd w Jupyterze (opcjonalnie)
Sigma(
    G,
    node_color="tag",
    node_label_size="degree",
    node_size="degree"
)

# ✅ Eksport do HTML
Sigma.write_html(
    G,
    'data/article_lpg_2.html',
    fullscreen=True,
    node_metrics=['louvain'],
    node_color='louvain',
    node_size='degree',
    node_size_range=(3, 20),
    max_categorical_colors=30,
    default_edge_type='curve',
    default_node_label_size=14,
    node_border_color_from='node'
)


SCHEMA = Namespace("http://schema.org/")
# Tworzenie słownika: {osoba: name}
people_names = {}
for subject, name in g.subject_objects(predicate=SCHEMA.name):
    people_names[str(subject)] = str(name)
# test = sorted(set([e for e in pairs_by_text['person_a'].to_list()]))
# test2 = [people_names.get(e) for e in cited_authors]

# test3 = set(pairs_by_text['person_a'].to_list() + pairs_by_text['person_b'].to_list())


KEY_AUTHORS_CITED = URIRef("https://example.org/jesuit_calvinist/keyAuthorsCited")

cited_authors = []
for text in out.subjects(RDF.type, SCHEMA.Text):
    for person in g.objects(text, KEY_AUTHORS_CITED):
        if (person, RDF.type, SCHEMA.Person) in out:
            if str(person) not in ['https://example.org/jesuit_calvinist/Person/P1440', 'https://example.org/jesuit_calvinist/Person/P61']:
                # print(str(person))
                cited_authors.append(str(person))

print(len(cited_authors), "Persons are cited as keyAuthorsCited")

from collections import Counter
cited_authors_counted = dict(Counter([people_names.get(e) for e in cited_authors]))

#%% 3. współwystępowanie motywów

SCHEMA = Namespace("http://schema.org/")
JECAL  = Namespace("https://example.org/jesuit_calvinist/")

# 2) zbieramy cytowanych autorów w obrębie każdego tekstu
#    oraz (opcjonalnie) datę publikacji
text_theme = {}   # {str(text_uri): [name1, name2, ...]}
text_date  = {} 

triple_iter = out if 'out' in globals() and out is not None else g

for s, p, o in triple_iter:
    s_key = str(s)

    # polemicalTheme -> osoba cytowana
    if str(p) == str(JECAL['polemicalTheme']):
        name = str(o)
        text_theme.setdefault(s_key, []).append(name)
    # (opcjonalnie) data publikacji
    elif str(p) == str(SCHEMA.datePublished) and s_key not in text_date:
        text_date[s_key] = str(o)

# 3) w obrębie każdego tekstu: unikalna lista osób i wszystkie pary (combos 2)
rows = []
for text_uri, themes in text_theme.items():
    unique_themes = sorted(set(n for n in themes if n and n not in exclude))
    if len(unique_themes) < 2:
        continue
    for a, b in combinations(unique_themes, 2):
        rows.append({
            "text_uri": text_uri,
            "theme_a": a,
            "theme_b": b,
            "date_published": text_date.get(text_uri)
        })

pairs_by_text = pd.DataFrame(rows).drop(columns=['text_uri'])
pairs_by_text.to_excel('data/article_lpg_3_with_dates.xlsx', index=False)

co_occurrence = pairs_by_text[['theme_a', 'theme_b']]
co_occurrence = pairs_by_text.groupby(['theme_a', 'theme_b']).size().reset_index(name='weight')
co_occurrence.to_excel('data/article_lpg_3.xlsx', index=False)

# Tworzymy graf
G = nx.Graph()

# Dodajemy węzły i krawędzie z wagami
for _, row in tqdm(co_occurrence.iterrows(), total = len(co_occurrence)):
    person = row['theme_a']
    response_person = row['theme_b']
    weight = row['weight']

    # Węzły
    G.add_node(person, type='theme_a', color='#e41a1c', size=10)
    G.add_node(response_person, type='theme_b', color='#377eb8', size=7)

    # Krawędź z wagą
    G.add_edge(person, response_person, weight=weight)


# Dodajemy stopnie jako metrykę (do rozmiarów i etykiet)
degree_dict = dict(G.degree())
nx.set_node_attributes(G, degree_dict, name='degree')

# ✅ Interaktywny podgląd w Jupyterze (opcjonalnie)
Sigma(
    G,
    node_color="tag",
    node_label_size="degree",
    node_size="degree"
)

# ✅ Eksport do HTML
Sigma.write_html(
    G,
    'data/article_lpg_3.html',
    fullscreen=True,
    node_metrics=['louvain'],
    node_color='louvain',
    node_size='degree',
    node_size_range=(3, 20),
    max_categorical_colors=30,
    default_edge_type='curve',
    default_node_label_size=14,
    node_border_color_from='node'
)

POLEMICAL_THEME = URIRef("https://example.org/jesuit_calvinist/polemicalTheme")

polemical_themes = []
for text in out.subjects(RDF.type, SCHEMA.Text):
    for theme in g.objects(text, POLEMICAL_THEME):
                # print(str(person))
        polemical_themes.append(str(theme))

print(len(polemical_themes), "Persons are cited as keyAuthorsCited")

from collections import Counter
polemical_themes_counted = dict(Counter(polemical_themes))

85/4909
76/4909
74/4909
70/4909
67/4909
60/4909
59/4909
57/4909
56/4909
52/4909

(85+76+74+70+67+60+59+57+56+52+52)/4909

#%% 3a. współwystępowanie tematu + parametr czasu

SCHEMA = Namespace("http://schema.org/")
JECAL  = Namespace("https://example.org/jesuit_calvinist/")

# 2) zbieramy cytowanych autorów w obrębie każdego tekstu
#    oraz (opcjonalnie) datę publikacji
text_theme = {}   # {str(text_uri): [name1, name2, ...]}
text_date  = {} 

triple_iter = out if 'out' in globals() and out is not None else g

for s, p, o in triple_iter:
    s_key = str(s)

    # polemicalTheme -> osoba cytowana
    if str(p) == str(JECAL['polemicalTheme']):
        name = str(o)
        text_theme.setdefault(s_key, []).append(name)
    # (opcjonalnie) data publikacji
    elif str(p) == str(SCHEMA.datePublished) and s_key not in text_date:
        text_date[s_key] = str(o)

# 3) w obrębie każdego tekstu: unikalna lista osób i wszystkie pary (combos 2)
rows = []
for text_uri, themes in text_theme.items():
    unique_themes = sorted(set(n for n in themes if n and n not in exclude))
    if len(unique_themes) < 2:
        continue
    for a, b in combinations(unique_themes, 2):
        rows.append({
            "text_uri": text_uri,
            "theme_a": a,
            "theme_b": b,
            "date_published": text_date.get(text_uri)
        })

pairs_by_text = pd.DataFrame(rows).drop(columns=['text_uri'])
#
import pandas as pd
import numpy as np
import re

# Zakładamy, że masz DataFrame df z kolumnami:
# ['person', 'response_person', 'date_published']

# --- 1) Ekstrakcja roku z daty ---
def extract_year(x):
    if pd.isna(x):
        return np.nan
    s = str(x)
    m = re.search(r'(\d{4})', s)
    return int(m.group(1)) if m else np.nan

df = pairs_by_text.copy()
df['year'] = df['date_published'].apply(extract_year)
df = df.dropna(subset=['year']).assign(year=lambda d: d['year'].astype(int))

# --- 2) Definicja okresów ---
bins = [
    (1574, 1583),
    (1584, 1593),
    (1594, 1603),
    (1604, 1613),
    (1614, 1623),
    (1624, 1633),
    (1634, 1647),
]
labels = [f"{a}–{b}" for a,b in bins]

def bin_label(y):
    for (a,b), lab in zip(bins, labels):
        if a <= y <= b:
            return lab
    return None

df['period'] = df['year'].apply(bin_label)
df = df.dropna(subset=['period'])

# --- 4) Liczenie relacji ---
relation_counts = (
    df.groupby('period')
      .size()
      .reindex(labels, fill_value=0)
      .reset_index(name='count_relations')
)

# --- 5) Liczenie unikalnych osób (autorów + adresatów) ---
persons = df[['theme_a', 'period']].rename(columns={'theme_a': 'theme'})
targets = df[['theme_b', 'period']].rename(columns={'theme_b': 'theme'})

participants = pd.concat([persons, targets], ignore_index=True)
participants = participants.dropna(subset=['theme'])

# usuń "Various"/"Anonymous" także tutaj, na wszelki wypadek
participants = participants[~participants['theme'].isin(exclude)]

people_counts = (
    participants.groupby('period')['theme']
               .nunique()
               .reindex(labels, fill_value=0)
               .reset_index(name='unique_themes')
)

# --- 6) Łączymy wyniki ---
result = relation_counts.merge(people_counts, on='period')
result["themes_per_relation"] = (result["unique_themes"] / result["count_relations"]).round(2)

print(result)
####
from rdflib import Graph, Namespace, URIRef
from rdflib.namespace import RDF
from rdflib.term import Literal
from collections import defaultdict, OrderedDict
from typing import Dict, Tuple, List
import re
import json

# ===== USTAWIENIA =====
SCHEMA = Namespace("http://schema.org/")
POLEMICAL = URIRef("https://example.org/jesuit_calvinist/polemicalTheme")

# Zdefiniowane okresy (przedziały domknięte)
PERIODS: List[Tuple[int, int]] = [
    (1574, 1583),
    (1584, 1593),
    (1594, 1603),
    (1604, 1613),
    (1614, 1623),
    (1624, 1633),
    (1634, 1647),
]
PERIOD_LABELS = [f"{a}-{b}" for a, b in PERIODS]


# ===== FUNKCJE POMOCNICZE =====
def extract_year(date_literal) -> int | None:
    """Zwraca rok (int) z literalu daty (np. '1602', '1602-03-15'); None jeśli się nie uda."""
    if date_literal is None:
        return None
    s = str(date_literal).strip()
    m = re.match(r"^\s*(\d{4})", s)
    return int(m.group(1)) if m else None


def period_label_for_year(y: int) -> str | None:
    """Mapuje rok na etykietę okresu, lub None gdy rok poza zasięgiem."""
    for (a, b), label in zip(PERIODS, PERIOD_LABELS):
        if a <= y <= b:
            return label
    return None


def build_period_theme_dict(g) -> Dict[str, Dict[str, int]]:
    # Przygotuj wynik z ustaloną kolejnością okresów
    result: Dict[str, Dict[str, int]] = OrderedDict((label, defaultdict(int)) for label in PERIOD_LABELS)

    # Iteruj po wszystkich schema:Text
    text_nodes = set(g.subjects(RDF.type, SCHEMA.Text))

    for subj in text_nodes:
        # Rok publikacji (weź pierwszy dostępny schema:datePublished)
        date_obj = next(g.objects(subj, SCHEMA.datePublished), None)
        year = extract_year(date_obj)
        if year is None:
            continue

        label = period_label_for_year(year)
        if label is None:
            continue

        # Każdy literalny theme = jedno zliczenie
        for theme_obj in g.objects(subj, POLEMICAL):
            if isinstance(theme_obj, Literal):
                theme = str(theme_obj).strip()
                if theme:
                    result[label][theme] += 1

    # Zamień defaultdict na zwykłe dict dla czystego outputu
    return OrderedDict(
        (period, dict(themes)) for period, themes in result.items()
    )

nested_counts = build_period_theme_dict(out)

# sum({sum(va for ka,va in v.items()) for k,v in nested_counts.items()})

##

term = "Audacity"
temp_dict = {}
for k,v in nested_counts.items():
    for ka,va in v.items():
        if ka == term:
            temp_dict.update({k:va})
    total = sum(temp_dict.values())
for kb,vb in temp_dict.items():
    print(kb, vb, round(vb/total*100,1))

#%% 4. współwystępowanie discussed issues

SCHEMA = Namespace("http://schema.org/")
JECAL  = Namespace("https://example.org/jesuit_calvinist/")

# 2) zbieramy cytowanych autorów w obrębie każdego tekstu
#    oraz (opcjonalnie) datę publikacji
text_issue = {}   # {str(text_uri): [name1, name2, ...]}
text_date  = {} 

triple_iter = out if 'out' in globals() and out is not None else g

for s, p, o in triple_iter:
    s_key = str(s)

    # polemicalTheme -> osoba cytowana
    if str(p) == str(JECAL['discussedIssue']):
        name = str(o)
        text_issue.setdefault(s_key, []).append(name)
    # (opcjonalnie) data publikacji
    elif str(p) == str(SCHEMA.datePublished) and s_key not in text_date:
        text_date[s_key] = str(o)

# 3) w obrębie każdego tekstu: unikalna lista osób i wszystkie pary (combos 2)
rows = []
for text_uri, themes in text_issue.items():
    unique_themes = sorted(set(n for n in themes if n and n not in exclude))
    if len(unique_themes) < 2:
        continue
    for a, b in combinations(unique_themes, 2):
        rows.append({
            "text_uri": text_uri,
            "issue_a": a,
            "issue_b": b,
            "date_published": text_date.get(text_uri)
        })

pairs_by_text = pd.DataFrame(rows).drop(columns=['text_uri'])
pairs_by_text.to_excel('data/article_lpg_4_with_dates.xlsx', index=False)

co_occurrence = pairs_by_text[['issue_a', 'issue_b']]
co_occurrence = pairs_by_text.groupby(['issue_a', 'issue_b']).size().reset_index(name='weight')
co_occurrence.to_excel('data/article_lpg_4.xlsx', index=False)

# Tworzymy graf
G = nx.Graph()

# Dodajemy węzły i krawędzie z wagami
for _, row in tqdm(co_occurrence.iterrows(), total = len(co_occurrence)):
    person = row['issue_a']
    response_person = row['issue_b']
    weight = row['weight']

    # Węzły
    G.add_node(person, type='issue_a', color='#e41a1c', size=10)
    G.add_node(response_person, type='issue_b', color='#377eb8', size=7)

    # Krawędź z wagą
    G.add_edge(person, response_person, weight=weight)


# Dodajemy stopnie jako metrykę (do rozmiarów i etykiet)
degree_dict = dict(G.degree())
nx.set_node_attributes(G, degree_dict, name='degree')

# ✅ Interaktywny podgląd w Jupyterze (opcjonalnie)
Sigma(
    G,
    node_color="tag",
    node_label_size="degree",
    node_size="degree"
)

# ✅ Eksport do HTML
Sigma.write_html(
    G,
    'data/article_lpg_4.html',
    fullscreen=True,
    node_metrics=['louvain'],
    node_color='louvain',
    node_size='degree',
    node_size_range=(3, 20),
    max_categorical_colors=30,
    default_edge_type='curve',
    default_node_label_size=14,
    node_border_color_from='node'
)

discussed_issue = URIRef("https://example.org/jesuit_calvinist/discussedIssue")

polemical_issues = []
for text in out.subjects(RDF.type, SCHEMA.Text):
    for issue in out.objects(text, discussed_issue):
                # print(str(person))
        polemical_issues.append(str(issue))

print(len(polemical_issues), "Persons are cited as keyAuthorsCited")

from collections import Counter
polemical_issues_counted = dict(Counter(polemical_issues))

#%% 5. statystyka dla języków i gatunków

from rdflib import Graph, Namespace, RDF
from collections import Counter
import json
from datetime import datetime

SCHEMA = Namespace("http://schema.org/")
BF = Namespace("http://id.loc.gov/ontologies/bibframe/")

# === Wyciągnięcie obiektów schema:Text ===
texts = set(out.subjects(RDF.type, SCHEMA.Text))

# === Pomocnicze funkcje ===
def extract_year(value):
    """Wyciąga rok z literału RDF."""
    s = str(value)
    try:
        return int(s[:4])
    except ValueError:
        try:
            return datetime.strptime(s, "%Y-%m-%d").year
        except Exception:
            return None

# === Przedziały czasowe ===
periods = [
    (1574, 1583),
    (1584, 1593),
    (1594, 1603),
    (1604, 1613),
    (1614, 1623),
    (1624, 1633),
    (1634, 1647),
]
labels = [f"{a}–{b}" for a, b in periods]

# === Przygotowanie struktur wynikowych ===
result = {label: {"languages": {}, "genres": {}} for label in labels}

# === Główna pętla ===
for t in texts:
    # rok publikacji
    years = [extract_year(o) for o in out.objects(t, SCHEMA.datePublished)]
    years = [y for y in years if y]
    if not years:
        continue
    year = years[0]

    # znajdź odpowiedni przedział
    label = None
    for (start, end), lbl in zip(periods, labels):
        if start <= year <= end:
            label = lbl
            break
    if label is None:
        continue

    # języki
    for lang in out.objects(t, SCHEMA.inLanguage):
        result[label]["languages"][str(lang)] = result[label]["languages"].get(str(lang), 0) + 1

    # gatunki (z mapowaniem na schema:name)
    for genre_uri in out.objects(t, SCHEMA.genre):
        names = [str(n) for n in out.objects(genre_uri, SCHEMA.name)]
        if not names and (genre_uri, RDF.type, BF.GenreForm) in out:
            continue
        for name in names:
            result[label]["genres"][name] = result[label]["genres"].get(name, 0) + 1


sum(result.get('1574–1583').get('genres').values())

result.keys()










