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




































