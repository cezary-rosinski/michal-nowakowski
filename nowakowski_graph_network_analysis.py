import rdflib
import networkx as nx
from ipysigma import Sigma
from rdflib import Graph, Namespace, Literal
import pandas as pd
from tqdm import tqdm
import plotly.graph_objects as go
import plotly.io as pio
import matplotlib.pyplot as plt
from collections import Counter
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import matplotlib.colors as mcolors

#%%
# 1. Wczytanie pliku TTL za pomocą rdflib
g = rdflib.Graph()
ttl_file = 'jecal.ttl'
g.parse(ttl_file, format='ttl')

#%% przygotowanie

uris = set()

ps = set()
for s, p, o in g:
    ps.add(str(p))
    # print(s, p, o)
    
    if isinstance(s, rdflib.term.URIRef):
        uris.add(str(s))
    if isinstance(p, rdflib.term.URIRef):
        uris.add(str(p))
    if isinstance(o, rdflib.term.URIRef):
        uris.add(str(o))

# 3. Posortowana lista i wypisanie
# for uri in sorted(uris):
#     print(uri)

SCHEMA = Namespace("http://schema.org/")

# Tworzenie słownika: {osoba: name}
people_names = {}

for subject, name in g.subject_objects(predicate=SCHEMA.name):
    people_names[subject] = str(name)

#
# Definicja przestrzeni nazw
JC = Namespace("https://example.org/jesuit_calvinist/")

# Filtracja zasobów, które mają editionType = "Original"
original_texts = []

for subject in g.subjects(predicate=JC.editionType, object=Literal("Original")):
    original_texts.append(subject)

# Wyświetlenie wyników
# print("Teksty z editionType = 'Original':")
# for text in original_texts:
#     print(text)

#%%
#(1) Text, Cited Author oraz Theme; 
# cited authors + Polemical Themes (Random Order)
# https://example.org/jesuit_calvinist/keyAuthorsCited
# https://example.org/jesuit_calvinist/polemicalTheme

text_author = dict()
text_theme = dict()
for s, p, o in g:
    if s in original_texts:
        if str(p) == 'https://example.org/jesuit_calvinist/keyAuthorsCited':
            if str(s) not in text_author:
                text_author[str(s)] = [people_names.get(o)]
            else: text_author[str(s)].append(people_names.get(o))
        elif str(p) == 'https://example.org/jesuit_calvinist/polemicalTheme':
            if str(s) not in text_theme:
                text_theme[str(s)] = [str(o)]
            else: text_theme[str(s)].append(str(o))

result_27a = []
for k, v in text_author.items():
    for ka, va in text_theme.items():
        if k == ka:
            for a in v:
                for t in va:
                    result_27a.append([a, t])
                    
df = pd.DataFrame(result_27a, columns= ['person', 'topic'])
co_occurrence = df.groupby(['person', 'topic']).size().reset_index(name='weight')
co_occurrence.to_excel('data/27a.xlsx', index=False)

# Tworzymy graf
G = nx.Graph()

# Dodajemy węzły i krawędzie z wagami
for _, row in tqdm(co_occurrence.iterrows(), total = len(co_occurrence)):
    person = row['person']
    topic = row['topic']
    weight = row['weight']

    # Węzły
    G.add_node(person, type='person', color='#e41a1c', size=10)
    G.add_node(topic, type='topic', color='#377eb8', size=7)

    # Krawędź z wagą
    G.add_edge(person, topic, weight=weight)


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
    'data/27a1.html',
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

#LUB

pagerank = nx.pagerank(G)

nx.set_node_attributes(G, pagerank, name='pagerank')

# ✅ Eksport do HTML z własnymi metrykami
Sigma.write_html(
    G,
    'data/27a2.html',
    fullscreen=True,
    node_color='pagerank',         # ręcznie przypisany
    node_size='degree',
    node_size_range=(3, 20),
    max_categorical_colors=30,
    default_edge_type='curve',
    default_node_label_size=14,
    node_border_color_from='node'
)

#%%

#(2) Text, Cited Author oraz Discussed Issue;  
# cited authors + Discussed Issues (Random Order)
# https://example.org/jesuit_calvinist/keyAuthorsCited
# https://example.org/jesuit_calvinist/discussedIssue

text_author = dict()
text_issue = dict()
for s, p, o in g:
    if s in original_texts:
        if str(p) == 'https://example.org/jesuit_calvinist/keyAuthorsCited':
            if str(s) not in text_author:
                text_author[str(s)] = [people_names.get(o)]
            else: text_author[str(s)].append(people_names.get(o))
        elif str(p) == 'https://example.org/jesuit_calvinist/discussedIssue':
            if str(s) not in text_issue:
                text_issue[str(s)] = [str(o)]
            else: text_issue[str(s)].append(str(o))

result_27b = []
for k, v in text_author.items():
    for ka, va in text_issue.items():
        if k == ka:
            for a in v:
                for t in va:
                    result_27b.append([a, t])
                    
df = pd.DataFrame(result_27b, columns= ['person', 'topic'])
co_occurrence = df.groupby(['person', 'topic']).size().reset_index(name='weight')
co_occurrence.to_excel('data/27b.xlsx', index=False)

# Tworzymy graf
G = nx.Graph()

# Dodajemy węzły i krawędzie z wagami
for _, row in tqdm(co_occurrence.iterrows(), total = len(co_occurrence)):
    person = row['person']
    topic = row['topic']
    weight = row['weight']

    # Węzły
    G.add_node(person, type='person', color='#e41a1c', size=10)
    G.add_node(topic, type='topic', color='#377eb8', size=7)

    # Krawędź z wagą
    G.add_edge(person, topic, weight=weight)


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
    'data/27b1.html',
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

#LUB

pagerank = nx.pagerank(G)

nx.set_node_attributes(G, pagerank, name='pagerank')

# ✅ Eksport do HTML z własnymi metrykami
Sigma.write_html(
    G,
    'data/27b2.html',
    fullscreen=True,
    node_color='pagerank',         # ręcznie przypisany
    node_size='degree',
    node_size_range=(3, 20),
    max_categorical_colors=30,
    default_edge_type='curve',
    default_node_label_size=14,
    node_border_color_from='node'
)

#%%
#(3) Text, Historical Figure Mentioned oraz Theme; 
# key historical figures + Polemical Themes (Random Order)
# https://example.org/jesuit_calvinist/keyHistoricalFiguresMentioned
# https://example.org/jesuit_calvinist/polemicalTheme

text_figure = dict()
text_theme = dict()
for s, p, o in g:
    if s in original_texts:
        if str(p) == 'https://example.org/jesuit_calvinist/keyHistoricalFiguresMentioned':
            if str(s) not in text_figure:
                text_figure[str(s)] = [people_names.get(o)]
            else: text_figure[str(s)].append(people_names.get(o))
        elif str(p) == 'https://example.org/jesuit_calvinist/polemicalTheme':
            if str(s) not in text_theme:
                text_theme[str(s)] = [str(o)]
            else: text_theme[str(s)].append(str(o))

result_27c = []
for k, v in text_figure.items():
    for ka, va in text_theme.items():
        if k == ka:
            for a in v:
                for t in va:
                    result_27c.append([a, t])
                    
df = pd.DataFrame(result_27c, columns= ['person', 'topic'])
co_occurrence = df.groupby(['person', 'topic']).size().reset_index(name='weight')
co_occurrence.to_excel('data/27c.xlsx', index=False)

# Tworzymy graf
G = nx.Graph()

# Dodajemy węzły i krawędzie z wagami
for _, row in tqdm(co_occurrence.iterrows(), total = len(co_occurrence)):
    person = row['person']
    topic = row['topic']
    weight = row['weight']

    # Węzły
    G.add_node(person, type='person', color='#e41a1c', size=10)
    G.add_node(topic, type='topic', color='#377eb8', size=7)

    # Krawędź z wagą
    G.add_edge(person, topic, weight=weight)


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
    'data/27c1.html',
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

#LUB

pagerank = nx.pagerank(G)

nx.set_node_attributes(G, pagerank, name='pagerank')

# ✅ Eksport do HTML z własnymi metrykami
Sigma.write_html(
    G,
    'data/27c2.html',
    fullscreen=True,
    node_color='pagerank',         # ręcznie przypisany
    node_size='degree',
    node_size_range=(3, 20),
    max_categorical_colors=30,
    default_edge_type='curve',
    default_node_label_size=14,
    node_border_color_from='node'
)

#%%
#(4) Text, Historical Figure Mentioned oraz Discussed Issue
#key historical figures + Discussed Issue
# https://example.org/jesuit_calvinist/keyHistoricalFiguresMentioned
# https://example.org/jesuit_calvinist/discussedIssue

text_figure = dict()
text_issue = dict()
for s, p, o in g:
    if s in original_texts:
        if str(p) == 'https://example.org/jesuit_calvinist/keyHistoricalFiguresMentioned':
            if str(s) not in text_figure:
                text_figure[str(s)] = [people_names.get(o)]
            else: text_figure[str(s)].append(people_names.get(o))
        elif str(p) == 'https://example.org/jesuit_calvinist/discussedIssue':
            if str(s) not in text_issue:
                text_issue[str(s)] = [str(o)]
            else: text_issue[str(s)].append(str(o))

result_27d = []
for k, v in text_figure.items():
    for ka, va in text_issue.items():
        if k == ka:
            for a in v:
                for t in va:
                    result_27d.append([a, t])
                    
df = pd.DataFrame(result_27d, columns= ['person', 'topic'])
co_occurrence = df.groupby(['person', 'topic']).size().reset_index(name='weight')
co_occurrence.to_excel('data/27d.xlsx', index=False)

# Tworzymy graf
G = nx.Graph()

# Dodajemy węzły i krawędzie z wagami
for _, row in tqdm(co_occurrence.iterrows(), total = len(co_occurrence)):
    person = row['person']
    topic = row['topic']
    weight = row['weight']

    # Węzły
    G.add_node(person, type='person', color='#e41a1c', size=10)
    G.add_node(topic, type='topic', color='#377eb8', size=7)

    # Krawędź z wagą
    G.add_edge(person, topic, weight=weight)


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
    'data/27d1.html',
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

#LUB

pagerank = nx.pagerank(G)

nx.set_node_attributes(G, pagerank, name='pagerank')

# ✅ Eksport do HTML z własnymi metrykami
Sigma.write_html(
    G,
    'data/27d2.html',
    fullscreen=True,
    node_color='pagerank',         # ręcznie przypisany
    node_size='degree',
    node_size_range=(3, 20),
    max_categorical_colors=30,
    default_edge_type='curve',
    default_node_label_size=14,
    node_border_color_from='node'
)

#%% 29
#1 authors and Response to Authors -- graf kierunkowy
# https://example.org/jesuit_calvinist/keyAuthorsCited
# https://example.org/jesuit_calvinist/responseToAuthor

text_author = dict()
text_response = dict()
# for s, p, o in g:
for s, p, o in out:
    if s in original_texts:
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
co_occurrence.to_excel('data/29a.xlsx', index=False)

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
    'data/29a1.html',
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

#LUB

pagerank = nx.pagerank(G)

nx.set_node_attributes(G, pagerank, name='pagerank')

# ✅ Eksport do HTML z własnymi metrykami
Sigma.write_html(
    G,
    'data/29a2.html',
    fullscreen=True,
    node_color='pagerank',         # ręcznie przypisany
    node_size='degree',
    node_size_range=(3, 20),
    max_categorical_colors=30,
    default_edge_type='curve',
    default_node_label_size=14,
    node_border_color_from='node'
)

nx.write_graphml(G, "data/29a.graphml")
#%% 29a2 + confessional type
JC = Namespace("https://example.org/jesuit_calvinist/")

text_people = {}
for t, p in g.subject_objects(predicate=SCHEMA.author):
    text_people[t] = p

text_confession = {}
for subject, conf in g.subject_objects(predicate=JC.confessionalProfile):
    text_confession[subject] = str(conf)
    
people_confession = {}
for t, p in text_people.items():
    for te, c in text_confession.items():
        if t == te:
            people_confession[people_names.get(p)] = c
people_confession = {k:v for k,v in people_confession.items() if k != 'Anonymous'}

people_confession.update({'Ames, William': 'Reformed Evangelical',
                          'Drohojewski, Jan': 'Reformed Evangelical',
                          'Bellarmine, Robert': 'Roman Catholic',
                          'Myślenta, Celestyn': 'Lutheran',
                          'Wądłowski, Łukasz': 'Reformed Evangelical',
                          'Reszka, Stanisław': 'Roman Catholic',
                          'Chemnitz, Martin': 'Lutheran',
                          'Chytraeus, David': 'Lutheran'})

text_author = dict()
text_response = dict()
for s, p, o in g:
    if s in original_texts:
        if str(p) == str(SCHEMA.author):
            if str(s) not in text_author:
                text_author[str(s)] = [people_names.get(o)]
            else: text_author[str(s)].append(people_names.get(o))
        elif str(p) == 'https://example.org/jesuit_calvinist/responseToAuthor':
            if str(s) not in text_response:
                text_response[str(s)] = [people_names.get(o)]
            else: text_response[str(s)].append(people_names.get(o))

result_29a2 = []
for k, v in text_author.items():
    for ka, va in text_response.items():
        if k == ka:
            for a in v:
                for t in va:
                    result_29a2.append([a, t])

df = pd.DataFrame(result_29a2, columns= ['person', 'response_person'])
df = df.loc[~df['person'].isin(['Anonymous', 'Various']) &
            ~df['response_person'].isin(['Anonymous', 'Various'])]
df['person_confession'] = df['person'].apply(lambda x: people_confession.get(x))
df['response_person_confession'] = df['response_person'].apply(lambda x: people_confession.get(x))

co_occurrence = df.groupby(['person', 'person_confession', 'response_person', 'response_person_confession']).size().reset_index(name='weight')
co_occurrence.to_excel('data/29a2.xlsx', index=False)

# Tworzymy graf
G = nx.Graph()

# Dodajemy węzły i krawędzie z wagami oraz wyznaniami
for _, row in tqdm(co_occurrence.iterrows(), total=len(co_occurrence)):
    person = row['person']
    person_confession = row['person_confession']
    response_person = row['response_person']
    response_confession = row['response_person_confession']
    weight = row['weight']

    # Węzły z atrybutami
    G.add_node(person, type='person', confession=person_confession)
    G.add_node(response_person, type='response_person', confession=response_confession)

    # Krawędź
    G.add_edge(person, response_person, weight=weight)

# Dodajemy metrykę stopnia
degree_dict = dict(G.degree())
nx.set_node_attributes(G, degree_dict, name='degree')

# (Opcjonalnie) PageRank
pagerank = nx.pagerank(G)
nx.set_node_attributes(G, pagerank, name='pagerank')


# Eksport 1 — kolor według wyznania
Sigma.write_html(
    G,
    'data/29a2_confession.html',
    fullscreen=True,
    node_color='confession',
    node_size='degree',
    node_size_range=(3, 20),
    max_categorical_colors=30,
    default_edge_type='curve',
    default_node_label_size=14,
    node_border_color_from='node'
)

# Eksport 2 — kolor według pagerank
Sigma.write_html(
    G,
    'data/29a2_pagerank.html',
    fullscreen=True,
    node_color='pagerank',
    node_size='degree',
    node_size_range=(3, 20),
    max_categorical_colors=30,
    default_edge_type='curve',
    default_node_label_size=14,
    node_border_color_from='node'
)

nx.write_graphml(G, "data/29a2.graphml")
#%% 29a3
from collections import defaultdict

# Tworzymy pusty graf
G = nx.Graph()

# 1. Dodajemy węzły i krawędzie oraz wyznania
for _, row in tqdm(co_occurrence.iterrows(), total=len(co_occurrence)):
    person = row['person']
    person_confession = row['person_confession']
    response_person = row['response_person']
    response_confession = row['response_person_confession']
    weight = row['weight']
    
    # Dodajemy węzły z atrybutem confession
    if not G.has_node(person):
        G.add_node(person, confession=person_confession)
    if not G.has_node(response_person):
        G.add_node(response_person, confession=response_confession)
    
    G.add_edge(person, response_person, weight=weight)

# 2. Grupa node'ów wg wyznania
confession_to_nodes = defaultdict(list)
for node, data in G.nodes(data=True):
    confession = data.get('confession', 'unknown')
    confession_to_nodes[confession].append(node)

# 3. Tworzymy wspólnoty: po dwie różne grupy wyznaniowe
confessions = list(confession_to_nodes.keys())
communities = []
community_id = 0

for i in range(len(confessions)):
    for j in range(i, len(confessions)):
        group_conf1 = confessions[i]
        group_conf2 = confessions[j]
        combined_nodes = confession_to_nodes[group_conf1] + confession_to_nodes[group_conf2]
        communities.append({
            'id': f'comm_{community_id}',
            'nodes': combined_nodes,
            'confessions': {group_conf1, group_conf2}
        })
        community_id += 1

# 4. Przypisujemy węzłom przynależność do wspólnot (może być wiele)
node_to_communities = defaultdict(list)
for comm in communities:
    for node in comm['nodes']:
        node_to_communities[node].append(comm['id'])

# 5. Wybieramy główną wspólnotę (pierwszą) jako primary_community
for node in G.nodes:
    comm_list = node_to_communities[node]
    G.nodes[node]['primary_community'] = comm_list[0] if comm_list else 'none'
    G.nodes[node]['all_communities'] = ', '.join(comm_list)
    G.nodes[node]['label'] = f"{node} (wspólnota: {G.nodes[node]['primary_community']})"

# 6. Dodajemy metryki
nx.set_node_attributes(G, dict(G.degree()), name='degree')
nx.set_node_attributes(G, nx.pagerank(G), name='pagerank')

# 7. Eksport do HTML – kolorujemy wg primary_community, nie confession
Sigma.write_html(
    G,
    'data/29a3_community_custom.html',
    fullscreen=True,
    node_color='primary_community',
    node_label='label',
    node_size='degree',
    node_size_range=(3, 20),
    max_categorical_colors=50,
    default_edge_type='curve',
    default_node_label_size=14,
    node_border_color_from='node'
)




#%% Sankey

# Wczytanie danych
df = pd.read_excel('data/29a.xlsx')

# Dodajemy sufiksy, by rozdzielić węzły autorów i adresatów
df['person_label'] = df['person'] + ' (autor)'
df['response_label'] = df['response_person'] + ' (adresat)'

# Lista unikalnych nazw węzłów
all_labels = pd.unique(df[['person_label', 'response_label']].values.ravel())
label_to_index = {name: i for i, name in enumerate(all_labels)}

# Grupowanie danych
links_df = df.groupby(['person_label', 'response_label']).size().reset_index(name='count')

# Mapowanie do indeksów
source_indices = links_df['person_label'].map(label_to_index)
target_indices = links_df['response_label'].map(label_to_index)
values = links_df['count']

# Tworzenie Sankey diagramu
fig = go.Figure(data=[go.Sankey(
    node=dict(
        pad=15,
        thickness=20,
        line=dict(color="black", width=0.5),
        label=list(label_to_index.keys())
    ),
    link=dict(
        source=source_indices,
        target=target_indices,
        value=values
    )
)])

fig.update_layout(title_text="Diagram Sankeya: Autorzy (lewa) i adresaci (prawa)", font_size=12)
fig.write_html("data/29a_sankey_diagram.html")

# pio.renderers.default = "browser"
# fig.show()

#%% 29a -- analiza sieciowa

# Wczytanie grafu z pliku GraphML
G = nx.read_graphml("data/29a.graphml")

# Konwersja na graf nieskierowany dla niektórych analiz (opcjonalnie)
G_undirected = G.to_undirected()

# Podstawowe informacje
print("Liczba wierzchołków:", G.number_of_nodes())
print("Liczba krawędzi:", G.number_of_edges())

# 1. Centralność stopnia (degree centrality)
degree_centrality = nx.degree_centrality(G)

# 2. Centralność pośrednictwa (betweenness centrality)
betweenness_centrality = nx.betweenness_centrality(G)

# 3. Centralność bliskości (closeness centrality)
closeness_centrality = nx.closeness_centrality(G)

# 4. Centralność wektora własnego (eigenvector centrality)
try:
    eigenvector_centrality = nx.eigenvector_centrality(G)
except nx.NetworkXError:
    eigenvector_centrality = {node: None for node in G.nodes()}
    print("Eigenvector centrality failed — graf może być niespójny.")

# 5. Pagerank
pagerank = nx.pagerank(G)

# 6. Spoistość i komponenty
components = list(nx.connected_components(G_undirected))
largest_component = max(components, key=len)
print("Liczba komponentów spójnych:", len(components))
print("Rozmiar największego komponentu:", len(largest_component))

# Globalna spoistość
print("Czy graf jest spójny?", nx.is_connected(G.to_undirected()))

# Liczba komponentów spójnych
print("Liczba komponentów:", nx.number_connected_components(G.to_undirected()))

# Średnie lokalne klastrowanie
print("Średni współczynnik klastrowania:", nx.average_clustering(G.to_undirected()))

# Gęstość grafu
print("Gęstość grafu:", nx.density(G))

# 7. Analiza klastrów (spójność lokalna)
clustering_coeffs = nx.clustering(G_undirected)

# Tworzenie tabeli wyników
df_centrality = pd.DataFrame({
    'Node': list(G.nodes()),
    'Degree Centrality': pd.Series(degree_centrality),
    'Betweenness Centrality': pd.Series(betweenness_centrality),
    'Closeness Centrality': pd.Series(closeness_centrality),
    'Eigenvector Centrality': pd.Series(eigenvector_centrality),
    'PageRank': pd.Series(pagerank),
    'Clustering Coefficient': pd.Series(clustering_coeffs)
}).sort_values(by='Degree Centrality', ascending=False)

# Wyświetlenie top 10
print("\nTop 10 węzłów według centralności stopnia:")
print(df_centrality.head(10))

# Zapis do pliku CSV
df_centrality.to_excel("data/centrality_analysis.xlsx", index=False)

# (Opcjonalnie) wizualizacja grafu z wielkością węzłów wg centralności
plt.figure(figsize=(12, 10))
pos = nx.spring_layout(G, seed=42)
node_sizes = [v * 5000 for v in degree_centrality.values()]
nx.draw(G, pos, with_labels=True, node_size=node_sizes, node_color="skyblue", edge_color="gray")
plt.title("Wizualizacja grafu z centralnością stopnia")
plt.show()

#%% 29b -- communities
import networkx as nx
import community.community_louvain as community_louvain
import matplotlib.pyplot as plt

# Load the graph from the uploaded .graphml file
graphml_path = "data/29a.graphml"
G = nx.read_graphml(graphml_path)

# Convert to undirected graph if it's directed (Louvain works on undirected graphs)
if G.is_directed():
    G = G.to_undirected()

# Apply Louvain community detection
partition = community_louvain.best_partition(G)

# Count number of communities
num_communities = len(set(partition.values()))

# Add community information to each node
nx.set_node_attributes(G, partition, 'community')

# Show number of communities
num_communities

# community_df = pd.DataFrame(list(partition.items()), columns=["node", "community"])
# community_df.sort_values(by="community", inplace=True)
# community_df.to_excel('data/29b_people_and_communities.xlsx', index=False)

community_df = pd.read_excel('data/29b_people_and_communities.xlsx')

community_values = set(partition.values())
#Key Historical Figures Mentioned, Key Authors Cited, Categorized Polemical Themes (Random Order), Discussed Issues (Random Order)

for com in tqdm(community_values):
    # com = list(community_values)[1]
    com_people = community_df.loc[community_df['community'] == com]['node'].to_list()
    texts = []
    for s, p, o in g:
        if s in original_texts and str(p) == str(SCHEMA.author) and people_names.get(o) in com_people:
            texts.append(str(s))
    key_historical_figures_mentioned = []
    key_authors_cited = []
    categorized_polemical_themes = []
    discussed_issues = []
    for s, p, o in g:
        if str(s) in texts:
            if str(p) == 'https://example.org/jesuit_calvinist/keyHistoricalFiguresMentioned':
                key_historical_figures_mentioned.append(people_names.get(o))
            elif str(p) == 'https://example.org/jesuit_calvinist/keyAuthorsCited':
                key_authors_cited.append(people_names.get(o))
            elif str(p) == 'https://example.org/jesuit_calvinist/categorizedPolemicalTheme':
                categorized_polemical_themes.append(str(o))
            elif str(p) == 'https://example.org/jesuit_calvinist/discussedIssue':
                discussed_issues.append(str(o))
    key_historical_figures_mentioned = [e for e in key_historical_figures_mentioned if e]
    key_authors_cited = [e for e in key_authors_cited if e]
    categorized_polemical_themes = [e for e in categorized_polemical_themes if e]
    discussed_issues = [e for e in discussed_issues if e]
    
    categories = {'Key Historical Figures Mentioned': key_historical_figures_mentioned,
                  'Key Authors Cited': key_authors_cited,
                  'Categorized Polemical Themes': categorized_polemical_themes,
                  'Discussed Issues': discussed_issues}
    
    for k,v in categories.items():
        # k = list(categories.keys())[0]
        # v = categories.get(k)
        counter = Counter(v)
        
        # Sortowanie od najczęstszych do najrzadszych i ograniczenie do top 20
        top_items = counter.most_common(20)
        unique_entry = [item[0] for item in top_items]
        entry_counter = [item[1] for item in top_items]
        
        # Styl wykresu
        plt.style.use("seaborn-v0_8-whitegrid")
        
        # Ustawienia kolorów: od ciemniejszego do jaśniejszego
        norm = mcolors.Normalize(vmin=min(entry_counter), vmax=max(entry_counter))
        colors = cm.viridis(norm(entry_counter))
        
        # Tworzenie wykresu
        plt.figure(figsize=(12, 7))
        bars = plt.bar(unique_entry, entry_counter, color=colors, edgecolor='black')
        
        # Etykiety nad słupkami
        for bar in bars:
            height = bar.get_height()
            plt.text(bar.get_x() + bar.get_width()/2, height + 0.5, str(height),
                     ha='center', va='bottom', fontsize=9)
        
        # Opisy i styl osi
        plt.title(f"Top 20 most frequent '{k}' in community {com}", fontsize=14, fontweight='bold')
        plt.xlabel("Themes", fontsize=12)
        plt.ylabel("Frequency", fontsize=12)
        plt.xticks(rotation=90, fontsize=10)
        plt.yticks(fontsize=10)
        plt.grid(axis='y', linestyle='--', alpha=0.4)
        
        # Dopasowanie layoutu
        plt.tight_layout()
        
        # Wyświetlenie wykresu
        # plt.show()
        plt.savefig(f"data/29b_community_{com}_top_20_{k}.png", dpi=300, bbox_inches="tight")
                


#%% rdf to lpg dla całego grafu

uris = set()
for s, p, o in g:
    if isinstance(s, rdflib.term.URIRef):
        uris.add(str(s))
    if isinstance(p, rdflib.term.URIRef):
        uris.add(str(p))
    if isinstance(o, rdflib.term.URIRef):
        uris.add(str(o))

# 3. Posortowana lista i wypisanie
for uri in sorted(uris):
    print(uri)



# 2. Utworzenie grafu skierowanego w networkx
G = nx.DiGraph()

# 3. Dodawanie węzłów i krawędzi na podstawie potrójek RDF
triples = []
for s, p, o in g:
    s_str = str(s)
    p_str = str(p)
    o_str = str(o)
    G.add_node(s_str)
    G.add_node(o_str)
    G.add_edge(s_str, o_str, predicate=p_str)
    triples.append((s_str, p_str, o_str))

# 4. Podstawowe statystyki grafu
print(f"Liczba węzłów: {G.number_of_nodes()}")
print(f"Liczba krawędzi: {G.number_of_edges()}")

# Opcjonalnie: wyświetlenie przykładowych krawędzi
print("\nPrzykładowe krawędzie (z atrybutem 'predicate'):")
for u, v, data in list(G.edges(data=True))[:10]:
    print(f"{u} -[{data['predicate']}]-> {v}")


Sigma(G, 
      node_color="tag",
      node_label_size=G.degree,
      node_size=G.degree
     )

Sigma.write_html(
    G,
    'data/jecal.html',
    fullscreen=True,
    node_metrics=['louvain'],
    node_color='louvain',
    node_size_range=(3, 20),
    max_categorical_colors=30,
    default_edge_type='curve',
    node_border_color_from='node',
    default_node_label_size=14,
    node_size=G.degree
)

#%% RDF filtering for the article

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Skrypt RDF do filtrowania schema:Text z editionType zawierającym "Original".
Gotowy do uruchomienia w Spyderze – wystarczy nacisnąć F5.
"""

from rdflib import Graph, Namespace, RDF, RDFS, URIRef, Literal
from pathlib import Path

# ------------------------------------------------------------
# 🔧 USTAWIENIA ŚCIEŻEK – ZMODYFIKUJ WG SWOJEGO KATALOGU
# ------------------------------------------------------------
INPUT_PATH = Path(r"jecal.ttl")
OUT_SUBGRAPH_PATH = Path(r"jecal_original_texts_subgraph.ttl")
OUT_CLASSES_PATH = Path(r"jecal_original_texts_classes.ttl")
INPUT_FORMAT = "turtle"  # zmień na np. "xml" jeśli plik RDF jest w RDF/XML

# ------------------------------------------------------------
# 🔧 NAZWY PRZESTRZENI
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

# ------------------------------------------------------------
# 🚀 GŁÓWNA LOGIKA
# ------------------------------------------------------------
def main():
    print("📘 Wczytywanie pliku RDF...")
    g = Graph()
    g.parse(INPUT_PATH, format=INPUT_FORMAT)
    print(f"✅ Załadowano {len(g)} trójek RDF z pliku {INPUT_PATH}")

    # 1️⃣ Znajdź wszystkie schema:Text, które mają editionType zawierające 'Original'
    texts = set(
        s for s in g.subjects(RDF.type, SCHEMA.Text)
        if any(literal_contains_original(o) for o in g.objects(s, JC.editionType))
    )
    print(f"🔎 Znaleziono {len(texts)} obiektów schema:Text z editionType='Original'")

    # 2️⃣ Zbuduj subgraf (ABox)
    subg = Graph()
    for prefix, ns in g.namespaces():
        subg.bind(prefix, ns)
    nodes = set()
    for s in texts:
        for p, o in g.predicate_objects(s):
            subg.add((s, p, o))
            nodes.add(s); nodes.add(o)
    for n in list(nodes):
        for c in g.objects(n, RDF.type):
            subg.add((n, RDF.type, c))
    subg.serialize(destination=OUT_SUBGRAPH_PATH, format="turtle")
    print(f"💾 Zapisano subgraf (ABox) → {OUT_SUBGRAPH_PATH} ({len(subg)} trójek)")

    # 3️⃣ Zbuduj graf klas (TBox-like)
    class_graph = Graph()
    for prefix, ns in g.namespaces():
        class_graph.bind(prefix, ns)
    observed_edges = set()
    for s, p, o in subg:
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
























