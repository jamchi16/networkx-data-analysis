import json
import re
import uuid
from tqdm import tqdm
import networkx as nx
import requests
import pickle
import scipy
import scipy.sparse as sp
from networkx.algorithms import bipartite
import pandas as pd
import numpy as np
from time import time
import matplotlib.pyplot as plt
import nxviz as nv
from nxviz import annotate, plots, highlights



def extract_data(input_string):
    winning_player = ''
    team_1_data = {
        'id': str(uuid.uuid4()),
        'won_battle': False,
        'pokemon': []
    }

    team_2_data = {
        'id': str(uuid.uuid4()),
        'won_battle': False,
        'pokemon': []
    }

    pokemon_pattern = r"\|poke\|(p1|p2)\|([a-zA-Z\s-]+),"
    user_pattern = r"\|player\|(p1|p2)\|([\w\s]+)\|"
    win_pattern = r"\|win\|([\w\s]+?)\n"

    # Find all matches in the input string
    pokemon_matches = re.findall(pokemon_pattern, input_string)
    user_matches = re.findall(user_pattern, input_string)
    winner_match = re.findall(win_pattern, input_string)
    if winner_match:
        winner_match = winner_match[0]

    team_1_data['pokemon'] = [pokemon for player, pokemon in pokemon_matches if player == 'p1']
    team_2_data['pokemon'] = [pokemon for player, pokemon in pokemon_matches if player == 'p2']
    for player, name in user_matches:
        if name == winner_match:
            winning_player = player

    if winning_player == 'p1':
        team_1_data['won_battle'] = True
    elif winning_player == 'p2':
        team_2_data['won_battle'] = True

    return [team_1_data, team_2_data]


def retrieve_high_rank_ids():
    num = 1
    ids = []
    pbar = tqdm(total=1000, desc="Fetching IDs")
    while len(ids) < 1000:
        url = f'https://replay.pokemonshowdown.com/search.json?format=gen9vgc2024reggbo3&page={num}'
        response = requests.get(url).json()
        for battle in response:
            if battle['rating'] and battle['rating'] > 1000:
                ids.append(battle['id'])
                pbar.update(1)
        num += 1
    pbar.close()
    with open('high_rank_ids.json', 'w') as f:
        json.dump(ids, f)


def write_export():
    battle_data_export = []
    with open('high_rank_ids.json', 'r') as f:
        ids = json.load(f)

    pbar = tqdm(total=len(ids), desc="Fetching Battle Data")
    for battle_id in ids:
        url = f'https://replay.pokemonshowdown.com/{battle_id}.json'
        response = requests.get(url).json()
        battle_data_export += extract_data(response['log'])
        pbar.update(1)
    pbar.close()
    with open('battle_data.json', 'w') as f:
        json.dump(battle_data_export, f)


def generate_networkx_graph():
    with open('battle_data.json', 'r') as f:
        battle_data = json.load(f)

    bG = nx.Graph()
    battles = []
    pokemon = []
    edges = []
    for battle in battle_data:
        battles.append((battle['id'], {'id': battle['id'], 'won_battle': battle['won_battle']}))

        pokemon += battle['pokemon']

        for pokemon_in_battle in battle['pokemon']:
            edges.append((battle['id'], pokemon_in_battle))

    bG.add_nodes_from(battles, bipartite="battles")
    bG.add_nodes_from(list(set(pokemon)), bipartite="pokemon")
    bG.add_edges_from(set(edges))

    with open("battle_graph.pkl", "wb") as f:
        pickle.dump(bG, f, pickle.HIGHEST_PROTOCOL)


def extract_partition_nodes(G: nx.Graph, partition: str):
    nodeset = [n for n, d in G.nodes(data=True) if d["bipartite"] == partition]
    if len(nodeset) == 0:
        raise Exception(f"No nodes exist in the partition {partition}!")
    return nodeset


def analyze_graph():
    pass

if __name__ == '__main__':
    with open("battle_graph.pkl", "rb") as f:
        G_loaded = pickle.load(f)

    battle_nodes = extract_partition_nodes(G_loaded, "battles")
    pokemon_nodes = extract_partition_nodes(G_loaded, "pokemon")
    mat = nx.bipartite.biadjacency_matrix(G_loaded, row_order=battle_nodes)
    customer_mat = mat @ mat.T
    degrees = customer_mat.diagonal()
    cust_idx = np.argmax(degrees)

    deg = (
        pd.Series(dict(nx.degree(G_loaded, battle_nodes)))
        .to_frame()
        .reset_index()
        .sort_values(0, ascending=False)
    )
    deg.head()

    customer_diags = sp.diags(degrees)
    off_diagonals = customer_mat - customer_diags
    most_similar = np.unravel_index(np.argmax(off_diagonals), customer_mat.shape)