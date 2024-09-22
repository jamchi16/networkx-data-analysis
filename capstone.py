import networkx as nx
import requests
import pickle
import scipy.sparse as sp
import pandas as pd
import numpy as np
from time import time
import matplotlib.pyplot as plt
import nxviz as nv
import random
from faker import Faker

fake = Faker()


def generate_graph():
    pokeGraph = nx.Graph()
    pokemon_response = requests.get('https://pokeapi.co/api/v2/pokemon?limit=100000&offset=0').json()
    pokemon_list = [poke['name'] for poke in pokemon_response['results']]

    trainers = []
    trainer_names = []
    while len(trainers) < 500:
        trainer_name = fake.name()
        trainer_age = random.randint(10, 70)
        trainers.append((trainer_name, {'age': trainer_age}))
        trainer_names.append(trainer_name)

    edges = []
    for trainer_name, data in trainers:
        max_balls_thrown = 5
        max_pokemon_caught = 100

        if data['age'] > 30 and data['age'] < 50:
            max_balls_thrown = 10
            max_pokemon_caught = 70
        elif data['age'] > 50:
            max_balls_thrown = 15
            max_pokemon_caught = 30

        num_pokemon_caught = random.randint(1, max_pokemon_caught)
        count = 0
        while count < num_pokemon_caught:
            pokemon = random.choice(pokemon_list)
            pokeballs_thrown = random.randint(1, max_balls_thrown)
            edges.append((pokemon, trainer_name, {'pokeballs_thrown': pokeballs_thrown}))
            count += 1

    pokeGraph.add_nodes_from(pokemon_list, bipartite="pokemon")
    pokeGraph.add_nodes_from(trainers, bipartite="trainers")
    pokeGraph.add_edges_from(edges)

    with open("caught_pokemon_graph.pkl", "wb") as f:
        pickle.dump(pokeGraph, f, pickle.HIGHEST_PROTOCOL)


with open("caught_pokemon_graph.pkl", "rb") as f:
    loaded_graph = pickle.load(f)

with open("small_graph.pkl", "rb") as f:
    small_graph = pickle.load(f)


def extract_partition_nodes(G: nx.Graph, partition: str):
    nodeset = [n for n, d in G.nodes(data=True) if d["bipartite"] == partition]
    return nodeset


# Function to perform basic queries and print out useful information about the graph
def basic_queries():
    # Extract trainer nodes and print information about them
    trainer_nodes = [(n, d) for n, d in loaded_graph.nodes(data=True) if d["bipartite"] == 'trainers']
    print("Sample of node data:", list(loaded_graph.nodes(data=True))[:2])
    print("Sample of trainer nodes:", trainer_nodes[:2])

    # Print sample of edges with data
    print("Sample of edges with data:", list(loaded_graph.edges(data=True))[:2])

    # Print sample of neighbors of a specific trainer
    print("Sample of neighbors of 'John Davis':", list(loaded_graph.neighbors('Steven Smith'))[:2])

    print("\n" + "-" * 50 + "\n")

    # Print trainers ranked by the number of Pokémon they've caught
    caught_ranking = pd.Series({n: len(list(loaded_graph.neighbors(n))) for n in loaded_graph.nodes()}).sort_values(
        ascending=False)
    print("Nodes ranked by number of neighbors:\n", caught_ranking)

    print("\n" + "-" * 50 + "\n")

    # Print trainers ranked by degree centrality
    print("Nodes ranked by degree centrality:\n",
          pd.Series(nx.degree_centrality(loaded_graph)).sort_values(ascending=False))


# Function to check if a path exists between two nodes
def path_exists(node1, node2):
    visited_nodes = set()
    queue = [node1]

    while len(queue) > 0:
        node = queue.pop()
        neighbors = list(loaded_graph.neighbors(node))
        if node2 in neighbors:
            return True
        else:
            visited_nodes.add(node)
            nbrs = [n for n in neighbors if n not in visited_nodes]
            queue += nbrs

    return False


# Function to print the most successful trainers based on the number of Pokémon they've caught
def successful_trainers():
    trainer_nodes = [node for node, data in loaded_graph.nodes(data=True) if data.get('bipartite') == 'trainers']
    caught_ranking = {trainer: len(list(loaded_graph.neighbors(trainer))) for trainer in trainer_nodes}
    print("Top trainers by number of Pokémon caught:")
    print(pd.Series(caught_ranking).sort_values(ascending=False))


# Function to compute the average degree centrality by age group of trainers
def average_degree_centrality_by_age_group():
    trainer_nodes = [node for node, data in loaded_graph.nodes(data=True) if data.get('bipartite') == 'trainers']
    age_group_degree = {'10-20': [], '21-30': [], '31-40': [], '41-50': [], '51-60': [], '61-70': []}
    degree_centrality = nx.degree_centrality(loaded_graph)

    for trainer in trainer_nodes:
        age = loaded_graph.nodes[trainer]['age']
        age_group = None

        if 10 <= age <= 20:
            age_group = '10-20'
        elif 21 <= age <= 30:
            age_group = '21-30'
        elif 31 <= age <= 40:
            age_group = '31-40'
        elif 41 <= age <= 50:
            age_group = '41-50'
        elif 51 <= age <= 60:
            age_group = '51-60'
        elif 61 <= age <= 70:
            age_group = '61-70'

        if age_group:
            age_group_degree[age_group].append(degree_centrality[trainer])

    average_degree_centrality = {}
    for age_group, centrality_values in age_group_degree.items():
        if centrality_values:
            average_degree_centrality[age_group] = sum(centrality_values) / len(centrality_values)
        else:
            average_degree_centrality[age_group] = 0

    print("Average degree centrality by age group:")
    print(pd.Series(average_degree_centrality).sort_values(ascending=False))


def hardest_pokemon_to_catch():
    avg_pokemon_caught_attempts = {}
    avg_trainer_throw_attempts = {}
    for pokemon, trainer, data in  list(loaded_graph.edges(data=True)):
        if pokemon not in avg_pokemon_caught_attempts.keys():
            avg_pokemon_caught_attempts[pokemon] = []

        if trainer not in avg_trainer_throw_attempts.keys():
            avg_trainer_throw_attempts[trainer] = []

        avg_pokemon_caught_attempts[pokemon].append(int(data['pokeballs_thrown']))
        avg_trainer_throw_attempts[trainer].append(int(data['pokeballs_thrown']))

    for pokemon in avg_pokemon_caught_attempts.keys():
        avg_pokemon_caught_attempts[pokemon] = sum(avg_pokemon_caught_attempts[pokemon])/len(avg_pokemon_caught_attempts[pokemon])

    new_avg_trainer_throw_attempts = {}
    for trainer in avg_trainer_throw_attempts:
        avg = sum(avg_trainer_throw_attempts[trainer]) / len(avg_trainer_throw_attempts[trainer])
        new_avg_trainer_throw_attempts[f'{trainer} Age: {loaded_graph.nodes(data=True)[trainer]["age"]}'] = avg

    print('Hardest Pokemon to Catch\n')
    print(pd.Series(avg_pokemon_caught_attempts).sort_values(ascending=False))
    print("\n" + "-" * 50 + "\n")
    print('Trainers that threw the most Pokeballs on average\n')
    print(pd.Series(new_avg_trainer_throw_attempts).sort_values(ascending=False))




# Function to perform similarity analysis between trainers
def similarity_analysis():
    start_time = time()
    trainer_nodes = [n for n, d in loaded_graph.nodes(data=True) if d["bipartite"] == 'trainers']

    projected_trainers = nx.bipartite.weighted_projected_graph(loaded_graph, trainer_nodes)
    most_similar_trainers = sorted(projected_trainers.edges(data=True), key=lambda x: x[2]['weight'], reverse=True)[0]

    end_time = time()
    print("Most similar trainers (using weighted graph projections):", most_similar_trainers)
    print(f"Elapsed time for Graph Projection Method: {end_time - start_time:.3f} seconds")

    print("\n" + "-" * 50 + "\n")

    start_time = time()
    mat = nx.bipartite.matrix.biadjacency_matrix(loaded_graph, trainer_nodes)
    train_mat = mat @ mat.T

    degrees = train_mat.diagonal()
    trainer_diags = sp.diags(degrees)
    off_diagonals = train_mat - trainer_diags
    c1, c2 = np.unravel_index(np.argmax(off_diagonals), train_mat.shape)

    end_time = time()
    print("Most similar trainers (using matrix operations):", trainer_nodes[c1], ",", trainer_nodes[c2])
    print(f"Elapsed time for Matrix Method: {end_time - start_time:.3f} seconds")


# Function to perform community detection using Louvain method
def community_detection_louvain():
    print("Communities detected using Louvain method:")
    print(nx.community.louvain_communities(loaded_graph, seed=None)[0])



basic_queries()
print("\n" + "-"*50 + "\n")
print("Path exists between 'John Davis' and 'William Smith':", path_exists('Kelly Wilson', 'Robert Hall'))
print("\n" + "-"*50 + "\n")
successful_trainers()
print("\n" + "-"*50 + "\n")
average_degree_centrality_by_age_group()
print("\n" + "-"*50 + "\n")
hardest_pokemon_to_catch()
print("\n" + "-"*50 + "\n")
similarity_analysis()
print("\n" + "-"*50 + "\n")
community_detection_louvain()



def graph_visualizations(type):
    match type:
        case "hairball":
            nx.draw(small_graph, with_labels=True)
        case "matrix":
            nv.matrix(
                small_graph,
                group_by="bipartite",
                node_color_by="bipartite"
            )
        case "arc":
            nv.arc(
                small_graph,
                group_by="bipartite",
                node_color_by="bipartite"
            )
        case "circos":
            nv.circos(
                small_graph,
                group_by="bipartite",
                node_color_by="bipartite"
            )
        case "hive":
            nv.hive(
                small_graph,
                group_by="bipartite",
                node_color_by="bipartite"
            )
        case _:
            return

    plt.show()
# graph_visualizations('matrix')

# generate_graph()

# print(nx.shortest_path(loaded_graph, source='Kelly Wilson', target='Robert Hall'))