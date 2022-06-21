import pickle
import torch
import pandas as pd


class TestsetLoader:
    def __init__(self, args):
        """
        - path: file directory path
        - filelist_path: list of files in path directory
        """
        self.docs = pd.read_csv(f'{args.test_dir}/{args.testset}/test.csv')
        self.batch_size = args.batch_size
        self.egrids = []
        with open(f'{args.test_dir}/{args.testset}/Egrid', 'rb') as f:
            while True:
                try:
                    self.egrids.append(pickle.load(f))
                except:
                    break

    def __iter__(self):
        '''
        Mohsen's function to include grids in batches
        :return:
        '''

        batch = []
        batch_grids = []
        assert len(self.docs) == len(self.egrids)

        for i in range(len(self.docs)):
            batch.append(self.docs[i])
            grid = pd.DataFrame.from_dict(self.egrids[i], orient='index')
            batch_grids.append(grid)
            if len(batch) == self.batch_size:
                yield batch, batch_grids
                batch = []
                batch_grids = []

        if len(batch) != 0:
            yield batch, batch_grids





TYPE_RELATIONS = {'S', 'O', 'X'}

MAP_EDGE_LABELS = {'SS': 0,
                   'OO': 1,
                   'XX': 2,
                   'SO': 3,
                   'OS': 4,
                   'SX': 5,
                   'XS': 6,
                   'OX': 7,
                   'XO': 8,
                   'ADJ': 9}




def ent_edges(grid, type_graph):

    edge_set = set()

    if type_graph not in ['ent','ent.gr','adj_and_ent','adj_and_ent.gr']:

        return edge_set

    for ent_idx, row in grid.iterrows():

        for src_sent_idx, src_sent_r in enumerate(row[1:]):

            if src_sent_r not in TYPE_RELATIONS:

                continue

            for tgt_sent_idx, tgt_sent_r in enumerate(row[src_sent_idx+2:], start= src_sent_idx+1):

                if tgt_sent_r not in TYPE_RELATIONS:

                    continue

                if type_graph in ['ent.gr', 'adj_and_ent.gr']:

                    edge_label = src_sent_r+tgt_sent_r

                    edge_label_id = MAP_EDGE_LABELS[edge_label]

                elif type_graph in ['ent','adj_and_ent']:

                    edge_label_id = 0

                else:

                    raise NotImplementedError()

                edge = (src_sent_idx,tgt_sent_idx, edge_label_id)

                edge_set.add(edge)

    return edge_set

def adj_edges(grid, type_graph):

    edge_set = set()

    if type_graph not in ['adj','adj_and_ent','adj_and_ent.gr']:

        return edge_set

    for src_sent_idx in range(grid.shape[1]-2):

        tgt_sent_idx = src_sent_idx + 1

        edge_label_id = MAP_EDGE_LABELS['ADJ']

        edge = (src_sent_idx, tgt_sent_idx, edge_label_id)

        edge_set.add(edge)

    return edge_set

def compute_grid(grid, type_graph):

    edge_set = set()

    if type_graph in ['ent','ent.gr','adj_and_ent','adj_and_ent.gr']:

        ent_edge_set = ent_edges(grid, type_graph)

        edge_set = edge_set.union(ent_edge_set)


    if type_graph in ['adj','adj_and_ent','adj_and_ent.gr']:

        adj_edge_set = adj_edges(grid, type_graph)

        edge_set = edge_set.union(adj_edge_set)

    return edge_set

def create_graphs_from_batch(b, type_graph='adj_and_ent'):
    '''

    :param b:
    :param type_graph:  'adj','ent','ent.gr','adj_and_ent','adj_and_ent.gr'
    :return:
    '''

    batch = []

    for d in b:
        graphs = create_graphs_from_grids(d, type_graph)

        batch.append(graphs[0])

    return batch

def create_graphs_from_grids(d, type_graph='adj_and_ent'):

    graphs = []

    for grid in d:

        edge_set = compute_grid(grid, type_graph)

        edge_index_1 = []
        edge_index_2 = []
        edges_types = []

        set_nodes = set()
        for edge in edge_set:
            node1, node2, rel = edge

            edge_index_1.append(node1)
            edge_index_2.append(node2)
            edges_types.append(rel)

            set_nodes.add(node1)
            set_nodes.add(node2)

        number_nodes = len(set_nodes)

        if True:
            edges_index = torch.tensor([edge_index_1, edge_index_2], dtype=torch.long)
        else:
            edges_index = torch.tensor([edge_index_2, edge_index_1], dtype=torch.long)

        edges_types = torch.tensor(edges_types, dtype=torch.long)

        graphs.append((edges_index, edges_types, number_nodes))

    return graphs


from itertools import islice

def chunks(data, SIZE=20):
    it = iter(data)
    for i in range(0, len(data), SIZE):
        yield {k:data[k] for k in islice(it, SIZE)}



def create_graph(d, type_graph):

    print("ola")

    d['graphs'] = {}

    for k, grid in d['grid'].items():

        print(grid)
        edge_set = compute_grid(grid, type_graph)
                #print(element)

        edge_list = []
        edge_label_list = []
        for edge in edge_set:
            node1, node2, rel = edge
            edge_list.append((node1, node2))
            edge_label_list.append(rel)

        d['graphs'][k] = (edge_list, edge_label_list)


def create_graphs_list(documents, type_graph):

    for k, d in documents.items():
        create_graph(d, type_graph)

    return documents

def create_graphs(docs, type_graph='both_directions'):
    chunk_docs = chunks(docs)

    for c, d in enumerate(chunk_docs):
        create_graphs_list(d, type_graph)


