import torch
# import gc
import torch.nn as nn
import numpy as np
import sys
import os
import random
from torch_geometric.nn import GATConv, GINConv, GatedGraphConv, RGCNConv
from src import utils, model, lm_model, data_load, utils_graph
import eval_utils, eval_args

parser = eval_args.argument_parser()
args = parser.parse_args()

if args.ELMo:
    print("**ELMo word Embeddings!")
    parser.set_defaults(learning_rate_step=2, embed_dim=256, GoogleEmbedding=False)
elif args.GoogleEmbedding:
    print("**word2vec Embeddings!")
else:
    print("**Random Embeddings!")

args = parser.parse_args()

if args.logging:
    log_path = os.path.join(args.experiment_folder, 'log.out')
    print(f"log_path: {log_path}")
    sys.stdout = open(log_path, "w")  # MM: redirect the output of print to a file

# MM
random.seed(args.seed)
np.random.seed(args.seed)
torch.manual_seed(args.seed)
torch.random.manual_seed(args.seed)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
# random.seed(0)
# torch.manual_seed(6)


# vocabs contain all vocab + <pad>, <bos>, <eos>, <unk>
args.vocabs = utils.load_file(args.vocab_path, file_type='json')
args.n_vocabs = len(args.vocabs)
args.word2idx = {tok: i for i, tok in enumerate(args.vocabs)}
args.idx2word = {i: tok for i, tok in enumerate(args.vocabs)}
args.padding_idx = args.word2idx[args.padding_symbol]

batch_generator_test = eval_utils.TestsetLoader(args)

# Sentence encoder
sentence_encoder = model.SentenceEmbeddingModel(args).to(args.device)

# Linear layer
coherence_scorer = model.LocalCoherenceScore(args).to(args.device)

if args.method == 'ours':

    atten1 = model.MultiHeadAttention(args).to(args.device)

    gnn1 = RGCNConv(args.hidden_dim * 2,
                    args.hidden_dim * 2,
                    10).to(args.device)

    local_global_model = nn.Sequential(sentence_encoder,
                                       gnn1,
                                       atten1,
                                       coherence_scorer)
else:
    raise NotImplementedError(f'args.method= {args.method} is unkown')


def calculate_scores_doc(x_batch_info, x_graphs):
    docu_batch_idx, batch_docs_len, batch_sentences_len, modified_batch_sentences_len = x_batch_info

    docu_batch_idx = docu_batch_idx.to(args.device)

    output, hidden = sentence_encoder(docu_batch_idx, modified_batch_sentences_len)

    hidden_out = hidden

    # batch_size, doc_len, hidden_dim*2

    batch_size = hidden_out.shape[0]

    max_sent_len = hidden_out.shape[1]

    graphs_select = utils_graph.get_embs_graph(x_graphs, hidden_out)

    graphs_select = graphs_select.to(args.device)

    hidden_out = hidden_out.view((batch_size * max_sent_len, -1))

    hidden_out = gnn1(hidden_out, graphs_select.edge_index, graphs_select.y)

    hidden_out = hidden_out.view((batch_size, max_sent_len, -1))

    attn_output, attn_output_weights = atten1(query=hidden_out,
                                              key=hidden_out,
                                              value=hidden_out)
    coh_vec = attn_output.mean(dim=1)

    scores = coherence_scorer(coh_vec)

    masked_score = scores

    return masked_score


def calculate_scores_scr(batch_info, graphs):
    pos_score = calculate_scores_doc(batch_info, graphs)
    doc_score = pos_score.mean(dim=1)
    return doc_score


def create_list_of_batches(batch_gen):
    batches = []
    for n_mini_batch, (batch, batch_grids) in enumerate(batch_gen):

        graphs = eval_utils.create_graphs_from_batch(batch_grids, type_graph=args.graph_type)

        if args.ELMo:
            # docu_batch_idx -> 4D Tensor of char_ids for ELMo model [doc->sentences->word->char_ids]
            docu_batch_idx, batch_docs_len, batch_sentences_len, modified_batch_sentences_len = utils.batch_preprocessing_elmo(
                batch, args)

        else:
            # docu_batch_idx -> 3D Tensor of word_ids for general embeddings model [doc->sentences->word_ids]
            docu_batch_idx, batch_docs_len, batch_sentences_len, modified_batch_sentences_len = utils.batch_preprocessing(
                batch, args)
        batch_info = (docu_batch_idx,
                      batch_docs_len,
                      batch_sentences_len,
                      modified_batch_sentences_len)

        batches.append((batch_info, graphs))

    return batches


test_batches = create_list_of_batches(batch_generator_test)

model_save_path = f"{args.experiment_path}/2022_3_13_11_42/Epoch_3"
with torch.no_grad():
    print(f'loading the best model on dev set {model_save_path}')
    local_global_model.load_state_dict(torch.load(model_save_path))
    local_global_model.eval()

    for n_mini_batch, (batch_info, graphs) in enumerate(test_batches):
        score = calculate_scores_scr(batch_info, graphs)
