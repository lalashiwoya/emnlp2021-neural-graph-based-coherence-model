import argparse

def argument_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument('--experiment_path', type=str, default="Experiments/WSJ/ours/")
    parser.add_argument('--test_dir', type=str, default="testsets")
    parser.add_argument('--testset', type=str, required = True)
    parser.add_argument('--seed', type=int, default=0, help='seed value')
    parser.add_argument('--pre_embedding_path', type=str,
                        default="./data/GoogleNews-vectors-negative300.bin", help='Pretrained word embedding path')
    parser.add_argument('--vocab_path', type=str,
                        default="./data/Dataset_Global/Dataset/vocab/Vocab", help='Vocab path')
    parser.add_argument('--padding_symbol', type=str,
                        default="<pad>", help='Vocab path')
    # Training Parameter-------------------------------------------------------------

    parser.add_argument('--device', type=str, default='cuda', help='cpu? cuda?')
    # Minibatch argument

    parser.add_argument('--batch_size_test', type=int,
                        default=3, help='Mini batch size for test/dev')

    # Network Parameter
    parser.add_argument('--n_vocabs', type=int,
                        help='Word embedding dim, it should be defined using the vocab list')
    parser.add_argument('--embed_dim', type=int,
                        default=300, help='Word embedding dim')
    # RNN Parameter
    parser.add_argument('--hidden_dim', type=int,
                        default=256, help='Hidden dim of RNN')
    parser.add_argument('--num_layers', type=int, default=1,
                        help='Number of RNN layers')
    parser.add_argument('--dropout', type=float, default=0,
                        help='Dropout ratio of RNN')
    parser.add_argument('--bidirectional', type=bool,
                        default=True, help='Bi-directional RNN?')
    parser.add_argument('--batch_first', type=bool, default=True, help='Dimension order')
    # Light-weight convolution Parameters
    parser.add_argument('--num_head', type=int, default=16,
                        help='Number of heads in DyConv')
    parser.add_argument('--kernel_size', type=int,
                        default=5, help='Kernel size of DyConv')
    parser.add_argument('--conv_dropout', type=float,
                        default=.0, help='DyConv kernel dropout rate')
    parser.add_argument('--kernel_padding', type=int,
                        default=3, help='DyConv kernel padding')
    parser.add_argument('--kernel_softmax', type=bool,
                        default=True, help='DyConv kernel softmax')

    embedding = parser.add_mutually_exclusive_group()
    embedding.add_argument('--GoogleEmbedding', type=bool,
                           default=False, help='Google embedding')
    embedding.add_argument('--RandomEmbedding', type=bool,
                           default=False, help='Random embedding')
    embedding.add_argument('--ELMo', type=bool,
                           default=False, help='ELMo embedding')
    parser.add_argument('--ELMo_Size', type=str,
                        default='small', help='Size of ELMo')
    parser.add_argument('--bilinear_dim', type=int,
                        default=32, help='bilinear output dim')
    parser.add_argument('--dataset', type=str, default='data-global',
                        help='Which data-set? Options: data-tokenized, data-full, data-global')

    parser.add_argument('--logging',
                        type=bool,
                        default=True,
                        help='True: save the stdout in a log file (log.out) in the experiment folder '
                             'False: display the stdout on screen --> no saved log.out')

    parser.add_argument('--model_name', type=str, default='',
                        help='the model that should be loaded when --only_eval is True')

    parser.add_argument('--graph_type', type=str, default="adj_and_ent",
                        help='options: adj,ent,ent.gr,adj_and_ent,adj_and_ent.gr')


    parser.add_argument('--task', type=str,
                        default="scr",
                        help='options: so, scr')

    parser.add_argument('--method',
                        type=str,
                        default="ours",
                        help="options: unified, ours")

    return parser

def prob_file_dict():
    file_dict = {'pronoun_down':"lingprobe_insted_animacy_pronoun_downgrade",
                 'pronoun_up':"lingprobe_insted_animacy_pronoun_upgrade",
                 "conj":"lingprobe_insted_conjunction_flip",
                 "gender":"lingprobe_insted_gender_pronoun_flip",
                 "number":"lingprobe_insted_number_exaggerate",
                 "past":"lingprobe_insted_past_to_future_flip",
                 "singular":"lingprobe_insted_singular_determiner_flip",
                 "neg":"lingprobe_insted_to_negation"
}
    return file_dict

elif args.testset.startswith('prob'):
    file_dict = prob_file_dict()
    name = file_dict[args.testset.split('_')[-1]]
    test_df = pd.read_csv(f'datasets/prob_tests/{name}.csv')
