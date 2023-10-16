from operator import index
import torch 
from collections import defaultdict
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.model_selection import StratifiedKFold
from rdkit import Chem
import pandas as pd
import numpy as np
from tqdm import tqdm
import pickle
import os
from rdkit.Chem import rdMolDescriptors
from rdkit.Chem import AllChem
import re
#from Featurize import Coformer
filename1 = f'D:\code\smiles2vec\drug_data_smi.pkl'
with open(filename1, 'rb') as f1:
    smile_data = pickle.load(f1)

word = [
        "#",
        "$",
        "&",
        "(",
        ")",
        "-",
        "/",
        "1",
        "2",
        "3",
        "4",
        "5",
        "6",
        "7",
        "8",
        "9",
        "=",
        "B",
        "Br",
        "C",
        "Cl",
        "F",
        "I",
        "N",
        "O",
        "P",
        "S",
        "[125I]",
        "[18F]",
        "[2H]",
        "[3H]",
        "[AlH2]",
        "[As]",
        "[Au]",
        "[B-]",
        "[C-]",
        "[C@@H]",
        "[C@@]",
        "[C@H]",
        "[C@]",
        "[CH-]",
        "[Cr]",
        "[Fe--]",
        "[Fe@@]",
        "[Fe@]",
        "[Fe]",
        "[Hg]",
        "[K]",
        "[Li]",
        "[Mg]",
        "[Mo]",
        "[N+]",
        "[N-]",
        "[N@+]",
        "[N@@+]",
        "[N@@]",
        "[N@H+]",
        "[N@]",
        "[NH+]",
        "[NH-]",
        "[NH2+]",
        "[NH3+]",
        "[N]",
        "[Na]",
        "[O+]",
        "[O-]",
        "[OH+]",
        "[O]",
        "[P+]",
        "[P@@]",
        "[P@]",
        "[PH]",
        "[P]",
        "[Pd]",
        "[Re]",
        "[Ru@@]",
        "[Ru]",
        "[S+]",
        "[S-]",
        "[S@+]",
        "[S@@+]",
        "[S@@H]",
        "[S@@]",
        "[S@H]",
        "[S@]",
        "[SH]",
        "[Sc]",
        "[S]",
        "[Sb]",
        "[SeH]",
        "[Se]",
        "[Si]",
        "[SnH]",
        "[Sn]",
        "[V]",
        "[Zn++]",
        "[c-]",
        "[n+]",
        "[n-]",
        "[nH+]",
        "[nH]",
        "[o+]",
        "[s+]",
        "[se]",
        "[W]",
        "[C]",
        "[c]",
        "[I+]",
        "[CH]",
        "\\",
        "^",
        "c",
        "n",
        "o",
        "p",
        "s",
        "."
    ]


def splitSmi(smi):
    '''
    description: 将smiles拆解为最小单元
    param {*} smi
    return {*}
    '''
    pattern =  "(\[[^\]]+]|Br?|Cl?|N|O|S|P|F|I|b|c|n|o|s|p|\(|\)|\.|=|#|-|\+|\\\\|\/|:|~|@|\?|>|\*|\$|\%[0-9]{2}|[0-9])"
    regex = re.compile(pattern)
    tokens = [token for token in regex.findall(smi)]
    assert smi == ''.join(tokens)
    return tokens

def fetchIndices(smiSplit, smiVoc, smiMaxLen):
    smiIndices = []
    # padding symbol: ^ ;
    smiSplit.extend(['^'] * (smiMaxLen - len(smiSplit)))
    smiIndices.append([smiVoc.index(smi) for smi in smiSplit])

    return np.array(smiIndices)

try:
    table = eval(open('./data/Mol_Blocks.dir').read())
except:
    able = [i.strip().split('\t') for i in open('./data/Mol_Blocks.dir').readlines()]
mol_blocks = eval(open('./data/Mol_Blocks.dir').read())

def one_of_k_encoding(k, possible_values):
    if k not in possible_values:
        raise ValueError(f"{k} is not a valid value in {possible_values}")
    return [k == e for e in possible_values]


def one_of_k_encoding_unk(x, allowable_set):
    if x not in allowable_set:
        x = allowable_set[-1]
    return list(map(lambda s: x == s, allowable_set))


def atom_features(atom, atom_symbols, explicit_H=True, use_chirality=False):

    results = one_of_k_encoding_unk(atom.GetSymbol(), atom_symbols + ['Unknown']) + \
            one_of_k_encoding(atom.GetDegree(),[0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10]) + \
            one_of_k_encoding_unk(atom.GetImplicitValence(), [0, 1, 2, 3, 4, 5, 6]) + \
                [atom.GetFormalCharge(), atom.GetNumRadicalElectrons()] + \
            one_of_k_encoding_unk(atom.GetHybridization(), [
                Chem.rdchem.HybridizationType.SP, Chem.rdchem.HybridizationType.SP2,
                Chem.rdchem.HybridizationType.SP3, Chem.rdchem.HybridizationType.
                                    SP3D, Chem.rdchem.HybridizationType.SP3D2
                ]) + [atom.GetIsAromatic()]
    # In case of explicit hydrogen(QM8, QM9), avoid calling `GetTotalNumHs`
    if explicit_H:
        results = results + one_of_k_encoding_unk(atom.GetTotalNumHs(),
                                                [0, 1, 2, 3, 4])
    if use_chirality:
        try:
            results = results + one_of_k_encoding_unk(
            atom.GetProp('_CIPCode'),
            ['R', 'S']) + [atom.HasProp('_ChiralityPossible')]
        except:
            results = results + [False, False
                            ] + [atom.HasProp('_ChiralityPossible')]

    results = np.array(results).astype(np.float32)
    #print(torch.from_numpy(results))
    return torch.from_numpy(results)


def edge_features(bond):
    bond_type = bond.GetBondType()
    return torch.tensor([
        bond_type == Chem.rdchem.BondType.SINGLE,
        bond_type == Chem.rdchem.BondType.DOUBLE,
        bond_type == Chem.rdchem.BondType.TRIPLE,
        bond_type == Chem.rdchem.BondType.AROMATIC,
        bond.GetIsConjugated(),
        bond.IsInRing()]).long()


def generate_drug_data(mol_graph, atom_symbols,smiles,id):
    edge_list = torch.LongTensor([(b.GetBeginAtomIdx(), b.GetEndAtomIdx(), *edge_features(b)) for b in mol_graph.GetBonds()])
    edge_list, edge_feats = (edge_list[:, :2], edge_list[:, 2:].float()) if len(edge_list) else (torch.LongTensor([]), torch.FloatTensor([]))
    #print(edge_list.shape)
    #print(edge_list)
    edge_list = torch.cat([edge_list, edge_list[:, [1, 0]]], dim=0) if len(edge_list) else edge_list
    edge_feats = torch.cat([edge_feats]*2, dim=0) if len(edge_feats) else edge_feats
    contribs = rdMolDescriptors._CalcCrippenContribs(mol_graph)
    chemfeat = torch.tensor(contribs[:3])
    fp = AllChem.GetMorganFingerprintAsBitVect(mol_graph, 2)
    fp = torch.tensor(fp)
    #print(smiles)
    smword = splitSmi(smiles)
    smvec = fetchIndices(smword , word , 94)
    smvec = smvec.squeeze()
    smvec = torch.tensor(smvec)
    if chemfeat.shape[0]==1:
        chemfeat = torch.reshape(chemfeat, [2])
        a = torch.tensor([0,0,0,0])
        chemfeat = torch.cat((chemfeat,a),dim = 0)
    elif chemfeat.shape[0]==2:
        chemfeat = torch.reshape(chemfeat, [4])
        a = torch.tensor([0,0])
        chemfeat = torch.cat((chemfeat,a),dim = 0)
    else:
        chemfeat = torch.reshape(chemfeat, [6])
    #print(type(contribs[:3]))
    features = [(atom.GetIdx(), atom_features(atom, atom_symbols)) for atom in mol_graph.GetAtoms()]
    features.sort() 
    _, features = zip(*features)
    features = torch.stack(features)
    #print(features)
    #print(type(features))
    '''try:
        c = Coformer(mol_blocks[smiles])
        #print(type(features))
    except:
        c = Coformer(mol_blocks[str(int(id[3:]))])
        #print(type(features))
    features = c.VertexMatrix.feature_matrix()
    features = torch.from_numpy(features)
    A = c.AdjacentTensor.OnlyCovalentBond(with_coo=True)
    a = np.hstack((A[0][0],A[0][1]))
    b = np.hstack((A[0][1],A[0][0]))
    edge_list = np.vstack((a,b))
    edge_list = torch.from_numpy(edge_list).T
    edge_list = edge_list.long()
    edge_feats = np.vstack((A[1],A[1]))
    edge_feats = torch.from_numpy(edge_feats)
    edge_feats = edge_feats.float()'''
    #print(edge_list.shape)
    #print(edge_feats.shape)
    #print(features)
    #print(features.shape)
    #print(edge_list.dtype)
    #print(edge_feats.dtype)
    line_graph_edge_index = torch.LongTensor([])

    if edge_list.nelement() != 0:
        conn = (edge_list[:, 1].unsqueeze(1) == edge_list[:, 0].unsqueeze(0)) & (edge_list[:, 0].unsqueeze(1) != edge_list[:, 1].unsqueeze(0))
        line_graph_edge_index = conn.nonzero(as_tuple=False).T

    '''print(edge_list[:, 1].unsqueeze(1))
    print(edge_list[:, 0].unsqueeze(0))
    print(edge_list[:, 1].unsqueeze(1) == edge_list[:, 0].unsqueeze(0))
    print(edge_list[:, 0].unsqueeze(1))
    print(edge_list[:, 1].unsqueeze(0))
    print(edge_list[:, 0].unsqueeze(1) != edge_list[:, 1].unsqueeze(0))
    print(conn)
    print(line_graph_edge_index)'''
    new_edge_index = edge_list.T
    #print(features)
    return features, new_edge_index, edge_feats, line_graph_edge_index, chemfeat , smiles , fp , smvec, smile_data[smiles].mean(axis = 0)


def load_drug_mol_data(args):

    data = pd.read_csv(args.dataset_filename, delimiter=args.delimiter)
    drug_id_mol_tup = []
    symbols = list()
    drug_smile_dict = {}

    for id1, id2, smiles1, smiles2, relation in zip(data[args.c_id1], data[args.c_id2], data[args.c_s1], data[args.c_s2], data[args.c_y]):
        drug_smile_dict[id1] = smiles1
        drug_smile_dict[id2] = smiles2
    print(len(drug_smile_dict))
    for id, smiles in drug_smile_dict.items():
        mol =Chem.MolFromSmiles(smiles.strip())
        if mol is not None:
            drug_id_mol_tup.append((id,smiles ,mol))
            symbols.extend(atom.GetSymbol() for atom in mol.GetAtoms())
    print(len(drug_id_mol_tup))

    symbols = list(set(symbols))
    drug_data = {id: generate_drug_data(mol, symbols,smiles,id) for id, smiles,mol in tqdm(drug_id_mol_tup, desc='Processing drugs')}
    #print(drug_data['CID000001057'])
    save_data(drug_data, 'drug_data.pkl', args)
    #print(drug_data['id'])
    return drug_data


def generate_pair_triplets(args):
    pos_triplets = []
    drug_ids = []

    with open(f'{args.dirname}/{args.dataset.lower()}/drug_data.pkl', 'rb') as f:
        drug_ids = list(pickle.load(f).keys())

    data = pd.read_csv(args.dataset_filename, delimiter=args.delimiter)
    for id1, id2, relation in zip(data[args.c_id1], data[args.c_id2],  data[args.c_y]):
        if ((id1 not in drug_ids) or (id2 not in drug_ids)): continue
        # Drugbank dataset is 1-based index, need to substract by 1
        if args.dataset in ('drugbank', ):
            relation -= 1
        pos_triplets.append([id1, id2, relation])

    if len(pos_triplets) == 0:
        raise ValueError('All tuples are invalid.')

    pos_triplets = np.array(pos_triplets)
    data_statistics = load_data_statistics(pos_triplets)
    drug_ids = np.array(drug_ids)

    '''neg_samples = []
    for pos_item in tqdm(pos_triplets, desc='Generating Negative sample'):
        temp_neg = []
        h, t, r = pos_item[:3]

        if args.dataset == 'drugbank':
            neg_heads, neg_tails = _normal_batch(h, t, r, args.neg_ent, data_statistics, drug_ids, args)
            temp_neg = [str(neg_h) + '$h' for neg_h in neg_heads] + \
                        [str(neg_t) + '$t' for neg_t in neg_tails]
        else:
            existing_drug_ids = np.asarray(list(set(
                np.concatenate([data_statistics["ALL_TRUE_T_WITH_HR"][(h, r)], data_statistics["ALL_TRUE_H_WITH_TR"][(h, r)]], axis=0)
                )))
            temp_neg = _corrupt_ent(existing_drug_ids, args.neg_ent, drug_ids, args)
        
        neg_samples.append('_'.join(map(str, temp_neg[:args.neg_ent])))'''
    
    df = pd.DataFrame({'Drug1_ID': pos_triplets[:, 0], 
                        'Drug2_ID': pos_triplets[:, 1],
                        'Y': pos_triplets[:, 2]})
                        #,'Neg samples': neg_samples})
    filename = f'{args.dirname}/{args.dataset}/pair_pos_neg_triplets.csv'
    df.to_csv(filename, index=False)
    print(f'\nData saved as {filename}!')
    save_data(data_statistics, 'data_statistics.pkl', args)


def load_data_statistics(all_tuples):
    
    print('Loading data statistics ...')
    statistics = dict()
    statistics["ALL_TRUE_H_WITH_TR"] = defaultdict(list)
    statistics["ALL_TRUE_T_WITH_HR"] = defaultdict(list)
    statistics["FREQ_REL"] = defaultdict(int)
    statistics["ALL_H_WITH_R"] = defaultdict(dict)
    statistics["ALL_T_WITH_R"] = defaultdict(dict)
    statistics["ALL_TAIL_PER_HEAD"] = {}
    statistics["ALL_HEAD_PER_TAIL"] = {}

    for h, t, r in tqdm(all_tuples, desc='Getting data statistics'):
        statistics["ALL_TRUE_H_WITH_TR"][(t, r)].append(h)
        statistics["ALL_TRUE_T_WITH_HR"][(h, r)].append(t)
        statistics["FREQ_REL"][r] += 1.0
        statistics["ALL_H_WITH_R"][r][h] = 1
        statistics["ALL_T_WITH_R"][r][t] = 1

    for t, r in statistics["ALL_TRUE_H_WITH_TR"]:
        statistics["ALL_TRUE_H_WITH_TR"][(t, r)] = np.array(list(set(statistics["ALL_TRUE_H_WITH_TR"][(t, r)])))
    for h, r in statistics["ALL_TRUE_T_WITH_HR"]:
        statistics["ALL_TRUE_T_WITH_HR"][(h, r)] = np.array(list(set(statistics["ALL_TRUE_T_WITH_HR"][(h, r)])))

    for r in statistics["FREQ_REL"]:
        statistics["ALL_H_WITH_R"][r] = np.array(list(statistics["ALL_H_WITH_R"][r].keys()))
        statistics["ALL_T_WITH_R"][r] = np.array(list(statistics["ALL_T_WITH_R"][r].keys()))
        statistics["ALL_HEAD_PER_TAIL"][r] = statistics["FREQ_REL"][r] / len(statistics["ALL_T_WITH_R"][r])
        statistics["ALL_TAIL_PER_HEAD"][r] = statistics["FREQ_REL"][r] / len(statistics["ALL_H_WITH_R"][r])
    
    print('getting data statistics done!')

    return statistics


def _corrupt_ent(positive_existing_ents, max_num, drug_ids, args):
    corrupted_ents = []
    while len(corrupted_ents) < max_num:
        candidates = args.random_num_gen.choice(drug_ids, (max_num - len(corrupted_ents)) * 2, replace=False)
        invalid_drug_ids = np.concatenate([positive_existing_ents, corrupted_ents], axis=0)
        mask = np.isin(candidates, invalid_drug_ids, assume_unique=True, invert=True)
        corrupted_ents.extend(candidates[mask])

    corrupted_ents = np.array(corrupted_ents)[:max_num]
    return corrupted_ents


def _normal_batch( h, t, r, neg_size, data_statistics, drug_ids, args):
    neg_size_h = 0
    neg_size_t = 0
    prob = data_statistics["ALL_TAIL_PER_HEAD"][r] / (data_statistics["ALL_TAIL_PER_HEAD"][r] + 
                                                            data_statistics["ALL_HEAD_PER_TAIL"][r])
    # prob = 2
    for i in range(neg_size):
        if args.random_num_gen.random() < prob:
            neg_size_h += 1
        else:
            neg_size_t +=1
    
    return (_corrupt_ent(data_statistics["ALL_TRUE_H_WITH_TR"][t, r], neg_size_h, drug_ids, args),
            _corrupt_ent(data_statistics["ALL_TRUE_T_WITH_HR"][h, r], neg_size_t, drug_ids, args))  


def save_data(data, filename, args):
    dirname = f'{args.dirname}/{args.dataset}'
    if not os.path.exists(dirname):
        os.makedirs(dirname)
    filename = dirname + '/' + filename
    with open(filename, 'wb') as f:
        pickle.dump(data, f)
    print(f'\nData saved as {filename}!')


def split_data(args):
    filename = f'{args.dirname}/{args.dataset}/pair_pos_neg_triplets.csv'
    df = pd.read_csv(filename)
    seed = args.seed
    class_name = args.class_name
    test_size_ratio = args.test_ratio
    n_folds = args.n_folds 
    save_to_filename = os.path.splitext(filename)[0]
    cv_split = StratifiedKFold(n_splits=n_folds,shuffle=True ,random_state=seed)
    for fold_i,(train_index, test_index) in enumerate(cv_split.split(X=df, y=df[class_name])):
        print(f'Fold {fold_i} generated!')
        train_df = df.iloc[train_index]
        test_df = df.iloc[test_index]
        train_df.to_csv(f'{save_to_filename}_train_fold{fold_i}.csv', index=False)
        print(f'{save_to_filename}_train_fold{fold_i}.csv', 'saved!')
        test_df.to_csv(f'{save_to_filename}_test_fold{fold_i}.csv', index=False)
        print(f'{save_to_filename}_test_fold{fold_i}.csv', 'saved!')
    '''cv_split = StratifiedShuffleSplit(n_splits=n_folds,test_size=test_size_ratio, random_state=seed)
    for fold_i, (train_index, test_index) in enumerate(cv_split.split(X=df, y=df[class_name])):
        print(f'Fold {fold_i} generated!')
        train_df = df.iloc[train_index]
        test_df = df.iloc[test_index]
        train_df.to_csv(f'{save_to_filename}_train_fold{fold_i}.csv', index=False)
        print(f'{save_to_filename}_train_fold{fold_i}.csv', 'saved!')
        test_df.to_csv(f'{save_to_filename}_test_fold{fold_i}.csv', index=False)
        print(f'{save_to_filename}_test_fold{fold_i}.csv', 'saved!')'''


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()

    parser.add_argument('-d', '--dataset', type=str, required=True, choices=['drugbank', 'twosides'], 
                            help='Dataset to preprocess.')
    parser.add_argument('-n', '--neg_ent', type=int, default=1, help='Number of negative samples')
    parser.add_argument('-s', '--seed', type=int, default=0, help='Seed for the random number generator')
    parser.add_argument('-o', '--operation', type=str, required=True, choices=['all', 'generate_triplets', 'drug_data', 'split'], help='Operation to perform')
    parser.add_argument('-t_r', '--test_ratio', type=float, default=0.1)
    parser.add_argument('-n_f', '--n_folds', type=int, default=3)

    dataset_columns_map = {
        'drugbank': ('ID1', 'ID2', 'X1', 'X2', 'Y'),
        'twosides': ('Drug1_ID', 'Drug2_ID', 'Drug1', 'Drug2', 'New Y'),
    }

    dataset_file_name_map = {
        'drugbank': ('data/drugbank.tab', '\t'),
        'twosides': ('data/twosides_ge_500.csv', ',')
    }
    args = parser.parse_args()
    args.dataset = args.dataset.lower()

    args.c_id1, args.c_id2, args.c_s1, args.c_s2, args.c_y = dataset_columns_map[args.dataset]
    args.dataset_filename, args.delimiter = dataset_file_name_map[args.dataset]
    args.dirname = 'data/preprocessed'

    args.random_num_gen = np.random.RandomState(args.seed)
    if args.operation in ('all', 'drug_data'):
        load_drug_mol_data(args)

    if args.operation in ('all','generate_triplets'):
        generate_pair_triplets(args)
    
    if args.operation in ('all', 'split'):
        args.class_name = 'Y'
        split_data(args)
