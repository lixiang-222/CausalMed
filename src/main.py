import argparse

import dill
import numpy as np
import torch

from modules.CausalMed import CausalMed
# from modules.causal_construction_easyuse import CausaltyGraph4Visit
from modules.causal_construction import CausaltyGraph4Visit
from training import Test, Train


def set_seed():
    torch.manual_seed(1203)
    np.random.seed(2048)


def parse_args():
    parser = argparse.ArgumentParser()
    # mode
    parser.add_argument("--debug", default=False,
                        help="debug mode, the number of samples, "
                             "the number of generations run are very small, "
                             "designed to run on cpu, the development of the use of")
    parser.add_argument("--Test", default=False, help="test mode")

    # environment
    parser.add_argument('--dataset', default='mimic4', help='mimic3/mimic4')
    parser.add_argument('--resume_path', default="../saved/mimic3/trained_model_0.5405", type=str,
                        help='path of well trained model, only for evaluating the model, needs to be replaced manually')
    parser.add_argument('--device', type=int, default=0, help='gpu id to run on, negative for cpu')

    # parameters
    parser.add_argument('--dim', default=64, type=int, help='model dimension')
    parser.add_argument('--lr', default=5e-4, type=float, help='learning rate')
    parser.add_argument('--dp', default=0.7, type=float, help='dropout ratio')
    parser.add_argument("--regular", type=float, default=0.005, help="regularization parameter")
    parser.add_argument('--target_ddi', type=float, default=0.06, help='expected ddi for training')
    parser.add_argument('--coef', default=2.5, type=float, help='coefficient for DDI Loss Weight Annealing')
    parser.add_argument('--epochs', default=50, type=int, help='the epochs for training')

    args = parser.parse_args()
    if args.Test and args.resume_path is None:
        raise FileNotFoundError('Can\'t Load Model Weight From Empty Dir')

    return args


if __name__ == '__main__':
    set_seed()
    args = parse_args()
    print(args)
    if not torch.cuda.is_available() or args.device < 0:
        device = torch.device("cpu")
        if not args.Test:
            print("GPU unavailable, switch to debug mode")
            args.debug = True
    else:
        device = torch.device(f'cuda:{args.device}')

    data_path = f'../data/{args.dataset}/output/records_final.pkl'
    voc_path = f'../data/{args.dataset}/output/voc_final.pkl'
    ddi_adj_path = f'../data/{args.dataset}/output/ddi_A_final.pkl'
    ddi_mask_path = f'../data/{args.dataset}/output/ddi_mask_H.pkl'
    with open(ddi_adj_path, 'rb') as Fin:
        ddi_adj = torch.from_numpy(dill.load(Fin)).to(device)
    with open(ddi_mask_path, 'rb') as Fin:
        ddi_mask_H = torch.from_numpy(dill.load(Fin)).to(device)
    with open(data_path, 'rb') as Fin:
        data = dill.load(Fin)
        adm_id = 0
        for patient in data:
            for adm in patient:
                adm.append(adm_id)
                adm_id += 1
        if args.debug:
            data = data[:5]
    with open(voc_path, 'rb') as Fin:
        voc = dill.load(Fin)

    diag_voc, pro_voc, med_voc = voc['diag_voc'], voc['pro_voc'], voc['med_voc']
    voc_size = (
        len(diag_voc.idx2word),
        len(pro_voc.idx2word),
        len(med_voc.idx2word)
    )

    split_point = int(len(data) * 2 / 3)
    data_train = data[:split_point]
    eval_len = int(len(data[split_point:]) / 2)
    data_test = data[split_point:split_point + eval_len]
    data_eval = data[split_point + eval_len:]

    causal_graph = CausaltyGraph4Visit(data, data_train, voc_size[0], voc_size[1], voc_size[2], args.dataset)

    model = CausalMed(
        causal_graph=causal_graph,
        tensor_ddi_adj=ddi_adj,
        dropout=args.dp,
        emb_dim=args.dim,
        voc_size=voc_size,
        device=device
    ).to(device)

    print("1.Training Phase")
    if args.Test:
        print("Test mode, skip training phase")
        with open(args.resume_path, 'rb') as Fin:
            model.load_state_dict(torch.load(Fin, map_location=device))
    else:
        model = Train(model, device, data_train, data_eval, voc_size, args)

    print("2.Testing Phase")
    Test(model, device, data_test, voc_size)
