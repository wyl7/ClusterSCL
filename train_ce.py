import argparse
from sklearn import metrics

import torch

from networks.graphcnn import SupConGraphCNN
from util import *


def parse_option():
    parser = argparse.ArgumentParser('argument for training')

    parser.add_argument('--epochs', type=int, default=10000,
                        help='number of training epochs (default: 10000), using early stopping')
    parser.add_argument('--patience', type=int, default=100,
                        help='number of patience (default: 100)')
    parser.add_argument('--device', type=str, default="0",
                        help='the id of the used GPU device (default: 0)')
    parser.add_argument('--seed', type=int, default="0",
                        help='the random seed (default: 0)')

    # optimization
    parser.add_argument('--learning_rate', type=float, default=0.1,
                        help='learning rate')
    parser.add_argument('--weight_decay', type=float, default=1e-4,
                        help='weight decay')

    # dataset
    parser.add_argument('--dataset', type=str, default='cora', help='dataset')
    parser.add_argument('--train_nodes_per_class', type=int, default=20,
                        help='number of nodes per class in the training set')

    # parameters in GraphCNN
    parser.add_argument('--num_graphcnn_layers', type=int, default=2,
                        help='number of layers in graphcnn (not including the input layer)')
    parser.add_argument('--hidden_dim', type=int, default=64,
                        help='dimensionality of hidden units at ALL layers')
    parser.add_argument('--num_head', type=int, default=8,
                        help='the number of heads in GAT layer (default: 8)')

    opt = parser.parse_args()

    return opt


def main():
    opt = parse_option()
    setup_seed(opt.seed)
    device = torch.device("cuda:" + str(opt.device)) if torch.cuda.is_available() else torch.device("cpu")

    #----------------------------------------------prepare dataset----------------------------------------------
    # Load data
    adj, feats, train_idx, val_idx, test_idx, labels, nb_nodes, ss_label = load_data(opt.dataset, opt.seed, 1, opt.train_nodes_per_class)

    opt.input_dim = feats.shape[1]
    opt.n_cls = len(set(labels[:, 1]))

    adj = adj.to(device)
    feats = torch.FloatTensor(feats).to(device)

    #----------------------------------------prepare model/loss/optimizer----------------------------------------
    model = SupConGraphCNN(opt.num_graphcnn_layers, opt.input_dim, opt.hidden_dim, opt.num_head, head='mlp').to(device)
    classifier = torch.nn.Linear(opt.hidden_dim, opt.n_cls).to(device)

    optimizer = torch.optim.Adam([{'params': model.parameters()},
                                  {'params': classifier.parameters()}],
                                  lr = opt.learning_rate,
                                  weight_decay= opt.weight_decay)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=50, gamma=0.5)
   
    #------------------------------------------------start training------------------------------------------------
    # training via CE / end-to-end model
    node_ids = labels[train_idx, 0].astype('int')
    node_labels = torch.LongTensor(labels[train_idx, 1]).to(device)

    cnt_wait = 0
    best = 0
    best_t = 0
    best_test = 0
    for epoch in range(1, opt.epochs+1):
        model.train()
        classifier.train()
        output = classifier(model.encoder(feats, adj))
        loss = torch.nn.CrossEntropyLoss()(output[node_ids], node_labels)
    
        #backprop
        if optimizer is not None:
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            scheduler.step()

        with torch.no_grad():
            val_acc = validate(val_idx, feats, adj, labels, model, classifier, opt)
            test_acc = validate(test_idx, feats, adj, labels, model, classifier, opt)

        if val_acc > best:
            best = val_acc
            best_t = epoch
            cnt_wait = 0
            best_test = test_acc
        else:
            cnt_wait += 1

        if cnt_wait == opt.patience:
            print('Early stopping!')
            break

    print('==================================Final results==================================')
    print("best val acc: {:.3f}, testing acc: {:.3f}".format(best, best_test))

    # save results
    fp=open("save/res_ce_{}_{}.txt".format(opt.dataset, opt.train_nodes_per_class), "a")
    fp.write("learning rate: {:.3f}, best val acc: {:.3f}, testing acc: {:.3f}\n".format(opt.learning_rate, best, best_test))
    fp.close()

    return
    

def validate(val_idx, feats, adj, labels, model, classifier, opt):
    """validation"""
    model.eval()
    classifier.eval()

    node_ids = labels[val_idx, 0].astype('int')
    node_labels = labels[val_idx, 1]

    with torch.no_grad():
        output = classifier(model.encoder(feats, adj))
        
    pred = torch.nn.Softmax(dim=1)(output.detach().cpu())[node_ids]
    pred = np.argmax(pred.numpy(), axis=1)
    
    return metrics.accuracy_score(node_labels, pred)


    
          

if __name__ == '__main__':
    main()