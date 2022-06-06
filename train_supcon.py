import argparse
from sklearn import metrics

import torch

from networks.graphcnn import SupConGraphCNN
from losses import SupConLoss
from util import *

def parse_option():
    parser = argparse.ArgumentParser('argument for training')

    parser.add_argument('--batch_size', type=int, default=32,
                        help='batch_size (default: 32)')
    parser.add_argument('--epochs', type=int, default=10000,
                        help='number of training epochs (default: 10000), using early stopping')
    parser.add_argument('--patience', type=int, default=100,
                        help='number of patience (default: 100)')
    parser.add_argument('--pretrain_epochs', type=int, default=30,
                        help='number of training epochs of CSupCon (default: 30)')
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
                        help='number of layers in graphcnn (INCLUDING the input layer)')
    parser.add_argument('--hidden_dim', type=int, default=64,
                        help='dimensionality of hidden units at ALL layers')
    parser.add_argument('--num_head', type=int, default=8,
                        help='the number of heads in GAT layer (default: 8)')
    
    # parameter in supervised contrastive loss
    parser.add_argument('--tau', type=float, default=0.07,
                        help='temperature for loss function')

    opt = parser.parse_args()

    return opt


def main():
    opt = parse_option()
    setup_seed(opt.seed)
    device = torch.device("cuda:" + str(opt.device)) if torch.cuda.is_available() else torch.device("cpu")

    #-----------------------------------------------prepare dataset----------------------------------------------
    # Load data
    adj, feats, train_idx, val_idx, test_idx, labels, nb_nodes, ss_label = load_data(opt.dataset, opt.seed, 1, opt.train_nodes_per_class)

    opt.input_dim = feats.shape[1]
    opt.n_cls = len(set(labels[:, 1]))

    # organize nodes per class
    nodes_per_cls = {}
    for n in range(opt.n_cls):
        nodes_per_cls[n] = labels[train_idx, 0][np.where(labels[train_idx, 1] == n)[0]].astype('int')

    # preprocess adjacency and initial features
    adj = adj.to(device)
    feats = torch.FloatTensor(feats).to(device)

    #----------------------------------------prepare model/loss/optimizer----------------------------------------
    model = SupConGraphCNN(opt.num_graphcnn_layers, opt.input_dim, opt.hidden_dim, opt.num_head, head='mlp').to(device)
    classifier = torch.nn.Linear(opt.hidden_dim, opt.n_cls).to(device)
    criterion = SupConLoss(device, temperature=opt.tau).to(device)

    optimizer = torch.optim.Adam([{'params': model.parameters()},
                                 {'params': criterion.parameters()}],
                                lr = opt.learning_rate,
                                weight_decay= opt.weight_decay)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=50, gamma=0.5)

    optimizer2 = torch.optim.Adam([{'params': model.parameters()},
                                  {'params': classifier.parameters()}],
                                 lr = opt.learning_rate,
                                 weight_decay= opt.weight_decay)
    scheduler2 = torch.optim.lr_scheduler.StepLR(optimizer2, step_size=50, gamma=0.5)
    
    
    #-----------------------------------------------start training-----------------------------------------------
    # training via SupCon at the first stage
    if len(train_idx) % opt.batch_size == 0:
        n_batch = len(train_idx) // opt.batch_size
    else:
        n_batch = len(train_idx) // opt.batch_size + 1

    for epoch in range(1, opt.pretrain_epochs+1):
        random.shuffle(train_idx)
        for n in range(n_batch):
            start = n * opt.batch_size
            end = (n + 1) * opt.batch_size
            node_id = train_idx[start: end]
            node_id2 = []
            for node in node_id:
                cls_id = labels[np.where(labels[:, 0]==node)[0], 1][0]
                no_self = list(nodes_per_cls[cls_id])
                no_self.remove(node)
                node_id2.extend([random.choice(no_self)])

            loss_supcon = train(feats, adj, node_id, node_id2, labels, nodes_per_cls, model, criterion, opt, device)
            
            if optimizer is not None:
                optimizer.zero_grad()
                loss_supcon.backward()
                optimizer.step()
                scheduler.step()

    # training via CE at the second stage
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
        if optimizer2 is not None:
            optimizer2.zero_grad()
            loss.backward()
            optimizer2.step()
            scheduler2.step()

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
    fp=open("save/res_supcon_{}_{}.txt".format(opt.dataset, opt.train_nodes_per_class), "a")
    fp.write("batch_size: {}, learning rate: {}, pretrain_epochs: {}, best val acc: {:.3f}, testing acc: {:.3f}\n".format(opt.batch_size, opt.learning_rate, opt.pretrain_epochs, best, best_test))
    fp.close()

    return


def train(feats, adj, node_id, node_id2, labels, nodes_per_cls, model, criterion, opt, device):
    """compute SupCon loss"""
    model.train()
    criterion.train()
    output = model(feats, adj)

    tmp_output = torch.cat([output[node_id], output[node_id2]], dim=1)
    tmp_feature = torch.reshape(torch.unsqueeze(tmp_output, 1), (-1, 2, opt.hidden_dim))
    tmp_label = torch.reshape(torch.FloatTensor(labels[node_id, 1]).to(device), (-1, 1))

    loss = criterion(tmp_feature, tmp_label)

    return loss


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
