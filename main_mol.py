import torch
from torch.utils.data import DataLoader
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.tensorboard import SummaryWriter
import dgl

from tqdm import tqdm
import argparse
import time
import os
import numpy as np

from ogb.graphproppred import DglGraphPropPredDataset, Evaluator

from gnn import GNN_mol
from utils import collate_dgl, add_positional_encoding


cls_criterion = torch.nn.BCEWithLogitsLoss()


def train(model, device, loader, optimizer, cls_criterion):
    model.train()
    avg_loss = 0

    for step, (batch_graphs, batch_labels) in enumerate(tqdm(loader, desc="Iteration")):
        batch_graphs = batch_graphs.to(device)
        batch_labels = batch_labels.to(device)
        batch_h = batch_graphs.ndata['feat'].to(device)
        batch_e = batch_graphs.edata['feat'].to(device)

        pred = model(batch_graphs, batch_h, batch_e)
        
        optimizer.zero_grad()
        
        # Ignore nan targets (unlabeled) when computing loss
        is_labeled = batch_labels == batch_labels
        loss = cls_criterion(pred.to(torch.float32)[is_labeled], batch_labels.to(torch.float32)[is_labeled])
        
        loss.backward()
        optimizer.step()
        
        avg_loss += loss.detach().item()
    
    avg_loss /= (step + 1) 
    return avg_loss


def eval(model, device, loader, evaluator, cls_criterion):
    model.eval()
    avg_loss = 0
    y_true = []
    y_pred = []

    for step, (batch_graphs, batch_labels) in enumerate(tqdm(loader, desc="Iteration")):
        batch_graphs = batch_graphs.to(device)
        batch_labels = batch_labels.to(device)
        batch_h = batch_graphs.ndata['feat'].to(device)
        batch_e = batch_graphs.edata['feat'].to(device)

        with torch.no_grad():
            pred = model(batch_graphs, batch_h, batch_e)
            
            # Ignore nan targets (unlabeled) when computing loss
            is_labeled = batch_labels == batch_labels
            loss = cls_criterion(pred.to(torch.float32)[is_labeled], batch_labels.to(torch.float32)[is_labeled])
            
        avg_loss += loss.item()
        
        y_true.append(batch_labels.view(pred.shape).detach().cpu())
        y_pred.append(pred.detach().cpu())

    y_true = torch.cat(y_true, dim = 0).numpy()
    y_pred = torch.cat(y_pred, dim = 0).numpy()

    input_dict = {"y_true": y_true, "y_pred": y_pred}
    
    avg_loss /= (step + 1)
    return avg_loss, evaluator.eval(input_dict)


def main(args):
    
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    
    device = torch.device("cuda:" + str(args.device)) if torch.cuda.is_available() else torch.device("cpu")
    
    # Load dataset and evaluator
    dataset = DglGraphPropPredDataset(name = args.dataset)
    split_idx = dataset.get_idx_split()
    evaluator = Evaluator(args.dataset)
    
    if args.pos_enc_dim > 0:
        # Add graph positional encodings
        print("Adding PEs...")
        dataset.graphs = [add_positional_encoding(g, args.pos_enc_dim) for g in tqdm(dataset.graphs)]
    
    # Basic pre-processing
    if args.dataset == 'ogbg-molpcba':
        print("Removing training graphs with 0 edges...")
        train_split = []
        for idx, g in enumerate(tqdm(dataset.graphs)):
            if idx in split_idx["train"] and g.number_of_edges() != 0:
                train_split.append(idx)
        split_idx["train"] = torch.LongTensor(train_split)

    # Prepare dataloaders
    train_loader = DataLoader(dataset[split_idx["train"]], batch_size=args.batch_size, shuffle=True, 
                              num_workers = args.num_workers, collate_fn=collate_dgl)
    valid_loader = DataLoader(dataset[split_idx["valid"]], batch_size=args.batch_size, shuffle=False, 
                              num_workers = args.num_workers, collate_fn=collate_dgl)
    test_loader = DataLoader(dataset[split_idx["test"]], batch_size=args.batch_size, shuffle=False, 
                             num_workers = args.num_workers, collate_fn=collate_dgl)
    
    # Initialize model, optimizer and scheduler
    if args.gnn in ['gated-gcn', 'gcn', 'mlp']:
        model = GNN_mol(gnn_type=args.gnn, num_tasks=dataset.num_tasks, num_layer=args.num_layer, 
                        emb_dim=args.emb_dim, dropout=args.dropout, batch_norm=True, 
                        residual=True, pos_enc_dim=args.pos_enc_dim, graph_pooling=args.pooling)
        model.to(device)
        print(model)
        total_param = 0
        for param in model.parameters():
            total_param += np.prod(list(param.data.size()))
        print(f'Total parameters: {total_param}')
        
        optimizer = optim.Adam(model.parameters(), lr=args.lr)
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode='min', factor=args.lr_reduce_factor, 
            patience=args.lr_scheduler_patience, verbose=True
        )
    else:
        raise ValueError('Invalid GNN type')
        
    # Define loss function
    cls_criterion = torch.nn.BCEWithLogitsLoss()
        
    # Create Tensorboard logger
    start_time_str = time.strftime("%Y%m%dT%H%M%S")
    log_dir = os.path.join(
            "logs",
            args.dataset,
            f"{args.expt_name}-{args.gnn}-L{args.num_layer}-h{args.emb_dim}-d{args.dropout}-LR{args.lr}-GPU{args.device}", 
            start_time_str
    )
    tb_logger = SummaryWriter(log_dir)
    
    # Training loop
    train_curve = []
    valid_curve = []
    test_curve = []

    for epoch in range(1, args.epochs + 1):
        print("=====Epoch {}".format(epoch))
        tb_logger.add_scalar('lr', optimizer.param_groups[0]['lr'], epoch)
        
        print('Training...')
        train(model, device, train_loader, optimizer, cls_criterion)

        print('Evaluating...')
        train_loss, train_perf = eval(model, device, train_loader, evaluator, cls_criterion)
        valid_loss, valid_perf = eval(model, device, valid_loader, evaluator, cls_criterion)
        _, test_perf = eval(model, device, test_loader, evaluator, cls_criterion)
        
        # Log statistics to Tensorboard, etc.
        print({'Train': train_perf, 'Validation': valid_perf, 'Test': test_perf})
        
        tb_logger.add_scalar('loss/train', train_loss, epoch)
        tb_logger.add_scalar(f'{dataset.eval_metric}/train', train_perf[dataset.eval_metric], epoch)
        tb_logger.add_scalar('loss/valid', valid_loss, epoch)
        tb_logger.add_scalar(f'{dataset.eval_metric}/valid', valid_perf[dataset.eval_metric], epoch)
        tb_logger.add_scalar(f'{dataset.eval_metric}/test', test_perf[dataset.eval_metric], epoch)
        
        train_curve.append(train_perf[dataset.eval_metric])
        valid_curve.append(valid_perf[dataset.eval_metric])
        test_curve.append(test_perf[dataset.eval_metric])
        
        if args.lr_scheduler_patience > 0:
            # Reduce LR using scheduler
            scheduler.step(valid_loss)

    if 'classification' in dataset.task_type:
        best_val_epoch = np.argmax(np.array(valid_curve))
        best_train = max(train_curve)
    else:
        best_val_epoch = np.argmin(np.array(valid_curve))
        best_train = min(train_curve)

    print('Finished training!')
    print('Best validation score: {}'.format(valid_curve[best_val_epoch]))
    print('Test score: {}'.format(test_curve[best_val_epoch]))

    torch.save({
        'BestEpoch': best_val_epoch,
        'Validation': valid_curve[best_val_epoch], 
        'Test': test_curve[best_val_epoch], 
        'Train': train_curve[best_val_epoch], 
        'BestTrain': best_train
    }, os.path.join(log_dir, "results.pt"))


if __name__ == "__main__":
    # Experiment settings
    parser = argparse.ArgumentParser(description='GNN baselines on ogbgmol* data with DGL')
    parser.add_argument('--dataset', type=str, default="ogbg-molhiv",
                        help='Dataset name (default: ogbg-molhiv)')
    parser.add_argument('--device', type=int, default=0,
                        help='Which gpu to use if any (default: 0)')
    parser.add_argument('--num_workers', type=int, default=0,
                        help='Number of workers (default: 0)')
    parser.add_argument('--expt_name', type=str, default="debug",
                        help='Experiment name to output result')
    parser.add_argument('--seed', type=int, default=7834,
                        help='Random seed')
    # GNN settings
    parser.add_argument('--gnn', type=str, default='gated-gcn',
                        help='GNN (default: gated-gcn)')
    parser.add_argument('--num_layer', type=int, default=5,
                        help='Number of GNN layers (default: 5)')
    parser.add_argument('--emb_dim', type=int, default=128,
                        help='Dimensionality of hidden units in GNNs (default: 128)')
    parser.add_argument('--dropout', type=float, default=0.5,
                        help='Dropout ratio (default: 0.5)')
    parser.add_argument('--pos_enc_dim', type=int, default=-1,
                        help='Positional encoding dimension (-1 to disable)')
    parser.add_argument('--pooling', type=str, default='mean',
                        help='Graph pooling operation (mean/sum/max)')
    # Training and LR settings
    parser.add_argument('--batch_size', type=int, default=256,
                        help='Input batch size for training (default: 256)')
    parser.add_argument('--epochs', type=int, default=50,
                        help='Number of epochs to train (default: 50)')
    parser.add_argument('--lr', type=float, default=1e-3,
                        help='Learning rate (default: 1e-3)')
    parser.add_argument('--lr_reduce_factor', type=float, default=0.5,
                        help='Learning rate scheduler reduce factor')
    parser.add_argument('--lr_scheduler_patience', type=float, default=5,
                        help='Learning rate scheduler patience epochs (-1 to disable scheduler)')
    args = parser.parse_args()
    
    main(args)
