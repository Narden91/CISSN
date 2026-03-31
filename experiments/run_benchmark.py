
import os
import sys
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import argparse
import time
from cissn.models.encoder import DisentangledStateEncoder
from cissn.models.forecast_head import ForecastHead
from cissn.losses.disentangle_loss import DisentanglementLoss
from cissn.data.data_loader import get_data_loader
from sklearn.metrics import mean_squared_error, mean_absolute_error

class Experiment:
    def __init__(self, args):
        self.args = args
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model = self._build_model().to(self.device)
        self.head = self._build_head().to(self.device)
        print(f"Model and Head initialized on {self.device}")

    def _build_model(self):
        return DisentangledStateEncoder(
            input_dim=self.args.enc_in,
            state_dim=self.args.state_dim,
            hidden_dim=self.args.d_model,
            dropout=self.args.dropout
        )

    def _build_head(self):
        return ForecastHead(
            state_dim=self.args.state_dim,
            output_dim=self.args.c_out,
            horizon=self.args.pred_len,
            hidden_dim=self.args.d_model // 2
        )

    def _get_data(self, flag):
        data_set, data_loader = get_data_loader(self.args, flag)
        return data_set, data_loader

    def _select_optimizer(self):
        params = list(self.model.parameters()) + list(self.head.parameters())
        return optim.Adam(params, lr=self.args.learning_rate)

    def _select_criterion(self):
        return nn.MSELoss()

    @staticmethod
    def _concatenate_batches(batches, name):
        if not batches:
            raise RuntimeError(f"No {name} batches were produced.")
        return np.concatenate(batches, axis=0)

    def train(self, setting):
        train_data, train_loader = self._get_data(flag='train')
        vali_data, vali_loader = self._get_data(flag='val')
        test_data, test_loader = self._get_data(flag='test')

        path = os.path.join(self.args.checkpoints, setting)
        if not os.path.exists(path):
            os.makedirs(path)

        time_now = time.time()
        train_steps = len(train_loader)
        early_stopping = EarlyStopping(patience=self.args.patience, verbose=True)

        model_optim = self._select_optimizer()
        criterion = self._select_criterion()
        disentangle_criterion = DisentanglementLoss(
            lambda_cov=self.args.lambda_cov,
            lambda_temporal=self.args.lambda_temp
        )

        for epoch in range(self.args.train_epochs):
            iter_count = 0
            train_loss = []
            
            self.model.train()
            self.head.train()
            epoch_time = time.time()
            
            for i, (batch_x, batch_y, batch_x_mark, batch_y_mark) in enumerate(train_loader):
                iter_count += 1
                model_optim.zero_grad()
                
                batch_x = batch_x.float().to(self.device)
                batch_y = batch_y.float().to(self.device)
                
                # Encoder Forward
                # We need all states for disentanglement loss
                states = self.model(batch_x, return_all_states=True) # (B, L, State)
                final_state = states[:, -1, :] # (B, State)
                
                # Forecast
                outputs = self.head(final_state) # (B, Pred, Out)
                
                # Slicing for loss
                # batch_y shape: (B, Label+Pred, Out)
                # We only want the last Pred_len steps
                f_dim = -1 if self.args.features == 'MS' else 0
                outputs = outputs[:, -self.args.pred_len:, f_dim:]
                batch_y = batch_y[:, -self.args.pred_len:, f_dim:].to(self.device)
                
                # Losses
                pred_loss = criterion(outputs, batch_y)
                dis_loss = disentangle_criterion(states)
                
                loss = pred_loss + dis_loss
                train_loss.append(loss.item())
                
                loss.backward()
                model_optim.step()
                
                if (i + 1) % 100 == 0:
                    print(f"\titers: {i+1}, epoch: {epoch+1} | loss: {loss.item():.7f}")
                    speed = (time.time() - time_now) / iter_count
                    left_time = speed * ((self.args.train_epochs - epoch) * train_steps - i)
                    print(f'\tspeed: {speed:.4f}s/iter; left time: {left_time:.4f}s')
                    iter_count = 0
                    time_now = time.time()
            
            print(f"Epoch: {epoch+1} cost time: {time.time()-epoch_time:.2f}")
            train_loss = np.average(train_loss)
            vali_loss = self.vali(vali_loader, criterion)
            test_loss = self.vali(test_loader, criterion)

            print(f"Epoch: {epoch+1}, Steps: {train_steps} | Train Loss: {train_loss:.7f} Vali Loss: {vali_loss:.7f} Test Loss: {test_loss:.7f}")
            
            if self.args.use_wandb:
                import wandb
                wandb.log({
                    "epoch": epoch + 1,
                    "train_loss": train_loss,
                    "vali_loss": vali_loss,
                    "test_loss": test_loss,
                    "lr": model_optim.param_groups[0]['lr']
                })
            
            early_stopping(vali_loss, self.model, self.head, path)
            if early_stopping.early_stop:
                print("Early stopping")
                break
            
            adjust_learning_rate(model_optim, epoch + 1, self.args)
            
        return self.model

    def vali(self, vali_loader, criterion):
        total_loss = 0.0
        total_weight = 0
        self.model.eval()
        self.head.eval()
        with torch.no_grad():
            for i, (batch_x, batch_y, _batch_x_mark, _batch_y_mark) in enumerate(vali_loader):
                batch_x = batch_x.float().to(self.device)
                batch_y = batch_y.float().to(self.device)
                
                final_state = self.model(batch_x)
                outputs = self.head(final_state)
                
                f_dim = -1 if self.args.features == 'MS' else 0
                outputs = outputs[:, -self.args.pred_len:, f_dim:]
                batch_y = batch_y[:, -self.args.pred_len:, f_dim:].to(self.device)
                
                loss = criterion(outputs, batch_y)
                batch_weight = outputs.numel()
                total_loss += loss.item() * batch_weight
                total_weight += batch_weight
            if total_weight == 0:
                raise RuntimeError("Validation loader produced no prediction elements.")
        
            total_loss = total_loss / total_weight
        self.model.train()
        self.head.train()
        return total_loss

    def test(self, setting):
        test_data, test_loader = self._get_data(flag='test')
        
        # Load best model
        path = os.path.join(self.args.checkpoints, setting)
        self.model.load_state_dict(torch.load(os.path.join(path, 'checkpoint.pth'), map_location=self.device))
        self.head.load_state_dict(torch.load(os.path.join(path, 'checkpoint_head.pth'), map_location=self.device))
        
        preds = []
        trues = []
        
        self.model.eval()
        self.head.eval()
        
        with torch.no_grad():
            for i, (batch_x, batch_y, _batch_x_mark, _batch_y_mark) in enumerate(test_loader):
                batch_x = batch_x.float().to(self.device)
                batch_y = batch_y.float().to(self.device)
                
                final_state = self.model(batch_x)
                outputs = self.head(final_state)
                
                f_dim = -1 if self.args.features == 'MS' else 0
                outputs = outputs[:, -self.args.pred_len:, f_dim:]
                batch_y = batch_y[:, -self.args.pred_len:, f_dim:].to(self.device)
                
                preds.append(outputs.detach().cpu().numpy())
                trues.append(batch_y.detach().cpu().numpy())
        
            preds = self._concatenate_batches(preds, 'prediction')
            trues = self._concatenate_batches(trues, 'target')
        
        # Metrics
        mae = mean_absolute_error(trues.flatten(), preds.flatten())
        mse = mean_squared_error(trues.flatten(), preds.flatten())
        
        print(f'mse:{mse}, mae:{mae}')
        
        if self.args.use_wandb:
            import wandb
            wandb.log({"test_mse": mse, "test_mae": mae})
        
        # Save results
        folder_path = './results/' + setting + '/'
        if not os.path.exists(folder_path):
            os.makedirs(folder_path)
        
        np.save(folder_path + 'metrics.npy', np.array([mae, mse, 0, 0])) # dummy rmse/mape
        np.save(folder_path + 'pred.npy', preds)
        np.save(folder_path + 'true.npy', trues)
        
        return

class EarlyStopping:
    def __init__(self, patience=7, verbose=False, delta=0):
        self.patience = patience
        self.verbose = verbose
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.val_loss_min = np.inf
        self.delta = delta

    def __call__(self, val_loss, model, head, path):
        score = -val_loss
        if self.best_score is None:
            self.best_score = score
            self.save_checkpoint(val_loss, model, head, path)
        elif score < self.best_score + self.delta:
            self.counter += 1
            print(f'EarlyStopping counter: {self.counter} out of {self.patience}')
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = score
            self.save_checkpoint(val_loss, model, head, path)
            self.counter = 0

    def save_checkpoint(self, val_loss, model, head, path):
        if self.verbose:
            print(f'Validation loss decreased ({self.val_loss_min:.6f} --> {val_loss:.6f}).  Saving model ...')
        torch.save(model.state_dict(), os.path.join(path, 'checkpoint.pth'))
        # Save head state too - simpler to just save model for now or separate file
        torch.save(head.state_dict(), os.path.join(path, 'checkpoint_head.pth'))
        self.val_loss_min = val_loss

def adjust_learning_rate(optimizer, epoch, args):
    # type: (optim.Optimizer, int, argparse.Namespace) -> None
    # Decay learning rate by 0.5 every 1 epoch (aggressive) or 0.1 every 2 epochs
    lr_adjust = {}
    if args.lradj == 'type1':
        lr_adjust = {epoch: args.learning_rate * (0.5 ** ((epoch - 1) // 1))}
    elif args.lradj == 'type2':
        lr_adjust = {
            2: 5e-5, 4: 1e-5, 6: 5e-6, 8: 1e-6,
            10: 5e-7, 15: 1e-7, 20: 5e-8
        }
    else:
        raise ValueError(f"Unknown lradj policy: {args.lradj!r}. Use 'type1' or 'type2'.")

    if epoch in lr_adjust.keys():
        lr = lr_adjust[epoch]
        for param_group in optimizer.param_groups:
            param_group['lr'] = lr
        print('Updating learning rate to {}'.format(lr))

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='CISSN Benchmark Runner')

    # Basic Config
    parser.add_argument('--data', type=str, required=True, default='ETTh1', help='data')
    parser.add_argument('--root_path', type=str, default='./data/ETT/', help='root path of the data file')
    parser.add_argument('--data_path', type=str, default='ETTh1.csv', help='data file')    
    parser.add_argument('--features', type=str, default='M', help='forecasting task, options:[M, S, MS]')
    parser.add_argument('--target', type=str, default='OT', help='target feature in S or MS task')
    parser.add_argument('--freq', type=str, default='h', help='freq for time features encoding')
    parser.add_argument('--checkpoints', type=str, default='./checkpoints/', help='location of model checkpoints')

    # Model Params
    parser.add_argument('--seq_len', type=int, default=96, help='input sequence length')
    parser.add_argument('--label_len', type=int, default=48, help='start token length')
    parser.add_argument('--pred_len', type=int, default=96, help='prediction sequence length')
    
    # CISSN Specific
    parser.add_argument('--enc_in', type=int, default=7, help='encoder input size')
    parser.add_argument('--c_out', type=int, default=7, help='output size')
    parser.add_argument('--d_model', type=int, default=64, help='dimension of model')
    parser.add_argument('--state_dim', type=int, default=5, help='dimension of latent state')
    parser.add_argument('--dropout', type=float, default=0.05, help='dropout')
    parser.add_argument('--lambda_cov', type=float, default=1.0, help='covariance loss weight')
    parser.add_argument('--lambda_temp', type=float, default=0.5, help='temporal consistency loss weight')

    # Optimization
    parser.add_argument('--num_workers', type=int, default=0, help='data loader num workers')
    parser.add_argument('--train_epochs', type=int, default=10, help='train epochs')
    parser.add_argument('--batch_size', type=int, default=32, help='batch size of train input data')
    parser.add_argument('--patience', type=int, default=3, help='early stopping patience')
    parser.add_argument('--learning_rate', type=float, default=0.001, help='optimizer learning rate')
    parser.add_argument('--lradj', type=str, default='type1', help='adjust learning rate policy')

    # Logging
    parser.add_argument('--use_wandb', action='store_true', help='use wandb for logging')
    parser.add_argument('--project_name', type=str, default='CISSN_Benchmark', help='wandb project name')

    args = parser.parse_args()

    print('Args in experiment:')
    print(args)

    setting = 'CISSN_{}_{}_sl{}_pl{}_sd{}'.format(
        args.data,
        args.features,
        args.seq_len,
        args.pred_len,
        args.state_dim
    )
    
    if args.use_wandb:
        import wandb
        wandb.init(project=args.project_name, config=args, name=setting)

    exp = Experiment(args) 
    
    print('>>>>>>>start training : {}>>>>>>>>>>>>>>>>>>>>>>>>>>'.format(setting))
    exp.train(setting)
    
    print('>>>>>>>testing : {}<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<'.format(setting))
    exp.test(setting)
    
    if args.use_wandb:
        wandb.finish()
