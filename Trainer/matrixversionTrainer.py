import numpy as np
import torch
from torchvision.utils import make_grid
from BaseClass.BaseTrainer import BaseTrainer
from torch.utils.tensorboard import SummaryWriter
from utils.tools import EarlyStopping
import os
from tqdm import tqdm
from utils.loss import nll_loss, dynamic_KL_w
from utils.metric import c_index
import torch.nn as nn
import pandas as pd
from torchmetrics.classification import Accuracy, AUROC, F1Score
import gc

class MatrixTrainer_v2(BaseTrainer):
    """
    Trainer class
    """
    def __init__(self, model, optimizer, config, device, fold,
                 data_loader, valid_data_loader=None, test_data_loader=None, num_class=2, lr_scheduler = None):
        super().__init__(model, optimizer, config)
        self.config = config
        self.device = device
        self.model = model
        self.data_loader = data_loader
        self.len_epoch = len(self.data_loader)
        self.fold = fold
        self.valid_data_loader = valid_data_loader
        self.test_data_loader = test_data_loader
        self.do_validation = self.valid_data_loader is not None
        self.log_step = int(np.sqrt(data_loader.batch_size))
        self.train_stage = None
        self.writer = SummaryWriter(config['trainer']['save_dir'] + '/{}'.format(fold), flush_secs=15)  
        self.epochs = config['trainer']['epochs']
        
        self.gc = 32
        self.num_class = num_class
        self.lr_scheduler = lr_scheduler
        
        self.model_path = os.path.join(config['trainer']['save_dir'], str(fold), 'model.pt')

        self.cls_w =  config['trainer']['cls_w']
        self.surv_w =  config['trainer']['surv_w']

    def _train_epoch(self, epoch, early_stopping):
        """
        Training logic for an epoch

        :param epoch: Integer, current training epoch.
        :return: A log that contains average loss and metric in this epoch.
        """
        self.model.train()
    
        all_risk_scores = np.zeros((len(self.data_loader)))
        all_censorships = np.zeros((len(self.data_loader)))
        all_event_times = np.zeros((len(self.data_loader)))

        MSE_loss = nn.MSELoss()
        CE_loss = nn.CrossEntropyLoss()

        train_loss, cls_loss_log, surv_loss_log = 0., 0., 0.
        train_pred = []
        train_target = []

        for batch_idx, (path_features, case_gene, Y_cls, Y_surv, event_time, c, case_id, genes) in enumerate(tqdm(self.data_loader)):

            if batch_idx == 491:
                print('yes')

            path_features = path_features.to(self.device)
            case_gene = case_gene.type(torch.FloatTensor).to(self.device)
            Y_surv = Y_surv.type(torch.LongTensor).to(self.device)
            Y_cls = Y_cls.type(torch.LongTensor).to(self.device)
            c = c.type(torch.FloatTensor).to(self.device)

            cls_logits, surv_logits, S, hazards, hazards_cls, S_cls = self.model(n = self.config['trainer']['lastlayer'], mode = self.config['trainer']['mode'], wsi_data = path_features, gene_data = case_gene)
            train_pred.append(cls_logits)
            train_target.append(Y_cls)

            cls_loss = nll_loss(hazards=hazards_cls, S=S_cls, Y=Y_surv, c=c)
            surv_loss = nll_loss(hazards=hazards, S=S, Y=Y_surv, c=c)
            loss = (cls_loss + surv_loss) / self.gc

            cls_loss_log += cls_loss.item()
            surv_loss_log += surv_loss.item()
            train_loss += loss.item()
            loss.backward() 
            if (batch_idx + 1) % self.gc == 0: 
                self.optimizer.step()
                self.optimizer.zero_grad()

            risk = -torch.sum(S, dim=1).detach().cpu().numpy()
            all_risk_scores[batch_idx] = risk
            all_censorships[batch_idx] = c.item()
            all_event_times[batch_idx] = event_time

        cindex = c_index(all_censorships, all_event_times, all_risk_scores)

        train_loss /= len(self.data_loader)

        self.writer.add_scalar('train/loss', train_loss, epoch)
        self.writer.add_scalar('train/cls_loss', cls_loss_log, epoch)
        self.writer.add_scalar('train/surv_loss', surv_loss_log, epoch)

        print("\n\n\n====\033[1;32mTraining\033[0m Statistics====")
        print('\033[1;34mTrain Loss\033[0m: \033[1;31m{:.4f}\033[0m'.format(train_loss))
        print('\033[1;34mC-index\033[0m: \033[1;31m{:.4f}\033[0m'.format(cindex))

        val_log = self._valid_epoch(epoch, early_stopping)

    def _valid_epoch(self, epoch, early_stopping):
        self.model.eval()
    
        all_risk_scores = np.zeros((len(self.valid_data_loader)))
        all_censorships = np.zeros((len(self.valid_data_loader)))
        all_event_times = np.zeros((len(self.valid_data_loader)))

        CE_loss = nn.CrossEntropyLoss()

        train_loss, cls_loss_log, surv_loss_log = 0., 0., 0.
        train_pred = []
        train_target = []

        for batch_idx, (path_features, case_gene, Y_cls, Y_surv, event_time, c, case_id, genes) in enumerate(tqdm(self.valid_data_loader)):
        # for batch_idx, (path_features, case_gene, Y_cls, Y_surv, event_time, c, case_id) in enumerate(tqdm(self.test_data_loader)):
            path_features = path_features.to(self.device)
            case_gene = case_gene.type(torch.FloatTensor).to(self.device)
            Y_surv = Y_surv.type(torch.LongTensor).to(self.device)
            Y_cls = Y_cls.type(torch.LongTensor).to(self.device)
            c = c.type(torch.FloatTensor).to(self.device)

            with torch.no_grad():
                cls_logits, surv_logits, S, hazards, hazards_cls, S_cls = self.model(n = self.config['trainer']['lastlayer'], mode = self.config['trainer']['mode'], wsi_data = path_features, gene_data = case_gene)

            cls_loss = CE_loss(cls_logits, Y_cls)
            surv_loss = nll_loss(hazards=hazards, S=S, Y=Y_surv, c=c)

            loss = (cls_loss + surv_loss) / self.gc

            cls_loss_log += cls_loss.item()
            surv_loss_log += surv_loss.item()
            train_loss += loss.item()


            risk = -torch.sum(S, dim=1).detach().cpu().numpy()
            all_risk_scores[batch_idx] = risk
            all_censorships[batch_idx] = c.item()
            all_event_times[batch_idx] = event_time

        cindex = c_index(all_censorships, all_event_times, all_risk_scores)

        train_loss /= len(self.data_loader)

        self.writer.add_scalar('val/loss', train_loss, epoch)
        self.writer.add_scalar('val/cls_loss', cls_loss_log, epoch)
        self.writer.add_scalar('val/surv_loss', surv_loss_log, epoch)

        print("\n\n\n====\033[1;32mValid\033[0m Statistics====")
        print('\033[1;34mValid Loss\033[0m: \033[1;31m{:.4f}\033[0m'.format(train_loss))
        print('\033[1;34mC-index\033[0m: \033[1;31m{:.4f}\033[0m'.format(cindex))

        metric = cindex
            
        early_stopping(epoch=epoch, metric=metric, models=self.model, ckpt_name=os.path.join(self.config['trainer']['save_dir'], str(self.fold)))   # 在验证阶段会累积score不改变的次数

    def test(self, fold):
        self.model.eval()
    
        all_risk_scores = np.zeros((len(self.test_data_loader)))
        all_censorships = np.zeros((len(self.test_data_loader)))
        all_event_times = np.zeros((len(self.test_data_loader)))

        train_pred = []
        train_target = []

        for batch_idx, (path_features, case_gene, Y_cls, Y_surv, event_time, c, case_id, genes) in enumerate(tqdm(self.test_data_loader)):
            path_features = path_features.to(self.device)
            case_gene = case_gene.type(torch.FloatTensor).to(self.device)
            Y_surv = Y_surv.type(torch.LongTensor).to(self.device)
            Y_cls = Y_cls.type(torch.LongTensor).to(self.device)
            c = c.type(torch.FloatTensor).to(self.device)

            with torch.no_grad():
                cls_logits, surv_logits, S, hazards, hazards_cls, S_cls = self.model(n = self.config['trainer']['lastlayer'], mode = self.config['trainer']['mode'], wsi_data = path_features, gene_data = case_gene)
            train_pred.append(cls_logits)
            train_target.append(Y_cls)

            risk = -torch.sum(S, dim=1).detach().cpu().numpy()
            all_risk_scores[batch_idx] = risk
            all_censorships[batch_idx] = c.item()
            all_event_times[batch_idx] = event_time

        cindex = c_index(all_censorships, all_event_times, all_risk_scores)

        if not os.path.exists(os.path.join(self.config['trainer']['save_dir'], str(fold), 'all_risk_scores.npz') ):
            np.savez(os.path.join(self.config['trainer']['save_dir'], str(fold), 'all_risk_scores.npz'), all_risk_scores)
            np.savez(os.path.join(self.config['trainer']['save_dir'], str(fold), 'all_censorships.npz'), all_censorships)
            np.savez(os.path.join(self.config['trainer']['save_dir'], str(fold), 'all_event_times.npz'), all_event_times)
        
        print("\n\n\n====\033[1;32mTesting\033[0m Statistics====")
        print('\033[1;34mC-index\033[0m: \033[1;31m{:.4f}\033[0m'.format(cindex))
        
        cindex = {'ci': cindex,}
        acc = {'acc': 0.0}
        return cindex, acc

    def test_valid(self, fold):
        self.model.eval()
        num = 200
    
        all_risk_scores = np.zeros(num)
        all_censorships = np.zeros(num)
        all_event_times = np.zeros(num)

        train_pred = []
        train_target = []

        for batch_idx, (path_features, case_gene, Y_cls, Y_surv, event_time, c, case_id) in enumerate(tqdm(self.valid_data_loader)):
            path_features = path_features.to(self.device)
            case_gene = case_gene.type(torch.FloatTensor).to(self.device)
            Y_surv = Y_surv.type(torch.LongTensor).to(self.device)
            Y_cls = Y_cls.type(torch.LongTensor).to(self.device)
            c = c.type(torch.FloatTensor).to(self.device)

            with torch.no_grad():
                cls_logits, surv_logits, S, hazards = self.model(n = self.config['trainer']['lastlayer'], mode = self.config['trainer']['mode'], wsi_data = path_features, gene_data = case_gene)
            train_pred.append(cls_logits)
            train_target.append(Y_cls)

            risk = -torch.sum(S, dim=1).detach().cpu().numpy()
            all_risk_scores[batch_idx] = risk
            all_censorships[batch_idx] = c.item()
            all_event_times[batch_idx] = event_time

            if batch_idx == num:
                break
        cindex = c_index(all_censorships, all_event_times, all_risk_scores)

        train_pred = torch.cat(train_pred, dim=0)
        train_target = torch.cat(train_target, dim=0)
        accuracy_metric = Accuracy(task="multiclass",num_classes=self.num_class).to(self.device)
        auc_metric = AUROC(task="multiclass", num_classes=self.num_class).to(self.device)
        F1_metric = F1Score(task="multiclass", num_classes=self.num_class).to(self.device)

        acc = accuracy_metric(train_pred, train_target)
        auc = auc_metric(train_pred, train_target)
        f1_score = F1_metric(train_pred, train_target)

        if not os.path.exists(os.path.join(self.config['trainer']['save_dir'], str(fold), 'all_risk_scores_valid.npz') ):
            np.savez(os.path.join(self.config['trainer']['save_dir'], str(fold), 'all_risk_scores_valid.npz'), all_risk_scores)
            np.savez(os.path.join(self.config['trainer']['save_dir'], str(fold), 'all_censorships_valid.npz'), all_censorships)
            np.savez(os.path.join(self.config['trainer']['save_dir'], str(fold), 'all_event_times_valid.npz'), all_event_times)



        print("\n\n\n====\033[1;32mTesting\033[0m Statistics====")
        print('\033[1;34mC-index\033[0m: \033[1;31m{:.4f}\033[0m'.format(cindex))
        print('\033[1;34mTest ACC\033[0m: \033[1;31m{:.4f}\033[0m'.format(acc))
        print('\033[1;34mTest AUC\033[0m: \033[1;31m{:.4f}\033[0m'.format(auc))
        print('\033[1;34mTest f1\033[0m: \033[1;31m{:.4f}\033[0m'.format(f1_score))

        cindex = {'ci': cindex,}
        acc = {'acc': acc.cpu().numpy(), 'auc': auc.cpu().numpy(), 'f1score': f1_score.cpu().numpy()}
        return cindex, acc

    def train(self, fold=0):
        early_stopping = EarlyStopping(warmup=2, patience=5, stop_epoch=18, verbose = True, logger=self.logger)
        if self.config['trainer']['test_phase']:
            self.model.load_state_dict(torch.load(self.model_path))
            cindex, cls_acc = self.test(fold)
            # cindex, cls_acc = self.test_valid(fold)
            
            result_file = os.path.join(self.config['trainer']['save_dir'], 'result.csv')
            summary_file = os.path.join(self.config['trainer']['save_dir'], 'summary.csv')
            result = {**cindex, **cls_acc}
            df = pd.DataFrame.from_dict(result, orient='index').T
            if fold == 0:
                df.to_csv(result_file, mode='a', header=True, index=False)  
            else:
                df.to_csv(result_file, mode='a', header=False, index=False)

            if fold == 3:  
                df = pd.read_csv(result_file)
                result = {'ci_avg': df['ci'].mean(), 'ci_std': df['ci'].std(), 'acc_avg': df['acc'].mean(), 'acc_std': df['acc'].std(), 'auc_avg': df['auc'].mean(), 'auc_std': df['auc'].std(), 'f1_avg': df['f1score'].mean(), 'f1_std': df['f1score'].std()}
                df = pd.DataFrame.from_dict(result, orient='index').T
                df.to_csv(summary_file, index=False)
                
        else:
            for epoch in range(0, self.epochs):
                print(f'Epoch : {epoch}:')

                self._train_epoch(epoch, early_stopping)

                if self.config['trainer']['lr_scheduler']:
                    self.lr_scheduler.step()
                if early_stopping.early_stop == True:   
                    self.logger.info('fold {}: Training stop at epoch {}'.format(str(fold), epoch))
                    break

                gc.collect()
                torch.cuda.empty_cache()
                
            if early_stopping.early_stop == False:
                self.logger.info('fold {}: Training stop at epoch {}'.format(str(fold), self.epochs-1))