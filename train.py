import math
import os
import torch
import numpy as np
# from test import test_main
from pathlib import Path
from model import Mymodel
from torch.utils.data import DataLoader
from Mydataloader import Dataset
from tqdm import tqdm
import time
from eval import *
from sklearn.metrics import matthews_corrcoef

peptide_type = ['AAP', 'ABP', 'ACP', 'ACVP', 'ADP', 'AEP', 'AFP', 'AHIVP', 'AHP', 'AIP', 'AMRSAP', 'APP', 'ATP',
                'AVP',
                'BBP', 'BIP',
                'CPP', 'DPPIP',
                'QSP', 'SBP', 'THP']


def same_seeds(seed):
    torch.manual_seed(seed)  # 固定随机种子（CPU）
    if torch.cuda.is_available():  # 固定随机种子（GPU)
        torch.cuda.manual_seed(seed)  # 为当前GPU设置
        torch.cuda.manual_seed_all(seed)  # 为所有GPU设置
    np.random.seed(seed)  # 保证后续使用random函数时，产生固定的随机数
    torch.backends.cudnn.benchmark = False  # GPU、网络结构固定，可设置为True
    torch.backends.cudnn.deterministic = True  # 固定网络结构


np.random.seed(101)
# same_seeds(101)

def add_weight_decay(net, l2_value, skip_list=()):
    decay, no_decay = [], []
    for name, param in net.named_parameters():
        
        if not param.requires_grad:
            continue  # frozen weights
        #if len(param.shape) == 1 or name.endswith(".bias") or name in skip_list:
        #    no_decay.append(param)
        if (name.startswith("conv") and name.endswith(".weight")) or (name.startswith("Dense") and name.endswith(".weight")) :
             decay.append(param)
        else:
            no_decay.append(param)
   
    return [{'params': no_decay, 'weight_decay': 0.0}, {'params': decay, 'weight_decay': l2_value}]


def counters(y_train):
    # counting the number of each peptide
    counterx = np.zeros(len(peptide_type) + 1, dtype='int')
    for i in y_train:
        a = np.sum(i)
        a = int(a)
        counterx[a] += 1
    print(counterx)


def train_and_test_method(train, test, para, model_num, model_path, data_size, embedding):
    # Implementation of training method
    Path(model_path).mkdir(exist_ok=True)

    train_dataset = Dataset(train)
    val_dataset = Dataset(test)

    train_dataloader = DataLoader(dataset=train_dataset, batch_size=32, shuffle=True, num_workers=8)
    val_dataloader = DataLoader(dataset=val_dataset, batch_size=2, shuffle=False, num_workers=2)

    '''
    # # data get
    # X_train, y_train = train[0], train[1]
    #
    # print(X_train.shape)
    # print(y_train.shape)
    # assert 0

    # index = np.arange(len(y_train))
    # np.random.shuffle(index)
    # X_train = X_train[index]
    # y_train = y_train[index]
    #
    # counters(y_train)
    #
    # # train
    # length = X_train.shape[1]
    # out_length = y_train.shape[1]
    #
    # t_data = time.localtime(time.time())
    # with open(os.path.join(model_path, 'time.txt'), 'a+') as f:
    #     f.write('data process finished: {}m {}d {}h {}m {}s\n'.format(t_data.tm_mon, t_data.tm_mday, t_data.tm_hour,
    #                                                                   t_data.tm_min, t_data.tm_sec))

    
    '''
    class_weights = []
    sumx = len(train[0])
    for m in range(len(data_size)):
        g = 5 * math.pow(int((math.log((sumx / data_size[m]), 2))), 2)
        if g <= 0:
            g = 1
        x = {m: g}
        class_weights.append(g)
    
    class Focalloss(torch.nn.Module):
        def __init__(self, gamma=1, alpha=0.60, reduction="mean"):
            super(Focalloss, self).__init__()
            self.gamma = gamma
            self.alpha = alpha
            self.reduction = reduction

        def forward(self, predict, target):
            # loss = -self.alpha * (1 - predict)**self.gamma * target * torch.log(predict) - (1- self.alpha) * predict**self.gamma*(1-target)*torch.log(1 - predict)
            zeros = torch.zeros_like(predict)
            pos_p_sub = torch.where(target > zeros, target - predict, zeros)
            neg_p_sub = torch.where(target > zeros, zeros, predict)
            per_entry_cross_ent = -self.alpha * (pos_p_sub ** self.gamma) * torch.log(
                torch.clamp(predict, 1e-8, 1.0)) - (1 - self.alpha) * (neg_p_sub ** self.gamma) * torch.log(
                torch.clamp(1.0 - predict, 1e-8, 1.0))
            return per_entry_cross_ent.sum()

    epoch_num = 60
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    for counter in range(1, model_num + 1):

        model = Mymodel(
            vocab_size=21,  # 词汇总量
            Embed_D=51,  # 词嵌入维度
            pool_size=5,  #
            dropout=0.6,
            output_length=21,
            convDim=64,
            hidden_size=100,
            Attention_Dim=80,
            Max_length=50,
            N_head=5,
            embedding=embedding
        )

        model.to(device)

        
        # loss_fun = torch.nn.BCELoss(size_average=False, reduce=False)
        loss_fun = Focalloss(gamma=1, alpha=0.60)
        # params = add_weight_decay(model, l2_value=0.002)
        optimizer = torch.optim.Adam(model.parameters(), lr = 0.001)
        scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.95)

        best_val_acc = 0.0
        for epoch in range(1, epoch_num+ 1):
            model.train()

            with tqdm(train_dataloader, total= len(train_dataloader),desc="train: Epoch {}/{}".format(epoch,epoch_num+1),ncols=150, miniters=0.1 ) as bar:
                right_sample = 0
                all_sample = 0
                for (train, label) in bar:

                    output = model(train.to(device))
                    lossvalue = loss_fun(output, label.to(device))
                    # cw = torch.from_numpy(np.array(class_weights)).to(device)
                    # losss = torch.mean(torch.sum(lossvalue * cw, dim=-1))
                    optimizer.zero_grad()
                    # losss.backward()
                    lossvalue.backward()
                    optimizer.step()
                    predict_result = output.detach().cpu().numpy()
                    predict_bool = (predict_result >= 0.5).astype(np.float32)
                    label_true = label.numpy()
                    ss = np.sum(np.abs(predict_bool - label_true),axis=1)
                    right_sample += np.sum((ss == 0).astype(np.float32))
                    all_sample += label.shape[0]
                    bar.set_postfix(loss = lossvalue.item(), train_accuracy = right_sample/(0.0 + all_sample) * 100)



            list_pred = []
            list_true = []
            # test val dataset
            model.eval()
            with torch.no_grad():
                if epoch // 10 == 0:
                    scheduler.step()
                right_number_count = 0
                all_number_count = 0
                with tqdm(val_dataloader, total=len(val_dataloader),desc="val test", ncols=150, miniters=0.3) as vbar:
                    for val, Tlabel in vbar:
                        val_output = model(val.to(device))
                        predict_result = val_output.cpu().numpy()
                        predict_bool = (predict_result >= 0.5).astype(np.float32)
                        label_true = Tlabel.numpy()
                        ss = np.sum(np.abs(predict_bool - label_true), axis=1)
                        all_number_count += ss.shape[0]
                        right_number_count += np.sum((ss == 0).astype(np.float32))
                        vbar.set_postfix(val_dataset_accuracy = round(100 * right_number_count / (all_number_count + 0.0) ,2))

                        list_pred.append(predict_bool)
                        list_true.append(label_true)

                    list_pred = np.concatenate(list_pred, axis=0)
                    list_true = np.concatenate(list_true, axis=0)
                if right_number_count / (all_number_count + 0.0) > best_val_acc:
                    best_val_acc = right_number_count / (all_number_count + 0.0)
                    torch.save(model.state_dict(),"./model/model_num_{}_save_{}.pt".format(counter, round(100 * right_number_count / (all_number_count + 0.0),2)))
                    print("save model, the accuracy is {}".format(round(100 * right_number_count / (all_number_count + 0.0),2)))
            Precision, Recall, Accuracy, absolute_true, absolute_false = evaluate(list_pred, list_true)
            mcc = matthews_corrcoef(list_pred.argmax(axis=1), list_true.argmax(axis=1))
            print(
                "current prediction case:\n Precision is {:.2f}\n Recall is {:.2f}\n Accuracy is {:.2f}\n absolute_true is {:.2f}\n absolute_false is {:.2f}\n mcc is {:.2f}".format(
                    Precision * 100, Recall * 100, Accuracy * 100, absolute_true * 100, absolute_false * 100, mcc * 100)
            )

def train_main(train, test, model_num, modelDir, data_size, embedding):
    # parameters
    ed = 128
    ps = 5
    fd = 128
    dp = 0.6
    lr = 0.001
    para = {'embedding_dimension': ed, 'pool_size': ps, 'fully_dimension': fd,
            'drop_out': dp, 'learning_rate': lr}
    # Conduct training
    train_and_test_method(train,test,  para, model_num, modelDir, data_size, embedding)

