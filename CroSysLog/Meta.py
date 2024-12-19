from Learner import Learner
import torch
from torch import optim, nn
from torch.nn import functional as F
import numpy as np
from copy import deepcopy


class MAML(nn.Module):
    """
        Meta Learner
        """

    def __init__(self, config, learner_config):
        '''
        N: num of classes
        K: num of instances for each class in the support set
        '''
        super(MAML, self).__init__()

        self.update_lr = config["update_lr"]
        self.meta_lr = config["meta_lr"]
        self.k_spt = config["k_spt"]
        self.k_qry = config["k_qry"]
        self.task_num = config["task_num"]
        self.update_step = config["update_step"]
        self.update_step_test = config["update_step_test"]

        self.window_size = config["window_size"]


        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.net = Learner(learner_config).to(device)
        self.meta_optim = optim.AdamW(self.net.parameters(), lr=self.meta_lr)

    def forward(self, x_spt, y_spt, x_qry, y_qry):
        """
         setsz for a task = k_spt * n_way
         querysz for a teask = k_query * n_way
                :param x_spt:   [Task_num, setsz,embedding_length]
                :param y_spt:   [Task_num, setsz]
                :param x_qry:   [Task_num, querysz, embedding_length]
                :param y_qry:   [Task_num, querysz]
                :return:

                temp_model = copy.deepcopy(self.net)
                """
        querysz = x_qry.size(1)

        losses_q = [0 for _ in range(self.update_step)]  # losses_q[i] is the loss on step i
        corrects = [0 for _ in range(self.update_step)]
        network_params = []

        for i in range(self.task_num):
            if i == 0:
                logits = self.net(x_spt[i])

                net_params = list(dict(self.net.named_parameters()).values())  # Extract parameter tensors and convert to list
                network_params = net_params
            else:
                #print("step 1")
                net_params = network_params
                logits = self.net.updated_forward(x_spt[i],net_params)

            logits_reshaped = logits.view(-1, 2)
            y_reshaped = y_spt[i].view(-1)
            loss = F.cross_entropy(logits_reshaped, y_reshaped)
            grad = torch.autograd.grad(loss, self.net.parameters())
            fast_weights = list(map(lambda p: p[1] - self.update_lr * p[0], zip(grad, self.net.parameters())))

            # this is the loss and accuracy after the first update
            with torch.no_grad():
                # [setsz, nway]
                logits_q = self.net.updated_forward(x_qry[i], fast_weights)
                logits_reshaped_q = logits_q.view(-1, 2)
                y_reshaped_q = y_qry[i].view(-1)
                loss_q = F.cross_entropy(logits_reshaped_q, y_reshaped_q)
                losses_q[1] += loss_q
                # [setsz]
                #pred_q = F.softmax(logits_q, dim=1).argmax(dim=1)
                #correct = torch.eq(pred_q, y_qry[i]).sum().item()
                #corrects[1] = corrects[1] + correct

            for k in range(1, self.update_step):
                # 1. run the i-th task and compute loss for k=1~K-1
                logits = self.net(x_spt[i])
                logits_reshaped = logits.view(-1, 2)
                y_reshaped = y_spt[i].view(-1)
                loss = F.cross_entropy(logits_reshaped, y_reshaped)
                # 2. compute grad on theta_pi
                grad = torch.autograd.grad(loss, self.net.parameters())
                # 3. theta_pi = theta_pi - train_lr * grad
                fast_weights = list(map(lambda p: p[1] - self.update_lr * p[0], zip(grad, fast_weights)))

                logits_q = self.net.updated_forward(x_qry[i], fast_weights)
                logits_reshaped_q = logits_q.view(-1, 2)
                y_reshaped_q = y_qry[i].view(-1)
                loss_q = F.cross_entropy(logits_reshaped_q, y_reshaped_q)
                losses_q[k] += loss_q

        loss_q = losses_q[-1] / self.task_num

        # optimize theta parameters
        self.meta_optim.zero_grad()
        loss_q.backward()

        self.meta_optim.step()

        accs = np.array(corrects) / (querysz * self.task_num)

        return loss_q  # loss_q, pred

    def finetunning(self, x_spt, y_spt, x_qry, y_qry):
        """
                :param x_spt:   [setsz, max_tokens, embedding_length]
                :param y_spt:   [setsz]
                :param x_qry:   [querysz, max_tokens, embedding_length]
                :param y_qry:   [querysz]
                :return:
                """
        f1 = [0 for _ in range(self.update_step_test+1)]
        precision = [0 for _ in range(self.update_step_test+1)]
        recall = [0 for _ in range(self.update_step_test+1)]

        outloop_net = deepcopy(self.net)

        # 1. run the i-th task and compute loss for k=0
        logits = outloop_net(x_spt)
        logits_reshaped = logits.view(-1, 2)
        y_reshaped = y_spt.view(-1)
        loss = F.cross_entropy(logits_reshaped, y_reshaped)

        grad = torch.autograd.grad(loss, outloop_net.parameters())
        fast_weights = list(map(lambda p: p[1] - self.update_lr * p[0], zip(grad, outloop_net.parameters())))

        with torch.no_grad():
            logits_q = outloop_net.updated_forward(x_qry, fast_weights)
            pred_q = F.softmax(logits_q, dim=2).argmax(dim=2)
            update_precision, update_recall, update_f1 = self.calculate_metrics(y_qry, pred_q)
            f1[1] += update_f1
            precision[1] += update_precision
            recall[1] += update_recall

        for k in range(1, self.update_step_test):
            logits = outloop_net(x_spt)
            logits_reshaped = logits.view(-1, 2)
            y_reshaped = y_spt.view(-1)
            loss = F.cross_entropy(logits_reshaped, y_reshaped)
            grad = torch.autograd.grad(loss, outloop_net.parameters())
            fast_weights = list(map(lambda p: p[1] - self.update_lr * p[0], zip(grad, fast_weights)))

            logits_q = outloop_net.updated_forward(x_qry, fast_weights)
            # loss_q will be overwritten and just keep the loss_q on last update step.
            # loss_q = F.cross_entropy(logits_q, y_qry)

            with torch.no_grad():
                pred_q = F.softmax(logits_q, dim=2).argmax(dim=2)
                update_precision, update_recall, update_f1 = self.calculate_metrics(y_qry, pred_q)
                f1[k+1] += update_f1
                precision[k+1] += update_precision
                recall[k+1] += update_recall

        del outloop_net

        return f1, precision, recall

    def calculate_metrics(self, y_true, y_pred):
        # 计算True Positives (TP), False Positives (FP), True Negatives (TN), False Negatives (FN)
        print("y_true shape:", y_true.shape, flush=True)
        print("y_pred shape:", y_pred.shape, flush=True)
        TP = torch.sum((y_pred == 1) & (y_true == 1)).item()
        FP = torch.sum((y_pred == 1) & (y_true == 0)).item()
        TN = torch.sum((y_pred == 0) & (y_true == 0)).item()
        FN = torch.sum((y_pred == 0) & (y_true == 1)).item()
        print("TP:", TP, " FP:", FP, " TN:", TN, " FN:",FN,flush=True)
        # Precision
        precision = TP / (TP + FP) if (TP + FP) > 0 else 0
        # Recall
        recall = TP / (TP + FN) if (TP + FN) > 0 else 0
        # F1 Score
        f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
        return precision, recall, f1
