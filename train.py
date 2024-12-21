import os
import torch
import numpy as np
from tqdm import tqdm
from torch.optim import Adam
from eval_utils import evaluate, evaluate_question
from torch.nn import BCEWithLogitsLoss
from torch.nn.functional import binary_cross_entropy

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ref: https://github.com/pykt-team/pykt-toolkit/blob/main/pykt/models/train_model.py
def cal_loss(ys, rshft, sm, preloss, model_name):
    if model_name in ['akt', 'folibikt', 'extrakt']:
        y = torch.masked_select(ys[0], sm)
        t = torch.masked_select(rshft, sm)
        loss = binary_cross_entropy(y.double(), t.double()) + preloss[0]
        return loss

    elif model_name in ["simplekt", "stablekt", "sparsekt"]:
        y = torch.masked_select(ys[0], sm)
        t = torch.masked_select(rshft, sm)
        # print(f"loss1: {y.shape}")
        loss = binary_cross_entropy(y.double(), t.double())
        return loss

    elif model_name == 'AlignKT':
        y = torch.masked_select(ys[0], sm)
        t = torch.masked_select(rshft, sm)
        loss = BCEWithLogitsLoss()
        l= loss(y.double(), t.double()) + preloss[0]  # preloss[0] is the sum of contrastive loss and l2_reg
        return l


def model_forward(model, data):
    dcur = data

    q, c, r, t = dcur["qseqs"].to(device), dcur["cseqs"].to(device), dcur["rseqs"].to(device), dcur["tseqs"].to(device)
    qshft, cshft, rshft, tshft = (dcur["shft_qseqs"].to(device), dcur["shft_cseqs"].to(device),
                                  dcur["shft_rseqs"].to(device), dcur["shft_tseqs"].to(device))
    m, sm = dcur["masks"].to(device), dcur["smasks"].to(device)
    cq = torch.cat((q[:,0:1], qshft), dim=1)
    cc = torch.cat((c[:,0:1], cshft), dim=1)
    cr = torch.cat((r[:,0:1], rshft), dim=1)

    ys, preloss = [], []
    if model.model_name in ['akt', 'folibikt', 'extrakt']:
        y, reg_loss = model(cc.long(), cr.long(), cq.long())
        ys.append(y[:,1:])
        preloss.append(reg_loss)

    elif model.model_name in ["simplekt", "stablekt", "sparsekt"]:
        y, y2, y3 = model(dcur, train=True)
        ys = [y[:,1:], y2, y3]

    elif model.model_name == 'AlignKT':
        logit, _, l2_reg = model(c.long(), r.long(), cshft.long(), sm.long(), q.long(), qshft.long())
        ys.append(logit)
        # maybe contrastive loss will be added
        preloss.append(l2_reg)

    # cal loss
    loss = cal_loss(ys, rshft, sm, preloss, model.model_name)
    return loss


def train_model(model, train_loader, valid_loader, test_loader=None,
                testq_data_loader=None,  save_model=True, **kwargs):
    num_epochs, ckpt_path = kwargs['num_epochs'], kwargs['save_dir']

    # model info
    model_name = model.model_name
    model_size = kwargs['d_model']
    model_d_ff = kwargs['d_ff']
    model_n_head = kwargs['n_heads']
    model_dropout = kwargs['drop_out']
    model_n_blocks = kwargs['n_blocks']
    model_dataset = kwargs['dataset_name']
    model_info = '_'.join([model_name, str(model_size), str(model_d_ff), str(model_n_head), str(model_dropout), str(model_n_blocks), model_dataset])

    # metrics
    max_auc, best_epoch = 0, -1
    train_step = 0

    opt = Adam(model.parameters(), lr=1e-4, weight_decay=1e-5)
    model.to(device)

    for i in range(1, num_epochs + 1):
        loss_mean = []

        for data in tqdm(train_loader):
            train_step += 1
            model.train()
            loss = model_forward(model, data)
            opt.zero_grad()
            loss.backward()  # compute gradients
            opt.step()  # update model’s parameters
            loss_mean.append(loss.detach().cpu().numpy())

            # if train_step % 10 == 0:
            #     text = f"Total train step is {train_step}, the loss is {loss.item():.5}"
            #     print(text)

        loss_mean = np.mean(loss_mean)
        auc, acc = evaluate(model, valid_loader)
        validauc, validacc = auc, acc

        if auc > max_auc + 3e-4:
            if save_model:
                torch.save(model.state_dict(), os.path.join(ckpt_path, model_info + "_model.ckpt"))
            max_auc = auc
            best_epoch = i

        print(f"Epoch: {i}")
        print(f"validauc: {validauc:.4}, validacc: {validacc:.4}, best epoch: {best_epoch}, best auc: {max_auc:.4}, train loss: {loss_mean}")

        if i - best_epoch >= 10:
            break

    print("finish training!")
    print(f"model: {model_name}, save_dir: {ckpt_path}, dataset: {model_dataset}\n")

    # load the best model from the checkpoint
    best_weights = torch.load(os.path.join(ckpt_path, model_info + "_model.ckpt"))
    model.eval()
    model.load_state_dict(best_weights)

    # evaluate the best model on testset
    print("start Evaluation...")
    testauc, testacc = evaluate(model, test_loader)

    print(f"The best model's performance on testset: ")
    print(f"testauc: {testauc}, testacc: {testacc}\n")

    # evaluate the best model on test-question set
    dres = {}
    # save_test_ques_res = os.path.join(ckpt_path, model_info + "_model_predres.txt")
    q_testaucs, q_testaccs = evaluate_question(model, testq_data_loader)
    for key in q_testaucs:
        dres["oriauc_"+key] = q_testaucs[key]
    for key in q_testaccs:
        dres["oriacc_"+key] = q_testaccs[key]
    print(dres)
    print("finish evaluation!")

    return testauc, testacc, best_epoch, max_auc
