import torch
import numpy as np
import pandas as pd
import torch.nn as nn
from tqdm import tqdm
from sklearn import metrics
from torch.nn import functional as F
from torch.nn.functional import one_hot

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ref: https://github.com/pykt-team/pykt-toolkit/blob/main/pykt/models/evaluate_model.py
def evaluate(model, test_loader):
    model_name = model.model_name
    with torch.no_grad():
        y_trues = []
        y_scores = []
        test_mini_index = 0
        for data in tqdm(test_loader):
            # if model_name in ["dkt_forget", "lpkt"]:
            #     q, c, r, qshft, cshft, rshft, m, sm, d, dshft = data
            if model_name in ["dkt_forget", "bakt_time"]:
                dcur, dgaps = data
            else:
                dcur = data
            if model_name in ["dimkt"]:
                q, c, r, sd,qd = dcur["qseqs"], dcur["cseqs"], dcur["rseqs"], dcur["sdseqs"],dcur["qdseqs"]
                qshft, cshft, rshft, sdshft,qdshft = dcur["shft_qseqs"], dcur["shft_cseqs"], dcur["shft_rseqs"], dcur["shft_sdseqs"],dcur["shft_qdseqs"]
                sd, qd, sdshft, qdshft = sd.to(device), qd.to(device), sdshft.to(device), qdshft.to(device)
            else:
                q, c, r = dcur["qseqs"], dcur["cseqs"], dcur["rseqs"]
                qshft, cshft, rshft= dcur["shft_qseqs"], dcur["shft_cseqs"], dcur["shft_rseqs"]
            m, sm = dcur["masks"], dcur["smasks"]
            q, c, r, qshft, cshft, rshft, m, sm = q.to(device), c.to(device), r.to(device), qshft.to(device), cshft.to(device), rshft.to(device), m.to(device), sm.to(device)

            model.eval()

            # print(f"before y: {y.shape}")
            cq = torch.cat((q[:,0:1], qshft), dim=1)
            cc = torch.cat((c[:,0:1], cshft), dim=1)
            cr = torch.cat((r[:,0:1], rshft), dim=1)
            if model_name in ["atdkt"]:
                '''
                y = model(dcur) 
                import pickle
                with open(f"{test_mini_index}_result.pkl",'wb') as f:
                    data = {"y":y,"cshft":cshft,"num_c":model.num_c,"rshft":rshft,"qshft":qshft,"sm":sm}
                    pickle.dump(data,f)
                '''
                y = model(dcur)
                y = (y * one_hot(cshft.long(), model.num_c)).sum(-1)
            # elif model_name in ["rkt"]:
            #     y, attn = model(dcur, rel)
            #     y = y[:,1:]
            #     if q.numel() > 0:
            #         c,cshft = q,qshft   #question level
            elif model_name in ["bakt_time"]:
                y = model(dcur, dgaps)
                y = y[:,1:]
            elif model_name in ["simplekt","stablekt", "sparsekt"]:
                y = model(dcur)
                y = y[:,1:]
            elif model_name in ["dkt", "dkt+"]:
                y = model(c.long(), r.long())
                y = (y * one_hot(cshft.long(), model.num_c)).sum(-1)
            elif model_name in ["dkt_forget"]:
                y = model(c.long(), r.long(), dgaps)
                y = (y * one_hot(cshft.long(), model.num_c)).sum(-1)
            elif model_name in ["dkvmn","deep_irt", "skvmn","deep_irt"]:
                y = model(cc.long(), cr.long())
                y = y[:,1:]
            elif model_name in ["kqn", "sakt"]:
                y = model(c.long(), r.long(), cshft.long())
            elif model_name == "saint":
                y = model(cq.long(), cc.long(), r.long())
                y = y[:, 1:]
            elif model_name in ["akt","extrakt","folibikt", "akt_vector", "akt_norasch", "akt_mono", "akt_attn", "aktattn_pos", "aktmono_pos", "akt_raschx", "akt_raschy", "aktvec_raschx"]:
                y, reg_loss = model(cc.long(), cr.long(), cq.long())
                y = y[:,1:]
            elif model_name == "AlignKT":
                logit, *_ = model(c.long(), r.long(), cshft.long(), sm.long(), q.long(), qshft.long())
                y = F.sigmoid(logit)
            elif model_name in ["dtransformer"]:
                output, *_ = model.predict(cc.long(), cr.long(), cq.long())
                sg = nn.Sigmoid()
                y = sg(output)
                y = y[:,1:]
            elif model_name in ["atkt", "atktfix"]:
                y, _ = model(c.long(), r.long())
                y = (y * one_hot(cshft.long(), model.num_c)).sum(-1)
            elif model_name == "gkt":
                y = model(cc.long(), cr.long())
            elif model_name == "hawkes":
                ct = torch.cat((dcur["tseqs"][:,0:1], dcur["shft_tseqs"]), dim=1)
                # csm = torch.cat((dcur["smasks"][:,0:1], dcur["smasks"]), dim=1)
                y = model(cc.long(), cq.long(), ct.long(), cr.long())#, csm.long())
                y = y[:, 1:]
            elif model_name == "dimkt":
                y = model(q.long(),c.long(),sd.long(),qd.long(),r.long(),qshft.long(),cshft.long(),sdshft.long(),qdshft.long())
            # print(f"after y: {y.shape}")


            y = torch.masked_select(y, sm).detach().cpu()
            # print(f"pred_results:{y}")
            t = torch.masked_select(rshft, sm).detach().cpu()

            y_trues.append(t.numpy())
            y_scores.append(y.numpy())
            test_mini_index+=1
        ts = np.concatenate(y_trues, axis=0)
        ps = np.concatenate(y_scores, axis=0)
        print(f"ts.shape: {ts.shape}, ps.shape: {ps.shape}")
        auc = metrics.roc_auc_score(y_true=ts, y_score=ps)

        prelabels = [1 if p >= 0.5 else 0 for p in ps]
        acc = metrics.accuracy_score(ts, prelabels)
    # if save_path != "":
    #     pd.to_pickle(dres, save_path+".pkl")
    return auc, acc


def early_fusion(curhs, model, model_name):
    p = None
    if model_name in ["dkvmn", "skvmn"]:
        p = model.p_layer(model.dropout_layer(curhs[0]))
        p = torch.sigmoid(p)
        p = p.squeeze(-1)
    elif model_name in ["deep_irt"]:
        # p = model.p_layer(curhs[0])
        stu_ability = model.ability_layer(curhs[0])  # equ 12
        que_diff = model.diff_layer(curhs[1])  # equ 13
        p = torch.sigmoid(3.0 * stu_ability - que_diff)  # equ 14
        p = p.squeeze(-1)
    elif model_name in ["akt", "extrakt", "folibikt", "dtransformer", "simplekt", "stablekt", "bakt_time", "sparsekt",
                        "akt_vector", "akt_norasch", "akt_mono", "akt_attn", "aktattn_pos", "aktmono_pos", "akt_raschx",
                        "akt_raschy", "aktvec_raschx"]:
        output = model.out(curhs[0]).squeeze(-1)
        m = nn.Sigmoid()
        p = m(output)
    elif model_name in ["AlignKT"]:
        logit = model.out(curhs[0]).squeeze(-1)
        p = F.sigmoid(logit)
    elif model_name == "saint":
        p = model.out(model.dropout(curhs[0]))
        p = torch.sigmoid(p).squeeze(-1)
    elif model_name == "sakt":
        p = torch.sigmoid(model.pred(model.dropout_layer(curhs[0]))).squeeze(-1)
    elif model_name == "kqn":
        logits = torch.sum(curhs[0] * curhs[1], dim=1)  # (batch_size, max_seq_len)
        p = model.sigmoid(logits)
    elif model_name == "hawkes":
        p = curhs[0].sigmoid()
    elif model_name == "lpkt":
        p = model.sig(model.linear_5(torch.cat((curhs[1], curhs[0]), 1))).sum(1) / model.d_k
    return p


def late_fusion(dcur, curdf, fusion_type=None):
    if fusion_type is None:
        fusion_type = ["mean", "vote", "all"]
    high, low = [], []
    for pred in curdf["preds"]:
        if pred >= 0.5:
            high.append(pred)
        else:
            low.append(pred)
    correctnum = []
    if "mean" in fusion_type:
        dcur.setdefault("late_mean", [])
        dcur["late_mean"].append(round(float(curdf["preds"].mean()), 4))
    if "vote" in fusion_type:
        dcur.setdefault("late_vote", [])
        correctnum = list(curdf["preds"] >= 0.5).count(True)
        late_vote = np.mean(high) if correctnum / len(curdf["preds"]) >= 0.5 else np.mean(low)
        dcur["late_vote"].append(late_vote)
    if "all" in fusion_type:
        dcur.setdefault("late_all", [])
        late_all = np.mean(high) if correctnum == len(curdf["preds"]) else np.mean(low)
        dcur["late_all"].append(late_all)
    return


def effective_fusion(df, model, model_name, fusion_type):
    dres = dict()
    df = df.groupby("qidx", as_index=True, sort=True)  # .mean()

    curhs, curr = [[], []], []
    dcur = {"late_trues": [], "qidxs": [], "questions": [], "concepts": [], "row": [], "concept_preds": []}
    hasearly = ["dkvmn", "deep_irt", "skvmn", "kqn", "akt", "extrakt", "folibikt", "dtransformer", "simplekt",
                "stablekt", "bakt_time", "sparsekt", "saint", "sakt", "hawkes", "akt_vector", "akt_norasch", "akt_mono",
                "akt_attn", "aktattn_pos", "aktmono_pos", "akt_raschx", "akt_raschy", "aktvec_raschx", "lpkt", "AlignKT"]
    for ui in df:
        # 一题一题处理
        curdf = ui[1]
        if model_name in hasearly and model_name not in ["kqn", "lpkt", "deep_irt"]:
            curhs[0].append(curdf["hidden"].mean().astype(float))
        elif model_name == "kqn":
            curhs[0].append(curdf["ek"].mean().astype(float))
            curhs[1].append(curdf["es"].mean().astype(float))
        elif model_name == "lpkt":
            curhs[0].append(curdf["h"].mean().astype(float))
            curhs[1].append(curdf["e_data"].mean().astype(float))
        elif model_name == "deep_irt":
            curhs[0].append(curdf["h"].mean().astype(float))
            curhs[1].append(curdf["k"].mean().astype(float))
        else:
            # print(f"model: {model_name} has no early fusion res!")
            pass

        curr.append(int(curdf["response"].mean()))
        dcur["late_trues"].append(int(curdf["response"].mean()))
        dcur["qidxs"].append(ui[0])
        dcur["row"].append(int(curdf["row"].mean()))
        dcur["questions"].append(",".join([str(int(s)) for s in curdf["questions"].tolist()]))
        dcur["concepts"].append(",".join([str(int(s)) for s in curdf["concepts"].tolist()]))
        late_fusion(dcur, curdf)
        # save original predres in concepts
        dcur["concept_preds"].append(",".join([str(round(s, 4)) for s in (curdf["preds"].tolist())]))

    for key in dcur:
        dres.setdefault(key, [])
        dres[key].append(np.array(dcur[key]))
    # early fusion
    if "early_fusion" in fusion_type and model_name in hasearly:
        curhs = [torch.tensor(np.array(curh)).float().to(device) for curh in curhs]
        curr = torch.tensor(curr).long().to(device)
        p = early_fusion(curhs, model, model_name)
        dres.setdefault("early_trues", [])
        dres["early_trues"].append(curr.cpu().numpy())
        dres.setdefault("early_preds", [])
        dres["early_preds"].append(p.cpu().numpy())
    return dres


def group_fusion(dmerge, model, model_name, fusion_type):
    hs, sms, cq, cc, rs, ps, qidxs, rests, orirows = dmerge["hs"], dmerge["sm"], dmerge["cq"], dmerge["cc"], dmerge[
        "cr"], dmerge["y"], dmerge["qidxs"], dmerge["rests"], dmerge["orirow"]
    if cq.shape[1] == 0:
        cq = cc

    hasearly = ["dkvmn", "deep_irt", "skvmn", "kqn", "dtransformer", "akt", "extrakt", "folibikt", "simplekt",
                "stablekt", "bakt_time", "sparsekt", "saint", "sakt", "hawkes", "akt_vector", "akt_norasch", "akt_mono",
                "akt_attn", "aktattn_pos", "aktmono_pos", "akt_raschx", "akt_raschy", "aktvec_raschx", "lpkt", "AlignKT"]

    alldfs, drest = [], dict()  # not predict infos!
    # print(f"real bz in group fusion: {rs.shape[0]}")
    # realbz = rs.shape[0]
    for bz in range(rs.shape[0]):
        cursm = ([0] + sms[bz].cpu().tolist())
        curqidxs = ([-1] + qidxs[bz].cpu().tolist())
        currests = ([-1] + rests[bz].cpu().tolist())
        currows = ([-1] + orirows[bz].cpu().tolist())
        curps = ([-1] + ps[bz].cpu().tolist())
        # print(f"qid: {len(curqidxs)}, select: {len(cursm)}, response: {len(rs[bz].cpu().tolist())}, preds: {len(curps)}")
        df = pd.DataFrame({"qidx": curqidxs, "rest": currests, "row": currows, "select": cursm,
                           "questions": cq[bz].cpu().tolist(), "concepts": cc[bz].cpu().tolist(),
                           "response": rs[bz].cpu().tolist(), "preds": curps})
        if model_name in hasearly and model_name not in ["kqn", "lpkt", "deep_irt"]:
            df["hidden"] = [np.array(a) for a in hs[0][bz].cpu().tolist()]
        elif model_name == "kqn":
            df["ek"] = [np.array(a) for a in hs[0][bz].cpu().tolist()]
            df["es"] = [np.array(a) for a in hs[1][bz].cpu().tolist()]
        elif model_name == "lpkt":
            # print(f"hidden:{hs[0].shape}")
            df["h"] = [np.array(a) for a in hs[0][bz].cpu().tolist()]
            # print(f"e_data:{hs[1].shape}")
            df["e_data"] = [np.array(a) for a in hs[1][bz].cpu().tolist()]
        elif model_name == "deep_irt":
            df["h"] = [np.array(a) for a in hs[0][bz].cpu().tolist()]
            df["k"] = [np.array(a) for a in hs[1][bz].cpu().tolist()]
        df = df[df["select"] != 0]
        alldfs.append(df)

    effective_dfs, rest_start = [], -1
    flag = False
    for i in range(len(alldfs) - 1, -1, -1):
        df = alldfs[i]
        counts = (df["rest"] == 0).value_counts()
        if not flag and False not in counts:  # has no question rest > 0
            flag = True
            effective_dfs.append(df)
            rest_start = i + 1
        elif flag:
            effective_dfs.append(df)
    if rest_start == -1:
        rest_start = 0
    # merge rest
    for key in dmerge.keys():
        if key == "hs":
            drest[key] = []
            if model_name in hasearly and model_name not in ["kqn", "lpkt", "deep_irt"]:
                drest[key] = [dmerge[key][0][rest_start:]]
            elif model_name in ["kqn", "lpkt", "deep_irt"]:
                drest[key] = [dmerge[key][0][rest_start:], dmerge[key][1][rest_start:]]
        else:
            drest[key] = dmerge[key][rest_start:]
    # restlen = drest["cr"].shape[0]

    dfs = dict()
    for df in effective_dfs:
        for i, row in df.iterrows():
            for key in row.keys():
                dfs.setdefault(key, [])
                dfs[key].extend([row[key]])
    df = pd.DataFrame(dfs)
    # print(f"real bz: {realbz}, effective_dfs: {len(effective_dfs)}, rest_start: {rest_start}, drestlen: {restlen}, predict infos: {df.shape}")

    if df.shape[0] == 0:
        return {}, drest

    dres = effective_fusion(df, model, model_name, fusion_type)

    dfinal = dict()
    for key in dres:
        dfinal[key] = np.concatenate(dres[key], axis=0)
    # early = False
    # if model_name in hasearly and "early_fusion" in fusion_type:
    #     early = True
    # save_question_res(dfinal, fout, early)
    return dfinal, drest


def evaluate_question(model, test_loader, fusion_type=None):
    # dkt / dkt+ / dkt_forget / atkt: give past -> predict all. has no early fusion!!!
    # dkvmn / akt / saint: give cur -> predict cur
    # sakt: give past+cur -> predict cur
    # kqn: give past+cur -> predict cur
    model_name = model.model_name
    if fusion_type is None:
        fusion_type = ["early_fusion", "late_fusion"]

    hasearly = ["dkvmn", "deep_irt", "skvmn", "kqn", "dtransformer", "akt", "extrakt", "folibikt", "simplekt",
                "stablekt", "bakt_time", "sparsekt", "saint", "sakt", "hawkes", "akt_vector", "akt_norasch", "akt_mono",
                "akt_attn", "aktattn_pos", "aktmono_pos", "akt_raschx", "akt_raschy", "aktvec_raschx", "lpkt", "AlignKT"]

    with torch.no_grad():
        dinfos = dict()
        dhistory = dict()
        history_keys = ["hs", "sm", "cq", "cc", "cr", "y", "qidxs", "rests", "orirow"]
        # for key in history_keys:
        #     dhistory[key] = []
        y_trues, y_scores = [], []
        lenc = 0
        for data in tqdm(test_loader):
            if model_name in ["dkt_forget", "bakt_time"]:
                dcurori, dgaps, dqtest = data
            else:
                dcurori, dqtest = data

            if model_name in ["dimkt"]:
                q, c, r, sd, qd = dcurori["qseqs"], dcurori["cseqs"], dcurori["rseqs"], dcurori["sdseqs"], dcurori[
                    "qdseqs"]
                qshft, cshft, rshft, sdshft, qdshft = dcurori["shft_qseqs"], dcurori["shft_cseqs"], dcurori[
                    "shft_rseqs"], dcurori["shft_sdseqs"], dcurori["shft_qdseqs"]
                sd, qd, sdshft, qdshft = sd.to(device), qd.to(device), sdshft.to(device), qdshft.to(device)
            else:
                q, c, r = dcurori["qseqs"], dcurori["cseqs"], dcurori["rseqs"]
                qshft, cshft, rshft = dcurori["shft_qseqs"], dcurori["shft_cseqs"], dcurori["shft_rseqs"]
            m, sm = dcurori["masks"], dcurori["smasks"]
            q, c, r, qshft, cshft, rshft, m, sm = q.to(device), c.to(device), r.to(device), qshft.to(device), cshft.to(
                device), rshft.to(device), m.to(device), sm.to(device)
            qidxs, rests, orirow = dqtest["qidxs"], dqtest["rests"], dqtest["orirow"]
            lenc += q.shape[0]
            # print("="*20)
            # print(f"start predict seqlen: {lenc}")
            model.eval()

            # print(f"before y: {y.shape}")
            cq = torch.cat((q[:, 0:1], qshft), dim=1)
            cc = torch.cat((c[:, 0:1], cshft), dim=1)
            cr = torch.cat((r[:, 0:1], rshft), dim=1)
            dcur = dict()
            if model_name in ["dkvmn", "skvmn"]:
                y, h = model(cc.long(), cr.long(), True)
                y = y[:, 1:]
            elif model_name in ["deep_irt"]:
                y, h, k = model(cc.long(), cr.long(), True)
                y = y[:, 1:]
            elif model_name in ["bakt_time"]:
                y, h = model(dcurori, dgaps, qtest=True, train=False)
                y = y[:, 1:]
                # start_hemb = torch.tensor([-1] * (h.shape[0] * h.shape[2])).reshape(h.shape[0], 1, h.shape[2]).to(device)
                # print(start_hemb.shape, h.shape)
                # h = torch.cat((start_hemb, h), dim=1) # add the first hidden emb
            elif model_name in ["simplekt", "stablekt", "sparsekt"]:
                y, h = model(dcurori, qtest=True, train=False)
                y = y[:, 1:]
            elif model_name in ["akt", "extrakt", "folibikt", "akt_vector", "akt_norasch", "akt_mono", "akt_attn",
                                "aktattn_pos", "aktmono_pos", "akt_raschx", "akt_raschy", "aktvec_raschx"]:
                y, reg_loss, h = model(cc.long(), cr.long(), cq.long(), True)
                y = y[:, 1:]
            elif model_name in ["AlignKT"]:
                logit, h, _ = model(c.long(), r.long(), cshft.long(), sm.long(), q.long(), qshft.long())
                y = F.sigmoid(logit)
                init_a_invalid_tensor_row = torch.zeros(h.shape[0], 1, model.d_model*2).to(device)
                h = torch.cat([init_a_invalid_tensor_row, h], dim=1)
            elif model_name in ["dtransformer"]:
                output, h, *_ = model.predict(cc.long(), cr.long(), cq.long())
                sg = nn.Sigmoid()
                y = sg(output)
                y = y[:, 1:]
            elif model_name == "saint":
                y, h = model(cq.long(), cc.long(), r.long(), True)
                y = y[:, 1:]
            elif model_name == "sakt":
                y, h = model(c.long(), r.long(), cshft.long(), True)
                start_hemb = torch.tensor([-1] * (h.shape[0] * h.shape[2])).reshape(h.shape[0], 1, h.shape[2]).to(
                    device)
                # print(start_hemb.shape, h.shape)
                h = torch.cat((start_hemb, h), dim=1)  # add the first hidden emb
            elif model_name == "kqn":
                y, ek, es = model(c.long(), r.long(), cshft.long(), True)
                # print(f"ek: {ek.shape},  es: {es.shape}")
                start_hemb = torch.tensor([-1] * (ek.shape[0] * ek.shape[2])).reshape(ek.shape[0], 1, ek.shape[2]).to(
                    device)
                ek = torch.cat((start_hemb, ek), dim=1)  # add the first hidden emb
                es = torch.cat((start_hemb, es), dim=1)  # add the first hidden emb
            elif model_name in ["atdkt"]:
                y = model(dcurori)  # c.long(), r.long(), q.long())
                y = (y * one_hot(cshft.long(), model.num_c)).sum(-1)
            elif model_name in ["dkt", "dkt+"]:
                y = model(c.long(), r.long())
                y = (y * one_hot(cshft.long(), model.num_c)).sum(-1)
            elif model_name in ["dkt_forget"]:
                y = model(c.long(), r.long(), dgaps)
                y = (y * one_hot(cshft.long(), model.num_c)).sum(-1)
            elif model_name in ["atkt", "atktfix"]:
                y, _ = model(c.long(), r.long())
                y = (y * one_hot(cshft.long(), model.num_c)).sum(-1)
            elif model_name == "gkt":
                y = model(cc.long(), cr.long())
            elif model_name == "hawkes":
                ct = torch.cat((dcurori["tseqs"][:, 0:1], dcurori["shft_tseqs"]), dim=1)
                y, h = model(cc.long(), cq.long(), ct.long(), cr.long(), True)
                y = y[:, 1:]
            elif model_name == "lpkt":
                cit = torch.cat((dcurori["itseqs"][:, 0:1], dcurori["shft_itseqs"]), dim=1)
                y, h, e_data = model(cq.long(), cr.long(), cit.long(), at_data=None, qtest=True)
                start_hemb = torch.tensor([-1] * (h.shape[0] * h.shape[2])).reshape(h.shape[0], 1, h.shape[2]).to(
                    device)  # add the first hidden emb
                h = torch.cat((start_hemb, h), dim=1)
                # e_data = torch.cat((start_hemb, e_data), dim=1)
                y = y[:, 1:]
            elif model_name == "dimkt":
                y = model(q.long(), c.long(), sd.long(), qd.long(), r.long(), qshft.long(), cshft.long(), sdshft.long(), qdshft.long())

            concepty = torch.masked_select(y, sm).detach().cpu()
            conceptt = torch.masked_select(rshft, sm).detach().cpu()

            y_trues.append(conceptt.numpy())
            y_scores.append(concepty.numpy())

            # hs, sms, rs, ps, qidxs, model, model_name, fusion_type
            hs = []
            if model_name == "kqn":
                hs = [ek, es]
            elif model_name == "lpkt":
                hs = [h, e_data]
            elif model_name == "deep_irt":
                hs = [h, k]
            elif model_name in hasearly:
                hs = [h]
            dcur["hs"], dcur["sm"], dcur["cq"], dcur["cc"], dcur["cr"], dcur["y"], dcur["qidxs"], dcur["rests"], dcur[
                "orirow"] = hs, sm, cq, cc, cr, y, qidxs, rests, orirow
            # merge history
            dmerge = dict()
            for key in history_keys:
                if len(dhistory) == 0:  # dhistory为空时，dmerge获得dcur的所有key。
                    dmerge[key] = dcur[key]
                else:  # 若不为空，则对dmerge获得dcur和dhistory的拼接。
                    if key == "hs":  # 对hs特殊处理，拼接hs的第一个元素，即h（通常只有一个h被放进hs中）。
                        dmerge[key] = []
                        if model_name == "kqn":
                            dmerge[key] = [[], []]
                            dmerge[key][0] = torch.cat((dhistory[key][0], dcur[key][0]), dim=0)
                            dmerge[key][1] = torch.cat((dhistory[key][1], dcur[key][1]), dim=0)
                        elif model_name == "lpkt":
                            dmerge[key] = [[], []]
                            dmerge[key][0] = torch.cat((dhistory[key][0], dcur[key][0]), dim=0)
                            dmerge[key][1] = torch.cat((dhistory[key][1], dcur[key][1]), dim=0)
                        elif model_name == "deep_irt":
                            dmerge[key] = [[], []]
                            dmerge[key][0] = torch.cat((dhistory[key][0], dcur[key][0]), dim=0)
                            dmerge[key][1] = torch.cat((dhistory[key][1], dcur[key][1]), dim=0)
                        elif model_name in hasearly:
                            dmerge[key] = [torch.cat((dhistory[key][0], dcur[key][0]), dim=0)]
                    else:
                        dmerge[key] = torch.cat((dhistory[key], dcur[key]), dim=0)

            dcur, dhistory = group_fusion(dmerge, model, model_name, fusion_type)  # 在test_set的循环，dhistory逐渐增长
            for key in dcur:
                dinfos.setdefault(key, [])
                dinfos[key].append(dcur[key])

            if "early_fusion" in dinfos and "late_fusion" in dinfos:
                assert dinfos["early_trues"][-1].all() == dinfos["late_trues"][-1].all()
            # import sys
            # sys.exit()
        # ori concept eval
        aucs, accs = dict(), dict()
        ts = np.concatenate(y_trues, axis=0)
        ps = np.concatenate(y_scores, axis=0)
        # print(f"ts.shape: {ts.shape}, ps.shape: {ps.shape}")
        auc = metrics.roc_auc_score(y_true=ts, y_score=ps)
        prelabels = [1 if p >= 0.5 else 0 for p in ps]
        acc = metrics.accuracy_score(ts, prelabels)
        aucs["concepts"] = auc
        accs["concepts"] = acc

        # print(f"dinfos: {dinfos.keys()}")
        for key in dinfos:
            if key not in ["late_mean", "late_vote", "late_all", "early_preds"]:
                continue
            ts = np.concatenate(dinfos['late_trues'], axis=0)  # early_trues == late_trues
            ps = np.concatenate(dinfos[key], axis=0)
            # print(f"key: {key}, ts.shape: {ts.shape}, ps.shape: {ps.shape}")
            auc = metrics.roc_auc_score(y_true=ts, y_score=ps)
            prelabels = [1 if p >= 0.5 else 0 for p in ps]
            acc = metrics.accuracy_score(ts, prelabels)
            if key == "late_mean":
                confusion_matrix = metrics.confusion_matrix(ts, prelabels)
                print(f"late_mean_confusion_matrix: {confusion_matrix}")
            aucs[key] = auc
            accs[key] = acc
    return aucs, accs
