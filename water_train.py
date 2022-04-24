import sys
import os
curPath = os.path.abspath(os.path.dirname(__file__))
rootPath = os.path.split(curPath)[0]
sys.path.append(rootPath)

import torch
import numpy as np
import time
import utils.util as util
from engine import trainer
from utils import earlystopping
from sklearn import metrics as sk_metrics
import config as Config


def train(engine,dataloader):
    dataloader['train_loader'].shuffle()
    loss_epoch = []
    for step, (x, y) in enumerate(dataloader['train_loader'].get_iterator()):
        train_x = torch.Tensor(x).to(engine.device).transpose(1, 3)
        train_y = torch.Tensor(y).to(engine.device).transpose(1, 3)
        loss_batch,_,_ = engine.train(train_x, train_y[:, 0:-1, :, :])
        loss_epoch.append(loss_batch)
    return np.mean(loss_epoch)


def validate(engine,dataloader):
    for step, (x, y) in enumerate(dataloader['val_loader'].get_iterator()):
        test_x = torch.Tensor(x).to(engine.device).transpose(1, 3)
        test_y = torch.Tensor(y).to(engine.device).transpose(1, 3)
        metrics = engine.eval(test_x, test_y[:, 0:-1, :, :])
        return metrics


def test(engine,dataloader,model_path):

    engine.model.load_state_dict(torch.load(model_path))
    scaler = dataloader['scaler']

    outputs = []
    realy = torch.Tensor(dataloader['y_test']).to(engine.device)
    if Config.in_dim == 2:
        realy = realy.transpose(1, 3)[:, 0, :, :]
    else:
        realy = realy.transpose(1, 3)[:, 0:-1, :, :]

    engine.model.eval()
    for iter, (x, y) in enumerate(dataloader['test_loader'].get_iterator()):
        testx = torch.Tensor(x).to(engine.device).transpose(1, 3)
        with torch.no_grad():
            preds = engine.input_model(testx).transpose(1, 3)
        outputs.append(preds.squeeze())

    yhat = torch.cat(outputs, dim=0)
    yhat = yhat[:realy.size(0), ...]

    print("Training finished")
    # print("The valid loss on best model is", str(round(his_loss[bestid], 4)))



    # single dimension
    if Config.in_dim == 2:
        # save predicted value as file
        preds = scaler.inverse_transform(yhat)
        if Config.fac_single:
            save_root = f"data/output/{Config.section}/y/singleWaveNet/{Config.out_dir}/{Config.fac_index}"
        else:
            save_root = f"data/output/{Config.section}/y/multiWaveNet/{Config.out_dir}/{Config.fac_index}"
        if not os.path.exists(save_root):
            os.makedirs(save_root)

        pred_np = preds.to('cpu').numpy()
        realy_np = realy.to('cpu').numpy()
        np.savez_compressed(
            os.path.join(save_root, f"out.npz"),
            y_pred=pred_np,
            y_test=realy_np
        )

        text_output = []

        amae = []
        amape = []
        armse = []
        ar2 = []
        for i in range(3):
            pred = scaler.inverse_transform(yhat[:, :, i])
            real = realy[:, :, i]
            metrics = util.metric(pred, real)
            r2 = sk_metrics.r2_score(real.to('cpu').numpy(),pred.to('cpu').numpy())
            log = f'horizon {i + 1}, Test MAE: {metrics[0]:.4f}, Test MAPE: {metrics[1]:.4f}, Test RMSE: {metrics[2]:.4f}, Test R2: {r2:.4f}'
            print(log)
            text_output.append(log)
            amae.append(metrics[0])
            amape.append(metrics[1])
            armse.append(metrics[2])
            ar2.append(r2)

        log = f'On average over 3 horizons, Test MAE: {np.mean(amae):.4f}, Test MAPE: {np.mean(amape):.4f},' \
              f' Test RMSE: {np.mean(armse):.4f}, Test R2: {np.mean(ar2):.4f}'
        print(log)
        text_output.append(log)

        # save result as file
        if Config.fac_single:
            save_root = f"data/output/{Config.section}/text/singleWaveNet/{Config.out_dir}/{Config.fac_index}"
        else:
            save_root = f"data/output/{Config.section}/text/multiWaveNet/{Config.out_dir}/{Config.fac_index}"
        if not os.path.exists(save_root):
            os.makedirs(save_root)
        fh = open(f'{save_root}/result.txt', 'w', encoding='utf-8')
        fh.write('\n'.join(text_output))
        fh.close()

        return np.mean(amae), np.mean(amape), np.mean(armse)


def run_once():
    # set seed
    util.set_seed(Config.seed)

    # load data
    device = torch.device(Config.device)
    sensor_ids, sensor_id_to_ind, adj_mx = util.load_adj(Config.adjdata,'doubletransition')
    dataloader = util.load_dataset(Config.data_dir, Config.batch_size, Config.batch_size, Config.batch_size)
    scaler = dataloader['scaler']
    supports = [torch.tensor(i).to(device).to(torch.float32) for i in adj_mx]
    # supports = [adj_mx[-1]]

    Config.print_param()

    # supports = [supports[-1]]
    supports = None

    if Config.fac_single:
        save_root = f"data/output/{Config.section}/model/singleWaveNet/{Config.out_dir}/{Config.fac_index}"
    else:
        save_root = f"data/output/{Config.section}/model/multiWaveNet/{Config.out_dir}/{Config.fac_index}"
    if not os.path.exists(save_root):
        os.makedirs(save_root)

    model_save_path = f'{save_root}/model.pth'
    full_model_save_path = f'{save_root}/full_model.pth'
    early_stopping = earlystopping.EarlyStopping(patience=Config.patience, path=model_save_path, verbose=True,full_path=full_model_save_path)

    engine = trainer(scaler, device, supports)

    print("start training...",flush=True)

    train_loss = []
    val_loss = []
    for e in range(1,Config.epoch+1):
        train_loss_epoch = train(engine, dataloader)
        train_loss.append(train_loss_epoch)

        val_loss_epoch,_,_ =validate(engine, dataloader)
        val_loss.append(val_loss_epoch)
        print("Epoch:{},validate loss:{}".format(e, val_loss_epoch))

        early_stopping(val_loss_epoch,engine.model)
        if early_stopping.early_stop:
            print("Early stopping.")
            break

    # save loss value as file
    train_loss = np.array(train_loss)
    val_loss = np.array(val_loss)
    if Config.fac_single:
        save_root = f"data/output/{Config.section}/loss/singleWaveNet/{Config.out_dir}/{Config.fac_index}"
    else:
        save_root = f"data/output/{Config.section}/loss/multiWaveNet/{Config.out_dir}/{Config.fac_index}"
    if not os.path.exists(save_root):
        os.makedirs(save_root)
    np.savez_compressed(
        os.path.join(save_root, f"loss.npz"),
        train_loss=train_loss,
        val_loss=val_loss
    )

    return test(engine,dataloader,model_save_path)


if __name__ == "__main__":
    t1 = time.time()
    run_once()
    t2 = time.time()
    print("Total time spent: {:.4f}".format(t2 - t1))





