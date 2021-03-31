import numpy as np
import torch
import pickle

import dgl

import matplotlib.pyplot as plt
import pandas as pd
from sklearn.model_selection import train_test_split
import torch
from torch.nn.functional import binary_cross_entropy_with_logits as BCELoss
from torch.utils.data import DataLoader
from tqdm import tqdm, trange

import matplotlib.pyplot as plt
import dgl
import dgl.function as fn
import torch as th
import torch.nn as nn
import torch.nn.functional as F
from dgl import DGLGraph
from sklearn.metrics import precision_score,recall_score,f1_score
import torch.nn as nn

import networkx as nx
import json
import models
import os
import argparse
from configs import read_config

device = "cuda"

np.random.seed(0)
torch.manual_seed(0)



import itertools
from Dataset import *


def train_discriminator(disc, gen, batch_size,device,batch, opt_d):

    pair = []
    opt_d.zero_grad()
    
    graphs1,maps1,graphs2,maps2, real_targets = batch

    graphs1 = graphs1.to(device)
    graphs2 = graphs2.to(device)

    maps1 = maps1.to(device)
    maps2 = maps2.to(device)
    real_targets = torch.Tensor(real_targets).to(device)

    

    steps = [graphs1,maps1,graphs2,maps2]
    real_preds, im1 = disc(steps)
    pair.append(im1.detach().cpu())

    real_loss = torch.nn.functional.binary_cross_entropy(real_preds, real_targets)
    real_score = torch.mean(real_preds).item()
    
    
    action = gen(steps)
    
    fake_steps = models.make_step_modifications(action, steps)
    
    graphs1,maps1,graphs2,maps2, fake_targets = fake_steps
    graphs1 = graphs1.to(device)
    graphs2 = graphs2.to(device)
    fake_targets = torch.zeros(len(maps2)).to(device)

    maps1 = maps1.to(device)
    maps2 = maps2.to(device)

    fake_steps = [graphs1,maps1,graphs2,maps2]
    
    
    
    fake_preds,im2 = disc(fake_steps)
    pair.append(im2.detach().cpu())
    fake_loss = torch.nn.functional.binary_cross_entropy(fake_preds, fake_targets)
    fake_score = torch.mean(fake_preds).item()


    total_loss = real_loss + fake_loss
    total_loss.backward()

    opt_d.step()
    return total_loss.item(), real_score, fake_score



def train_generator(gen, disc, batch_size, opt_g, batch, device):
    
    opt_g.zero_grad()

    graphs1,maps1,graphs2,maps2, real_targets = batch
    graphs1 = graphs1.to(device)
    graphs2 = graphs2.to(device)

    maps1 = maps1.to(device)
    maps2 = maps2.to(device)
    real_targets = torch.ones(len(maps2)).to(device)

    steps = [graphs1,maps1,graphs2,maps2]

    action = gen(steps)
    

    fake_steps = models.make_step_modifications(action, steps)
    

    graphs1,maps1,graphs2,maps2, fake_targets = fake_steps
    graphs1 = graphs1.to(device)
    graphs2 = graphs2.to(device)

    maps1 = maps1.to(device)
    maps2 = maps2.to(device)
    fake_targets = torch.zeros(len(maps2)).to(device)#torch.Tensor(fake_targets).to(device)

    fake_steps = [graphs1,maps1,graphs2,maps2]
    
    
    preds,_ = disc(fake_steps)
    
    
    
    loss = torch.nn.functional.binary_cross_entropy(preds, fake_targets)
    
    
    loss.backward()
    
    
    opt_g.step()
    
    return loss.item()




#Training function. Takes in Discriminator and Generator. Generator can be just a random sampler of action. 

def fit(disc, gen, train_loader,epochs, device,lr_g = 1e-4, lr_d = 1e-4, start_idx=1, run_on_valid = False, test_loader = None,batch_size = 32, model_name = "novelty_gan"):
    model_name = model_name+"_best"
    f1s = []
    torch.cuda.empty_cache()
    disc.cuda()
    gen.cuda()
    disc.train()
    gen.train()
    # Losses & scores
    losses_g = []
    losses_d = []
    real_scores = []
    fake_scores = []
    best_val_f1 = -np.inf
    # Create optimizers
    opt_d = torch.optim.Adam(disc.parameters(), lr=lr_d, betas=(0.5, 0.999))
    opt_g = None
    if gen is not None and type(gen).__name__ != "RandomGenerator":
        opt_g = torch.optim.Adam(gen.parameters(), lr=lr_g, betas=(0.5, 0.999))
    
    for epoch in range(epochs):
        loss_g = 0
        loss_d = 0
        counter = 0
        real_score = 0
        fake_score = 0
        for batch in tqdm(train_loader):
            counter+=1
            
            # Train discriminator
            loss_d, real_score, fake_score = train_discriminator(disc,gen,batch_size,device,batch, opt_d)
            # Train generator
            if opt_g is not None:
                for j in range(1):#if counter % 1 == 0:#for j in range(5):
                    
                    loss_g = train_generator(gen,disc,batch_size,opt_g,batch, device)
            else:
                loss_g = 0
                
                
        losses_g.append(loss_g)
        losses_d.append(loss_d)
        real_scores.append(real_score)
        fake_scores.append(fake_score)
        
        print("Epoch [{}/{}], loss_g: {:.4f}, loss_d: {:.4f}, real_score: {:.4f}, fake_score: {:.4f}".format(
            epoch+1, epochs, loss_g, loss_d, real_score, fake_score))
        
    
        if run_on_valid:
            precision, recall,f1,score = models.validate_model(disc,test_loader)
            if f1>best_val_f1:
                  best_val_f1 = f1
                  torch.save(gen.state_dict(),os.path.join("./models",model_name+"_generator.model"))
                  torch.save(disc.state_dict(),os.path.join("./models",model_name+"_discriminator.model"))
                  file = open(os.path.join("./models",model_name+"_stats.json"),'w')
                  json.dump({"precision":precision,"recall":recall,"f1":f1},file)
                  file.close()
            f1s.append(f1)
        print("Epoch [{}/{}], loss_g: {:.4f}, loss_d: {:.4f}, real_score: {:.4f}, fake_score: {:.4f}, valid_precision: {:.4f}, valid_recall: {:.4f}, valid_F1: {:.4f}, valid_acc: {:.4f}".format(
            epoch+1, epochs, loss_g, loss_d, real_score, fake_score, precision,recall,f1, score))
    
    return losses_g, losses_d, real_scores, fake_scores, f1s

if __name__=="__main__":
    

    parser = argparse.ArgumentParser(description='Process some integers.')
    parser.add_argument('--config', 
                        help='sum the integers (default: find the max)')

    args = parser.parse_args()
    config = read_config(args.config)
    basepath = config["base_path"]
    print('Loading data')
    path = os.path.join(basepath,config["train_path"],"normal_graphs_maps.pkl")#'/media/panagiotis/TOSHIBA EXT1/Research/Novelty_detection/datasets/gridworlds_data/novelgridworlds_no_nov_1_easy/normal_graphs_maps.pkl'
    with open(path, 'rb') as f:
        graphs,maps = pickle.load(f)


    path = os.path.join(basepath,config["train_path"],"normal_nodeids.pkl")#'/media/panagiotis/TOSHIBA EXT1/Research/Novelty_detection/datasets/gridworlds_data/novelgridworlds_no_nov_1_easy/normal_nodeids.pkl'
    with open(path, 'rb') as f:
        node_ids = pickle.load(f)

    data = Dataset(graphs,maps)
    batch_size = config["batch_size"]

    train_loader = torch.utils.data.DataLoader(data, batch_size=batch_size, shuffle=True, collate_fn = my_collate)
    path = os.path.join(basepath,config["valid_path"],"valid_graphs_maps.pkl")#'/media/panagiotis/TOSHIBA EXT1/Research/Novelty_detection/datasets/gridworlds_data/validation/valid_graphs_maps.pkl'
    with open(path, 'rb') as f:
        nov_graphs,nov_maps = pickle.load(f)

    #path = '/media/panagiotis/TOSHIBA EXT1/Research/Novelty_detection/datasets/gridworlds_data/validation/valid_nodeids.pkl'
    path = os.path.join(basepath,config["valid_path"],"valid_nodeids.pkl")
    with open(path, 'rb') as f:
        nov_node_ids = pickle.load(f)

    ##################################################
    test_data = Dataset(nov_graphs,nov_maps, ids =  nov_node_ids)
    targets = test_data.make_targets_from_ids()
    test_data.y = targets
    valid_loader = torch.utils.data.DataLoader(test_data, batch_size=1, shuffle=False, collate_fn = my_collate)
    

    if config["model_save_path"] not in os.listdir("./"):
        print(os.listdir(basepath))
        os.mkdir(config["model_save_path"])

    #Initialize models and Fit. 

    model_predictor = 'agent'
    disc = models.MapGraphModel(50,16, model_predictor=model_predictor)
    device = "cuda"
    disc.to(device)
    gen =models.RandomGenerator(batch_size)#MapGraphNoveltyInjector(50,16)
    gen.to(device)


    #opt_g = torch.optim.Adam(gen.parameters(), lr = 1e-3, betas=(0.5, 0.999))
    epochs = 10

    f1s_replay = fit(disc, gen, train_loader,epochs, device = device,lr_d = 1e-4, lr_g = 1e-3, test_loader=valid_loader, run_on_valid= True,batch_size = batch_size, model_name=f"novelty_gan_{model_predictor}_test")[-1]


    #Load Best Model

    disc = models.MapGraphModel(50,16)
    device = "cuda"
    disc.to(device)
    gen = models.RandomGenerator(batch_size)#MapGraphNoveltyInjector(50,8)
    gen.to(device)
    
    gen.load_state_dict(torch.load(f"{config['model_save_path']}/novelty_gan_{config['model_predictor']}_test_best_generator.model"))

    disc.load_state_dict(torch.load(f"{config['model_save_path']}/novelty_gan_{config['model_predictor']}_test_best_discriminator.model"))
    gen.eval()
    disc.eval()



    print('Loading test data')
    path = os.path.join(basepath,config["test_path"],"test_graphs_maps.pkl")#'/media/panagiotis/TOSHIBA EXT1/Research/Novelty_detection/datasets/gridworlds_data/test/test_graphs_maps.pkl'
    with open(path, 'rb') as f:
        nov_graphs,nov_maps = pickle.load(f)


    path =os.path.join(basepath,config["test_path"],"test_nodeids.pkl") #'/media/panagiotis/TOSHIBA EXT1/Research/Novelty_detection/datasets/gridworlds_data/test/test_nodeids.pkl'
    with open(path, 'rb') as f:
        nov_node_ids = pickle.load(f)


    test_data = Dataset(nov_graphs,nov_maps, ids =  nov_node_ids)
    targets = test_data.make_targets_from_ids()
    test_data.y = targets
    test_loader = torch.utils.data.DataLoader(test_data, batch_size=1, shuffle=False, collate_fn = my_collate)

    p,r,f1,acc,X,y,pred= models.validate_model(disc,test_loader, return_data=True)

    print(p,r,f1,acc)