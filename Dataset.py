import torch
import numpy as np
import dgl

def my_collate(batch):
    
    graphs1 = []
    maps1 = []
    
    graphs2 = []
    maps2 = []
    
    target = []
    for item in batch:

        step, y = item
        g1,mp1,g2,mp2 = step
       

        target.append(y)
        graphs1.append(g1)
        maps1.append(mp1)
        

        graphs2.append(g2)
        maps2.append(mp2)
        
    
    
    graphs1 = dgl.batch(graphs1)
    graphs2 = dgl.batch(graphs2)

    if(type(maps1[0]).__name__ == 'Tensor'):
      maps1 = torch.cat([torch.unsqueeze(i,0) for i in maps1])
      maps2 = torch.cat([torch.unsqueeze(i,0) for i in maps2])
      

    else:
      maps1 = torch.Tensor(maps1)
      maps2 = torch.Tensor(maps2)
    
    #data = [item[0] for item in batch]
    #target = [item[1] for item in batch]
    #target = torch.LongTensor(target)

    return [graphs1,maps1,graphs2,maps2, target]

class Dataset(torch.utils.data.Dataset):
  'Characterizes a dataset for PyTorch'
  def __init__(self, graphs, maps,  y = None, ids= None):
        'Initialization'
        self.graphs = graphs
        self.maps = maps
        
        id_pairs = []
        
        step_pairs = []
        
        for i in range(len(graphs)):
            for j in range(len(graphs[i])-1):
                step_pairs.append(([graphs[i][j],maps[i][j],graphs[i][j+1],maps[i][j+1]]))
                if(ids is not None):
                      id_pairs.append((ids[i][j],ids[i][j+1]))
              
        self.step_pairs = step_pairs
        self.id_pairs = id_pairs

        if y is None:
            self.y = torch.ones((len(step_pairs)))
        elif y ==0:
            self.y = torch.zeros((len(step_pairs)))  


        
  
  def __len__(self):
        'Denotes the total number of samples'
        return len(self.step_pairs)

  def __getitem__(self, index):
        'Generates one sample of data'
        # Select sample
        step_pair = self.step_pairs[index]
        #print(len(self.y), len(self.step_pairs), index)
        y = self.y[index]

        return step_pair,y

  def make_targets_from_ids(self):
        y = []
        for pair in self.id_pairs:
            a,b = pair
            if "fence" in a.keys() or "fence" in b.keys() or "axe" in a.keys() or "axe" in b.keys():
                  y.append(0)
            else:
                  y.append(1)
        return y


  def modify_map(self, mp, ids):
        open_positions = np.where(mp == -1)
        nodes=  list(ids.values())
        for i in range(len(open_positions[0])):
              
              if np.random.randn()<0.001:
                    mp[open_positions[0][i],open_positions[1][i]] = torch.tensor(np.random.choice([0]+list(range(18,26,1)),1 ))
                    
        return mp

  def sample_random_pairs(self, n = 10, ids = None):
        size = len(self.step_pairs)
        samples =[]
        for i in range(n):
            idx = np.random.randint(size)
            pair1 = self.__getitem__(idx)

            #mod_map =self.modify_map( pair1[0][0][1], ids)
            #pair1 = (pair1[0][0][0],mod_map)

            idx = np.random.randint(size)
            pair2 = self.__getitem__(idx)

            pair = (pair1[0][0],pair2[0][1])
            samples.append((pair,0))
      
         
        return my_collate(samples)