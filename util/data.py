import torch
from torch.utils.data import Dataset
import numpy as np

from .preprocess import preprocess

class CustomDataset(Dataset):
    def __init__(self, id_list, label_list, point_list, mode='train'):
        self.mode = mode
        assert mode =='train' or mode =='val'
        self.id_list = id_list
        self.label_list = label_list
        self.point_list = point_list
        
    def __getitem__(self, index):
        image_id = self.id_list[index]
        
        points = self.point_list[str(image_id)][:]
        image = self.get_vector(points)
        
        if self.label_list is not None:
            label = self.label_list[index]
            return torch.Tensor(image).unsqueeze(0), label
        else:
            return torch.Tensor(image).unsqueeze(0)

    def __len__(self):
        return len(self.id_list)
    
    def get_vector(self, points, x_y_z=[16, 16, 16]):
        if self.mode == 'train':
            points = preprocess(points, 
                                rotation_prob=0.25, 
                                flip_prob=0.25, 
                                shear_prob=0.0, 
                                scaling_prob=0.25, 
                                noise_prob=0.25)
        
        xyzmin = np.min(points, axis=0) - 0.001
        xyzmax = np.max(points, axis=0) + 0.001

        diff = max(xyzmax-xyzmin) - (xyzmax-xyzmin)
        xyzmin = xyzmin - diff / 2
        xyzmax = xyzmax + diff / 2

        segments = []
        shape = []

        for i in range(3):
            if type(x_y_z[i]) is not int:
                raise TypeError("x_y_z[{}] must be int".format(i))
            s, step = np.linspace(xyzmin[i], xyzmax[i], num=(x_y_z[i] + 1), retstep=True)
            segments.append(s)
            shape.append(step)

        n_voxels = x_y_z[0] * x_y_z[1] * x_y_z[2]
        n_x = x_y_z[0]
        n_y = x_y_z[1]
        n_z = x_y_z[2]

        structure = np.zeros((len(points), 4), dtype=int)
        structure[:,0] = np.searchsorted(segments[0], points[:,0]) - 1
        structure[:,1] = np.searchsorted(segments[1], points[:,1]) - 1
        structure[:,2] = np.searchsorted(segments[2], points[:,2]) - 1
        structure[:,3] = ((structure[:,1] * n_x) + structure[:,0]) + (structure[:,2] * (n_x * n_y)) 

        vector = np.zeros(n_voxels)
        count = np.bincount(structure[:, 3])
        
        # normalize시 학습 안되는 현상 발생
        # if self.normalize:
        #     count = count.astype(float)
        #     count /= count.max()

        count = np.clip(count, 0, 1)

        vector[:len(count)] = count
        vector = vector.reshape(n_z, n_y, n_x)
        return vector