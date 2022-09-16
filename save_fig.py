import h5py
import numpy as np
import plotly.graph_objs as go

def save_fig(points):
    for idx in range(50000, 90000):
        data = np.array(points[str(idx)])  
        layout = go.Layout(title = "[Test] Index :" + str(idx))
        plot_data = go.Scatter3d(x = data[:,0], y = data[:,1], z = data[:,2],
                            mode = 'markers', marker = dict(size = 1))
        fig = go.Figure(data = [plot_data], layout=layout)
        fig.write_image(f'./test_imgs/{idx}.png')
        print(f'save ./test_imgs/{idx}.png')

if __name__ == '__main__':
    test_points = h5py.File('/data/jyji/datasets/3D_NUMBER/test.h5', 'r')
    save_fig(test_points)