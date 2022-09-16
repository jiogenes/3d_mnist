import numpy as np

def preprocess(points, rotation_prob=0, flip_prob=0, shear_prob=0, scaling_prob=0, noise_prob=0):
    points = random_rotation(points, rotation_prob)
    points = random_flip(points, flip_prob)
    points = random_shear(points, shear_prob)
    points = random_scaling(points, scaling_prob)
    points = add_noise(points, noise_prob)
    return points

def random_rotation(points, prob=0.5):
        x, y, z = .0, .0, .0
        if np.random.rand() < prob:
            x = np.pi * np.random.uniform(-0.5, 0.5)
        if np.random.rand() < prob:
            y = np.pi * np.random.uniform(-0.5, 0.5)
        if np.random.rand() < prob:
            z = np.pi * np.random.uniform(-0.5, 0.5)
        m = _rotation(x, y, z)
        points = np.dot(points, m.T)
        return points

def random_flip(points, prob=0.5):
    x, y, z = .0, .0, .0
    if np.random.rand() < prob:
        x = np.pi
    if np.random.rand() < prob:
        y = np.pi
    if np.random.rand() < prob:
        z = np.pi
    m = _rotation(x, y, z)
    points = np.dot(points, m.T)
    return points

def random_shear(points, prob=0.5):
    y_x, z_x, x_y, z_y, x_z, y_z = .0, .0, .0, .0, .0, .0
    if np.random.rand() < prob:
        y_x = np.pi * np.random.uniform(-0.5, 0.5)
    if np.random.rand() < prob:
        z_x = np.pi * np.random.uniform(-0.5, 0.5)
    if np.random.rand() < prob:
        x_y = np.pi * np.random.uniform(-0.5, 0.5)
    if np.random.rand() < prob:
        z_y = np.pi * np.random.uniform(-0.5, 0.5)
    if np.random.rand() < prob:
        x_z = np.pi * np.random.uniform(-0.5, 0.5)
    if np.random.rand() < prob:
        y_z = np.pi * np.random.uniform(-0.5, 0.5)
    m = _shear(y_x, z_x, x_y, z_y, x_z, y_z)
    points = np.dot(points, m)
    return points
    

def random_scaling(points, prob):
    s_x, s_y, s_z = .0, .0, .0
    if np.random.rand() < prob:
        s_x = np.pi
    if np.random.rand() < prob:
        s_y = np.pi
    if np.random.rand() < prob:
        s_z = np.pi
    m = _scaling(s_x, s_y, s_z)
    points = np.dot(points, m)
    return points

def add_noise(points, prob=0.5):
    if np.random.rand() < prob:
        points += 0.05 * np.random.randn(*points.shape)
    return points

def _rotation(x, y, z):
    mx = np.array([[1, 0, 0], [0, np.cos(x), -np.sin(x)], [0, np.sin(x), np.cos(x)]])
    my = np.array([[np.cos(y), 0, np.sin(y)], [0, 1, 0], [-np.sin(y), 0, np.cos(y)]])
    mz = np.array([[np.cos(z), -np.sin(z), 0], [np.sin(z), np.cos(z), 0], [0, 0, 1]])
    rotation_matrix = np.dot(np.dot(mx, my), mz)
    return rotation_matrix

def _shear(y_x, z_x, x_y, z_y, x_z, y_z):
    shear_matrix = np.array([
        [1, y_x, z_x],
        [x_y, 1, z_y],
        [x_z, y_z, 1],
    ])
    return shear_matrix

def _scaling(s_x, s_y, s_z):
    scaling_matrix = np.array([
        [s_x, 0, 0],
        [0, s_y, 0],
        [0, 0, s_z],
    ])
    return scaling_matrix