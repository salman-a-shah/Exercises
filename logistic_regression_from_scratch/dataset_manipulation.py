import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


def build_dataset():
    mean_1 = (3, 3)
    mean_2 = (0, 0)
    cov = [[1, 0], [0, 1]]
    
    class_1 = np.random.multivariate_normal(mean_1, cov, 100)
    class_2 = np.random.multivariate_normal(mean_2, cov, 100)
    
    df1 = pd.DataFrame({
            'x' : class_1[:,0],
            'y' : class_1[:,1],
            'target' : 0
            })
    
    df2 = pd.DataFrame({
            'x' : class_2[:,0],
            'y' : class_2[:,1],
            'target' : 1
            })
    dataset = pd.concat([df1, df2], ignore_index=True)
    dataset.to_csv("toy_dataset.csv", index=False)
    
def visualize_dataset(dataset):
    plt.scatter(dataset.x, dataset.y, c=dataset.target, cmap='bwr')
    plt.show()
    
def visualize_decision_boundary(dataset, decision_function):
    
    def _compute_z(decision_function, grid):
        result = np.zeros((50,50))
        for i in range(50):
            for j in range(50):
                x = grid[0][i][j]
                y = grid[1][i][j]
                result[i][j] = decision_function(x, y)
        return result

    class_1 = dataset[dataset.target == 0].iloc[:,:-1].values
    class_2 = dataset[dataset.target == 1].iloc[:,:-1].values
    f, ax = plt.subplots(figsize=(7, 7))
    ax.scatter(*class_1.T)
    ax.scatter(*class_2.T)
    ax.set_autoscale_on(False)
    grid = np.array(np.meshgrid(
                        np.linspace(*ax.get_xlim()),
                        np.linspace(*ax.get_ylim())
                    ))
    z = _compute_z(decision_function, grid)
    ax.contour(grid[0], grid[1], z, levels=[0], cmap="Greys_r")
    plt.plot()