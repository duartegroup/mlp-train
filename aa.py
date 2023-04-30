import numpy as np
import mlptrain as mlt
import matplotlib.pyplot as plt

if __name__ == '__main__':

    data = np.load('h2_mace.npz', allow_pickle=True)
    time = np.array(range(len(data['E_predicted'])))

    fig, ax = plt.subplots()
    ax.plot(time, data['E_predicted'])
    fig.savefig('aa.pdf')