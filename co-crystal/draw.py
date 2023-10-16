import matplotlib.pyplot as plt
import torch
import numpy as np

def plot_loss(n):
    enc = np.load(r'D:\code\GMPNN-CS-master\GMPNN-CS-master\Auc.npy')
    # enc = torch.load('D:\MobileNet_v1\plan1-AddsingleLayer\loss\epoch_{}'.format(i))
    y = list(enc)
    x = range(0,len(y))
    plt.plot(x, y, '.-')
    plt_title = 'auc'
    plt.title(plt_title)
    plt.xlabel('100 epoch')
    plt.ylabel('Auc')
    plt.savefig('Auc.png')
    plt.show()

if __name__ == "__main__":
    plot_loss(20)