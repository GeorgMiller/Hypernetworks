import os
import timeit
import os
import csv
import matplotlib.pyplot as plt
from datetime import datetime
import numpy as np

def plot():

    ddqn1_path = 'exp_new_5e6_dvd_0.001/0_performance.csv'
    ddqn2_path = 'experiments/1/ddqn_new_75e5/0_performance.csv'
    algorithm1a = 'Cosine loss'
    algorithm2a = 'DDQN reward'
    algorithm1b = 'DvD loss'
    algorithm2b = 'DDQN percentage right'

    with open(ddqn1_path) as csvfile:
        reader = csv.DictReader(csvfile)
        xs = []
        ys1 = []
        ys2 = []
        for row in reader:
            xs.append(int(row['generation']))
            ys1.append(float(row['cos']))
            ys2.append(float(row['dvd']))

        fig, ax = plt.subplots()
        ax2 = ax.twinx()

    with open(ddqn2_path) as csvfile:#
        reader = csv.DictReader(csvfile)
        zs1 = []
        zs2 = []
        for row in reader:
            zs1.append(float(row['reward']))
            zs2.append(float(row['dvd']))

    ax.plot(xs, ys1, label=algorithm1a, c='b')
    #ax.plot(xs, zs1, label=algorithm2a, c='g')

    ax2.plot(xs, ys2, label=algorithm1b, c='g')
    #ax2.plot(xs, zs2, label=algorithm2b, c='y')
    
    ax.set(xlabel='episode', ylabel='Cosine loss')
    ax2.set(ylim=(0,10), xlabel='episode', ylabel='DvD loss')


    ax.legend(loc='upper center', bbox_to_anchor=(0.2, -0.05),fancybox=True, shadow=True, ncol=5)
    ax2.legend(loc='upper center', bbox_to_anchor=(0.8, -0.05),fancybox=True, shadow=True, ncol=5)
    #ax.legend(loc='lower center')
    #ax2.legend(loc='lower right')
    ax.grid()
    #ax2.set_yticks([-5,0,5,10,15,20])

    ax.set_yticks([0,0.2,0.4,0.6,0.8,1])
    #ax2.grid()
    
    z1 = np.polyfit(xs, ys1, 10)
    p1 = np.poly1d(z1)
    #plt.plot(xs,p1(xs),"k--")

    z2 = np.polyfit(xs, ys2, 10)
    p2 = np.poly1d(z2)
    #plt.plot(xs,p2(xs),"k--")


    save_path = 'experiments' + '/Doudizhu_dvd_vs_cos_0001.png'
    plt.savefig(save_path)
    plt.show()


plot()