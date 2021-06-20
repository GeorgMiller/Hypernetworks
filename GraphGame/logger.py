import os
import csv
import matplotlib.pyplot as plt
import numpy as np


class Logger(object):

    def __init__(self, log_dir, row):
        
        self.log_dir = log_dir
        self.txt_path = os.path.join(log_dir, '{}_log.txt'.format(row))
        self.csv_path = os.path.join(log_dir, '{}_performance.csv'.format(row))
        self.fig_path = os.path.join(log_dir, '{}_fig.png'.format(row))

        if not os.path.exists(log_dir):
            os.makedirs(log_dir)

        self.txt_file = open(self.txt_path, 'w')
        self.csv_file = open(self.csv_path, 'w')
        
        fieldnames = ['generation', 'reward', 'actor_loss', 'critic_loss', 'entropy_loss', 'diversity', 'loss', 'learning_rate',\
                      'right_up', 'right_down', 'kl_diversity', 'cosine_actions']
        
        self.writer = csv.DictWriter(self.csv_file, fieldnames=fieldnames)
        self.writer.writeheader()


    def log(self, text):
        
        self.txt_file.write(text+'\n')
        self.txt_file.flush()
        print(text)

    def log_performance(self, generation, reward, actor_loss, critic_loss, entropy_loss, diversity, loss, learning_rate,\
                        right_up, right_down, kl_diversity, cosine_actions, row):
        ''' Log a point in the curve
        Args:
            timestep (int): the timestep of the current point
            reward (float): the reward of the current point
        '''
        self.writer.writerow({'generation': generation,
                             'reward': reward,
                             'actor_loss': actor_loss,
                             'critic_loss': critic_loss,
                             'entropy_loss': entropy_loss,
                             'diversity': diversity,
                             'loss': loss,
                             'learning_rate': learning_rate,
                             'right_up': right_up,
                             'right_down': right_down,
                             'kl_diversity': kl_diversity,
                             'cosine_actions': cosine_actions})
        print('')
        self.log('----------------------------------------')
        self.log('  generation   |  ' + str(generation))
        self.log('  reward       |  ' + str(reward))
        self.log('  actor loss   |  ' + str(actor_loss))
        self.log('  criic loss   |  ' + str(critic_loss))
        self.log('  entropy loss |  ' + str(entropy_loss))
        self.log('  diversity    |  ' + str(diversity))
        self.log('  loss         |  ' + str(loss))
        self.log('  learningrate |  ' + str(learning_rate))
        self.log('  right up     |  ' + str(right_up))
        self.log('  right down   |  ' + str(right_down))
        self.log('  kl diversity |  ' + str(kl_diversity))
        self.log('  cos actions  |  ' + str(cosine_actions))
        self.log('----------------------------------------')

    def plot(self, algorithm):

        plot(self.csv_path, self.fig_path, algorithm)
    
    def plot_diversity(self, algorithm):

        plot_diversity(self.csv_path, self.fig_path, algorithm)

    def close_files(self):
        
        if self.txt_path is not None:
            self.txt_file.close()
        if self.csv_path is not None:
            self.csv_file.close()

def plot(csv_path, save_path, algorithm):
    
    with open(csv_path) as csvfile:
        print(csv_path)
        reader = csv.DictReader(csvfile)
        xs = []
        ys1 = []
        ys2 = []
        for row in reader:
            xs.append(int(row['generation']))
            ys1.append(float(row['reward']))
            ys2.append(float(row['loss']))

        fig, ax = plt.subplots()
        ax2 = ax.twinx()

        ax.plot(xs, ys1, label=algorithm)
        ax2.plot(xs, ys2, label=algorithm)
        ax.set(xlabel='generation', ylabel='reward')
        ax2.set(xlabel='generation', ylabel='loss')
        ax.legend()
        ax2.legend()
        ax.grid()
        ax2.grid()

        save_dir = os.path.dirname(save_path)
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)

        fig.savefig(save_path)


def plot_diversity():

    path = 'experiments/hypernetwork_learning_rate_ddqn_'
    titel = 'hypernetwork DDQN'
    num = [2,3,4]
    xlabel = 'iterations'
    ylabel = 'cosine_actions'
    xs, ys = [], []

    for i in num:
        csv_path = path + str(i) + '/0_performance.csv' 
        y = []
        with open(csv_path) as csvfile:
            print(csv_path)
            reader = csv.DictReader(csvfile)
            
            for row in reader:
                if i == 2:
                    xs.append(int(row['generation']))
                y.append(float(row[ylabel]))
        ys.append(y)

    #ys1, ys2, ys3, ys4, ys5, ys6 = ys

    ys1, ys2, ys3 = ys

    plt.plot(xs, ys1, c='b')
    plt.plot(xs, ys2, c='g')
    plt.plot(xs, ys3, c='r')
    #plt.plot(xs, ys4, c='c')
    #plt.plot(xs, ys5, c='m')
    #plt.plot(xs, ys6, c='y')
    plt.xlabel(str(xlabel))
    plt.ylabel(str(ylabel))
    plt.axes().yaxis.grid()
    plt.yticks([-1,-0.75,-0.5,-0.25,0,0.25,0.5,0.75,1])
    plt.title(str(titel))
    plt.legend(['1e-4', '6e-5', '3e-5', '1e-5'])#, '6e-6', '3e-6'])
    '''
    fig, ax = plt.subplots()

    ax2 = ax.twinx()
    ax3 = ax.twinx()
    ax4 = ax.twinx()
    ax5 = ax.twinx()
    ax6 = ax.twinx()

    ax.plot(xs, ys1, label=algorithm, c='b')
    ax2.plot(xs, ys2, label=algorithm, c='g')
    ax3.plot(xs, ys3, label=algorithm, c='r')
    ax4.plot(xs, ys4, label=algorithm, c='c')
    ax5.plot(xs, ys5, label=algorithm, c='m')
    ax6.plot(xs, ys6, label=algorithm, c='y')

    ax.set(xlabel='steps', ylabel='1e-4')
    ax2.set(xlabel='steps', ylabel='6e-5')
    ax3.set(xlabel='steps', ylabel='3e-5')
    ax4.set(xlabel='steps', ylabel='1e-5')
    ax5.set(xlabel='steps', ylabel='6e-6')
    ax6.set(xlabel='steps', ylabel='3e-6')
    
    ax.legend()
    ax2.legend()
    ax3.legend()
    ax4.legend()
    ax5.legend()
    ax6.legend()
    ax.grid()
    ax2.grid()
    ax3.grid()
    ax4.grid()
    ax5.grid()
    ax6.grid()
    '''
    save_dir = os.path.dirname(path)
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    save_path = 'Graphgame/experiments' + '/hyper_ddqn_kernel.png'
    plt.savefig(save_path)

#plot_diversity()

def plot_DDQN_A2C():

    a2c_path = 'experiments/hypernetwork_weight_init_ddqn/0_performance.csv'
    ddqn_path = 'experiments/hypernetwork_weight_init_ddqn/1_performance.csv'
    algorithm1a = 'A2C reward'
    algorithm2a = 'DDQN reward'
    algorithm1b = 'A2C percentage right'
    algorithm2b = 'DDQN percentage right'

    with open(a2c_path) as csvfile:
        reader = csv.DictReader(csvfile)
        xs = []
        ys1 = []
        ys2 = []
        for row in reader:
            xs.append(int(row['generation']))
            ys1.append(float(row['reward']))
            ys2.append(float(row['right']))

        fig, ax = plt.subplots()
        ax2 = ax.twinx()

    with open(ddqn_path) as csvfile:
        reader = csv.DictReader(csvfile)
        zs1 = []
        zs2 = []
        for row in reader:
            zs1.append(float(row['reward']))
            zs2.append(float(row['right']))

    ax.plot(xs, ys1, label=algorithm1a, c='b')
    ax.plot(xs, zs1, label=algorithm2a, c='r')

    ax2.plot(xs, ys2, label=algorithm1b, c='g')
    ax2.plot(xs, zs2, label=algorithm2b, c='y')
    
    ax.set(xlabel='iteration', ylabel='reward')
    ax2.set(xlabel='iteration', ylabel='percentage right')
    ax.legend(loc='upper left')
    ax2.legend(loc='upper right')
    ax.grid()
    ax2.grid()
    save_path = 'Graphgame/experiments' + '/hyper_ddqn_kernel.png'
    plt.savefig(save_path)
    plt.show()



def plot_main(csv_path):

    fig, axs = plt.subplots(2, 2)
    axs[0, 0].plot(x, y)
    axs[0, 0].set_title('Axis [0, 0]')
    axs[0, 1].plot(x, y, 'tab:orange')
    axs[0, 1].set_title('Axis [0, 1]')
    axs[1, 0].plot(x, -y, 'tab:green')
    axs[1, 0].set_title('Axis [1, 0]')
    axs[1, 1].plot(x, -y, 'tab:red')
    axs[1, 1].set_title('Axis [1, 1]')

    for ax in axs.flat:
        ax.set(xlabel='x-label', ylabel='y-label')

    # Hide x labels and tick labels for top plots and y ticks for right plots.
    for ax in axs.flat:
        ax.label_outer()

def plot_main_2():

    pylab.figure(figsize=(18, 9))
    def PlotModel(self, score, episode):
        self.scores.append(score)
        self.episodes.append(episode)
        self.average.append(sum(self.scores[-50:]) / len(self.scores[-50:]))
        if str(episode)[-2:] == "00":# much faster than episode % 100
            pylab.plot(self.episodes, self.scores, 'b')
            pylab.plot(self.episodes, self.average, 'r')
            pylab.ylabel('Score', fontsize=18)
            pylab.xlabel('Steps', fontsize=18)
            try:
                pylab.savefig(self.path+".png")
            except OSError:
                pass

        return self.average[-1]




def plot_genetic():

    a2c_path = 'experiments_ga/0_performance.csv'
    ddqn_path = 'experiments_genetic/0_performance.csv'
    algorithm1a = 'GA reward'
    algorithm2a = 'ES reward'
    algorithm1b = 'GA percentage right'
    algorithm2b = 'ES percentage right'

    with open(a2c_path) as csvfile:
        reader = csv.DictReader(csvfile)
        xs = []
        ys1 = []
        ys2 = []
        i = 0
        for row in reader:
            if i % 10 == 0:
                
                xs.append(int(row['generation']))
                ys1.append(float(row['reward']))
                ys2.append(float(row['right_down']))
            i+=1
        fig, ax = plt.subplots()
        ax2 = ax.twinx()

    with open(ddqn_path) as csvfile:
        reader = csv.DictReader(csvfile)
        zs1 = []
        zs2 = []
        zs3 = []
        i = 0
        for row in reader:
            if i % 10 == 0:
                
                zs1.append(float(row['reward']))
                zs2.append(float(row['right_down']))
                zs3.append(float(row['right_up']))
            i +=1

    ax.plot(xs, ys1, label=algorithm1a, c='b')
    #ax.plot(xs, ys1, label=algorithm2a, c='b')

    ax2.plot(xs, ys2, label=algorithm1b, c='g')
    #ax2.plot(xs, ys2, label=algorithm2b, c='g')
    #ax2.plot(xs, zs3, label=algorithm2b, c='r')

    
    ax.set(xlabel='generation', ylabel='reward')
    ax2.set(xlabel='generation', ylabel='percentage right')
    #ax.legend(loc='upper left')
    #ax2.legend(loc='upper right')
    
    ax.legend(loc='upper center', bbox_to_anchor=(0.2, -0.05),fancybox=True, shadow=True, ncol=5)
    ax2.legend(loc='upper center', bbox_to_anchor=(0.8, -0.05),fancybox=True, shadow=True, ncol=5)
    ax.grid()
    z1 = np.polyfit(xs, ys1, 10)
    p1 = np.poly1d(z1)
    plt.plot(xs,p1(xs),"r--")

    z2 = np.polyfit(xs, ys2, 10)
    p2 = np.poly1d(z2)
    plt.plot(xs,p2(xs),"k--")
    #ax2.grid()
    save_path = 'experiments' + '/genetic_ga.png'
    plt.savefig(save_path)
    plt.show()


#plot_genetic()

def plot_hyper_DDQN_A2C():

    a2c_path = 'experiments/hypernetwork_weight_init_ddqn/0_performance.csv'
    ddqn_path = 'experiments/hypernetwork_weight_init_ddqn/1_performance.csv'
    algorithm1a = 'A2C reward'
    algorithm2a = 'DDQN reward'
    algorithm1b = 'A2C percentage right'
    algorithm2b = 'DDQN percentage right'

    with open(a2c_path) as csvfile:
        reader = csv.DictReader(csvfile)
        xs = []
        ys1 = []
        ys2 = []
        for row in reader:
            xs.append(int(row['generation']))
            ys1.append(float(row['reward']))
            ys2.append(float(row['right']))

        fig, ax = plt.subplots()
        ax2 = ax.twinx()

    with open(ddqn_path) as csvfile:
        reader = csv.DictReader(csvfile)
        zs1 = []
        zs2 = []
        for row in reader:
            zs1.append(float(row['reward']))
            zs2.append(float(row['right']))

    ax.plot(xs, ys1, label=algorithm1a, c='b')
    ax.plot(xs, zs1, label=algorithm2a, c='r')

    ax2.plot(xs, ys2, label=algorithm1b, c='g')
    ax2.plot(xs, zs2, label=algorithm2b, c='y')
    
    ax.set(xlabel='iteration', ylabel='reward')
    ax2.set(xlabel='iteration', ylabel='percentage right')
    ax.legend(loc='upper left')
    ax2.legend(loc='upper right')
    ax.grid()
    ax2.grid()
    save_path = 'Graphgame/experiments' + '/hyper_ddqn_kernel.png'
    plt.savefig(save_path)
    plt.show()


def plot_DDQN_cos():

    a2c_path = 'experiments/NEW_hypernetwork_learning_rate_ddqn_cos_lambdas_batch_size_2/1_performance.csv'
    ddqn_path = 'experiments/NEW_hypernetwork_learning_rate_ddqn_cos_lambdas_batch_size_2/1_performance.csv'
    algorithm1a = 'reward A2C' ############
    algorithm2a = 'reward DDQN' #########
    algorithm1b = 'percentage right' ########
    algorithm2b = 'percentage right'  ##########

    with open(a2c_path) as csvfile:
        reader = csv.DictReader(csvfile)
        xs = []
        ys1 = []
        ys2 = []
        for row in reader:
            xs.append(int(row['generation']))
            ys1.append(float(row['reward']))
            ys2.append(float(row['cosine_actions']))

        fig, ax = plt.subplots()
        ax2 = ax.twinx()

    with open(ddqn_path) as csvfile:
        reader = csv.DictReader(csvfile)
        zs1 = []
        zs2 = []
        for row in reader:
            zs1.append(float(row['reward']))
            zs2.append(float(row['cosine_actions']))

    #ax.plot(xs[:550], ys1[:550], label=algorithm1a, c='r')
    ax.plot(xs, zs1, label=algorithm2a, c='b')

    #ax2.plot(xs, ys2, label=algorithm1b, c='g')
    ax2.plot(xs, zs2, label=algorithm2b, c='g')
    
    ax.set(xlabel='iteration', ylabel='reward')
    ax2.set(xlabel='iteration', ylabel='percentage right')
    ax.legend(loc='upper left')
    ax2.legend(loc='upper left', bbox_to_anchor=(0, 0.9))
    ax.grid()
    #ax2.grid()
    save_path = 'experiments' + '/hypernetwork_cos_1.png'
    plt.savefig(save_path)
    plt.show()

plot_DDQN_cos()

def plot_DDQN_dvd():

    a2c_path = 'experiments/hypernetwork_learning_rate_ddqn_dvd_3_lambdas_batch_size_8/3_performance.csv'
    ddqn_path = 'experiments/hypernetwork_learning_rate_ddqn_dvd_3_lambdas_batch_size_2/0_performance.csv'
    algorithm1a = 'DvD loss lambda=0.05' ############
    algorithm2a = '' #########
    algorithm1b = 'percentage right' ########
    algorithm2b = 'percentage right'  ##########

    with open(a2c_path) as csvfile:
        reader = csv.DictReader(csvfile)
        xs = []
        ys1 = []
        ys2 = []
        for row in reader:
            xs.append(int(row['generation']))
            ys1.append(float(row['kl_diversity']))
            ys2.append(float(row['right']))

        fig, ax = plt.subplots()
        ax2 = ax.twinx()

    with open(ddqn_path) as csvfile:
        reader = csv.DictReader(csvfile)
        zs1 = []
        zs2 = []
        for row in reader:
            zs1.append(float(row['reward']))
            zs2.append(float(row['right']))

    ax.plot(xs, ys1, label=algorithm1a, c='b')
    #ax.plot(xs, zs1, label=algorithm2a, c='r')

    ax2.plot(xs, ys2, label=algorithm1b, c='g')
    #ax2.plot(xs, zs2, label=algorithm2b, c='y')
    
    ax.set(xlabel='iteration', ylabel='DvD loss')
    ax2.set(xlabel='iteration', ylabel='percentage right')
    #ax.legend(loc='upper left', bbox_to_anchor=(0, 0.8))
    #ax2.legend(loc='upper left',bbox_to_anchor=(0, 0.9))

    ax.legend(loc='upper center', bbox_to_anchor=(0.2, -0.05),fancybox=True, shadow=True, ncol=5)
    ax2.legend(loc='upper center', bbox_to_anchor=(0.8, -0.05),fancybox=True, shadow=True, ncol=5)
    ax.grid()
    #ax2.grid()
    save_path = 'experiments' + '/hypernetwork_ddqn_dvd_diversity_loss_batch_8.png'
    plt.savefig(save_path)
    plt.show()

#plot_DDQN_dvd()

def plot_difference():

    a2c_path = 'experiments/hypernetwork_learning_rate_ddqn_dvd_3_lambdas/1_performance.csv'
    ddqn_path = 'experiments/hypernetwork_learning_rate_ddqn_cos/0_performance.csv'
    algorithm1a = 'DvD loss' ############
    algorithm2a = '' #########
    algorithm1b = 'Cosine loss' ########
    algorithm2b = 'percentage right lambda=0.2'  ##########

    with open(a2c_path) as csvfile:
        reader = csv.DictReader(csvfile)
        xs = []
        ys1 = []
        ys2 = []
        for row in reader:
            xs.append(int(row['generation']))
            ys1.append(float(row['kl_diversity']))
            ys2.append(float(row['cosine_actions']))

        fig, ax = plt.subplots()
        ax2 = ax.twinx()

    with open(ddqn_path) as csvfile:
        reader = csv.DictReader(csvfile)
        zs1 = []
        zs2 = []
        for row in reader:
            zs1.append(float(row['kl_diversity']))
            zs2.append(float(row['cosine_actions']))

    ax.plot(xs, ys1, label=algorithm1a, c='b')
    #ax.plot(xs, zs1, label=algorithm2a, c='r')

    ax2.plot(xs, ys2, label=algorithm1b, c='g')
    #ax2.plot(xs, zs2, label=algorithm2b, c='y')
    
    ax.set(xlabel='iteration', ylabel='DvD loss')
    ax2.set(xlabel='iteration', ylabel='cosine loss')
    #ax.legend(loc='upper left', bbox_to_anchor=(0, 0.8))
    #ax2.legend(loc='upper left',bbox_to_anchor=(0, 0.9))

    ax.legend(loc='upper center', bbox_to_anchor=(0.2, -0.05),fancybox=True, shadow=True, ncol=5)
    ax2.legend(loc='upper center', bbox_to_anchor=(0.8, -0.05),fancybox=True, shadow=True, ncol=5)
    ax.grid()
    #ax2.grid()
    save_path = 'experiments' + '/hypernetwork_ddqn_difference.png'
    plt.savefig(save_path)
    plt.show()

#plot_difference()

def plot_cos_dvd_reward():

    a2c_path = 'experiments/hypernetwork_weight_init_ddqn/0_performance.csv'
    ddqn_path = 'experiments/hypernetwork_weight_init_ddqn/1_performance.csv'
    algorithm1a = 'dvd loss lambda=0.2' ############
    algorithm2a = 'dvd loss lambda=0.3' #########
    algorithm1b = 'percentage right lambda=0.2' ########
    algorithm2b = 'percentage right lambda=0.2'  ##########

    with open(a2c_path) as csvfile:
        reader = csv.DictReader(csvfile)
        xs = []
        ys1 = []
        ys2 = []
        for row in reader:
            xs.append(int(row['generation']))
            ys1.append(float(row['kl_diversity']))
            ys2.append(float(row['right']))

        fig, ax = plt.subplots()
        ax2 = ax.twinx()

    with open(ddqn_path) as csvfile:
        reader = csv.DictReader(csvfile)
        zs1 = []
        zs2 = []
        for row in reader:
            zs1.append(float(row['kl_diversity']))
            zs2.append(float(row['right']))

    ax.plot(xs, ys1, label=algorithm1a, c='b')
    ax.plot(xs, zs1, label=algorithm2a, c='r')

    ax2.plot(xs, ys2, label=algorithm1b, c='g')
    ax2.plot(xs, zs2, label=algorithm2b, c='y')
    
    ax.set(xlabel='iteration', ylabel='dvd loss')
    ax2.set(xlabel='iteration', ylabel='percentage right')
    ax.legend(loc='upper left')
    ax2.legend(loc='upper right')
    ax.grid()
    ax2.grid()
    save_path = 'Graphgame/experiments' + '/hyper_ddqn_kernel.png'
    plt.savefig(save_path)
    plt.show()


#plot_DDQN_A2C()
#plot_hyper_DDQN_A2C()
#plot_DDQN_cos()
#plot_DDQN_dvd()

#plot_cos_dvd_loss()
#plot_cos_dvd_reward()