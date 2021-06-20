import os
import csv
import matplotlib.pyplot as plt



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
                      'left', 'right', 'kl_diversity', 'cosine_actions']
        
        self.writer = csv.DictWriter(self.csv_file, fieldnames=fieldnames)
        self.writer.writeheader()


    def log(self, text):
        
        self.txt_file.write(text+'\n')
        self.txt_file.flush()
        print(text)

    def log_performance(self, generation, reward, actor_loss, critic_loss, entropy_loss, diversity, loss, learning_rate,\
                        left, right, kl_diversity, cosine_actions, row):
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
                             'left': left,
                             'right': right,
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
        self.log('  left         |  ' + str(left))
        self.log('  right        |  ' + str(right))
        self.log('  kl diversity |  ' + str(kl_diversity))
        self.log('  cos actions  |  ' + str(cosine_actions))
        self.log('----------------------------------------')

    def plot(self, algorithm):

        plot(self.csv_path, self.fig_path, algorithm)
        

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

def plot_diversity(path, titel, num, xlabel, ylabel):

    xs, ys = [], []

    for i in range(num):
        csv_path = path +'/' + str(i) + '_performance.csv' 
        y = []
        with open(csv_path) as csvfile:
            print(csv_path)
            reader = csv.DictReader(csvfile)
            
            for row in reader:
                if i == 0:
                    xs.append(int(row['generation']))
                y.append(float(row['reward']))
        ys.append(y)

    ys1, ys2, ys3, ys4, ys5, ys6 = ys

    plt.plot(xs, ys1, c='b')
    plt.plot(xs, ys2, c='g')
    plt.plot(xs, ys3, c='r')
    plt.plot(xs, ys4, c='c')
    plt.plot(xs, ys5, c='m')
    plt.plot(xs, ys6, c='y')
    plt.xlabel(str(xlabel))
    plt.ylabel(str(ylabel))
    plt.axes().yaxis.grid()
    plt.yticks([-1,-0.75,-0.5,-0.25,0,0.25,0.5,0.75,1])
    plt.title(str(titel))
    plt.legend(['1e-4', '6e-5', '3e-5', '1e-5', '6e-6', '3e-6'])
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
    save_path = path + '/fig.png'
    plt.savefig(save_path)


