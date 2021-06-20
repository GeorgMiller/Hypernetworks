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
        
        fieldnames = ['generation', 'reward', 'policy_loss', 'value_loss', 'lamBda', 'loss', 'learning_rate',\
                      'dvd', 'cos']
        
        self.writer = csv.DictWriter(self.csv_file, fieldnames=fieldnames)
        self.writer.writeheader()


    def log(self, text):
        
        self.txt_file.write(text+'\n')
        self.txt_file.flush()
        print(text)

    def log_performance(self, generation, reward, policy_loss, value_loss, lamBda, loss, learning_rate,\
                        dvd, cos, row):
        ''' Log a point in the curve
        Args:
            timestep (int): the timestep of the current point
            reward (float): the reward of the current point
        '''
        self.writer.writerow({'generation': generation,
                             'reward': reward,
                             'policy_loss': policy_loss,
                             'value_loss': value_loss,
                             'lamBda': lamBda,
                             'loss': loss,
                             'learning_rate': learning_rate,
                             'dvd': dvd,
                             'cos': cos})
        print('')
        self.log('----------------------------------------')
        self.log('  generation   |  ' + str(generation))
        self.log('  reward       |  ' + str(reward))
        self.log('  policy loss  |  ' + str(policy_loss))
        self.log('  value loss   |  ' + str(value_loss))
        self.log('  lamBda       |  ' + str(lamBda))
        self.log('  loss         |  ' + str(loss))
        self.log('  learningrate |  ' + str(learning_rate))
        self.log('  dvd          |  ' + str(dvd))
        self.log('  cos          |  ' + str(cos))
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

