import os
import csv
import matplotlib.pyplot as plt


class Logger(object):
    ''' Logger saves the running results and helps make plots from the results
    '''

    def __init__(self, log_dir):
        ''' Initialize the labels, legend and paths of the plot and log file.

        Args:
            log_path (str): The path the log files
        '''
        self.log_dir = log_dir
        self.txt_path = os.path.join(log_dir, 'log.txt')
        self.csv_path = os.path.join(log_dir, 'performance.csv')
        self.fig_path = os.path.join(log_dir, 'fig.png')

        if not os.path.exists(log_dir):
            os.makedirs(log_dir)

        self.txt_file = open(self.txt_path, 'w')
        self.csv_file = open(self.csv_path, 'w')
        fieldnames = ['iteration', 'reward', 'loss', 'state', 'actions', 'predictions',
                      'q_values', 'current_q_values', 'expected_q_values']
        self.writer = csv.DictWriter(self.csv_file, fieldnames=fieldnames)
        self.writer.writeheader()

    def log(self, text):
        ''' Write the text to log file then print it.
        Args:
            text(string): text to log
        '''
        self.txt_file.write(text+'\n')
        self.txt_file.flush()
        print(text)

    """
    def log_performance(self, timestep, reward):
        ''' Log a point in the curve
        Args:
            timestep (int): the timestep of the current point
            reward (float): the reward of the current point
        '''
        self.writer.writerow({'timestep': timestep, 'reward': reward})
        print('')
        self.log('----------------------------------------')
        self.log('  timestep     |  ' + str(timestep))
        self.log('  reward       |  ' + str(reward))
        self.log('----------------------------------------')
    """

    def log_performance(self, iteration, reward, loss=None, states=None, actions=None, predictions=None, q_values=None, current_q=None,
                        expected_q=None):
        ''' Log a point in the curve
        Args:
            iteration (int): the iteration of the current point
            reward (float): the reward of the current point
            loss(float): the loss of the current point
            states(str): the raw_obs of the current point
            actions(int): integer of actions taken by the agent
            predictions(int): predicted action by the network
        '''
        self.writer.writerow({'iteration': iteration,
                              'reward': reward,
                              'loss': loss,
                              'state': states,
                              'actions': actions,
                              'predictions': predictions,
                              'q_values': q_values,
                              'current_q_values': current_q,
                              'expected_q_values': expected_q},

                             )
        print('')
        self.log('----------------------------------------')
        self.log('  iteration    |  ' + str(iteration))
        self.log('  reward       |  ' + str(reward))
        self.log('  loss         |  ' + str(loss))
        self.log('  state        |  ' + str(states))
        self.log('  actions      |  ' + str(actions))
        self.log('  predictions  |  ' + str(predictions))
        self.log('  q_values     |  ' + str(q_values))
        self.log('  current_q_values   |  ' + str(current_q))
        self.log('  expected_q_values  |  ' + str(expected_q))
        self.log('----------------------------------------')

    def plot(self, algorithm):
        plot(self.csv_path, self.fig_path, algorithm)

    def close_files(self):
        ''' Close the created file objects
        '''
        if self.txt_path is not None:
            self.txt_file.close()
        if self.csv_path is not None:
            self.csv_file.close()


def plot(csv_path, save_path, algorithm):
    ''' Read data from csv file and plot the results
    '''

    with open(csv_path) as csvfile:
        print(csv_path)
        reader = csv.DictReader(csvfile)
        xs = []
        ys = []
        for row in reader:
            xs.append(int(row['iteration']))
            ys.append(float(row['reward']))
        fig, ax = plt.subplots()
        ax.plot(xs, ys, label=algorithm)
        ax.set(xlabel='iteration', ylabel='reward')
        ax.legend()
        ax.grid()

        save_dir = os.path.dirname(save_path)
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)

        fig.savefig(save_path)
