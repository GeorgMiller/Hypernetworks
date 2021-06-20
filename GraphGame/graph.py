import numpy as np 

class Graph_Game:

    def __init__(self, size=100, final_states = [[99,99]]):

        self.final_states = final_states
        self.size = size
        self.all_nodes = []
        self.directions = [[1,0], [0,1], [-1,0], [0,-1]]
        self.step = 0
        self.hole_min = 4
        self.hole_max = 4
        self.max_steps = 30

    def build(self):

        for x in range(8):
            for y in range(8):
                self.all_nodes.append([x,y])
        

    def neighbors(self, node):

        result = []

        for direction in self.directions:
            neighbor = [node[0] + direction[0], node[1] + direction[1]]
            if 0 <= neighbor[0] < self.size and 0 <= neighbor[1] < self.size:

                if self.hole_min < neighbor[0] < self.hole_max and self.hole_min < neighbor[1] < self.hole_max:
                    pass
                    
                else:
                    result.append(neighbor)                        

        return result

    def hot_encoding(self, state):

        state_x = np.zeros([7])
        state_y = np.zeros([7])
        state_x[state[0]] = 1
        state_y[state[1]] = 1
        state = np.concatenate([state_x,state_y])
        state = np.reshape(state, (1,-1))

        return state

    def start(self):

        self.state = self.all_nodes[9]
        
        return self.hot_encoding(self.state)

    def next(self, action):
        
        self.step += 1
        done = False
        reward = - 0.005
        direction = self.directions[action]
        next_state = [self.state[0] + direction[0], self.state[1] + direction[1]]
        neighbor = self.neighbors(self.state)

        if next_state in neighbor:
            self.state = next_state
            if self.state in self.final_states:
                reward = 1
                done = True
                self.step = 0

        if self.step == self.max_steps:
            reward -= 1
            done = True
            self.step = 0

        return self.hot_encoding(self.state), reward, done

