import numpy as np
import os
import math
import torch
from DRL_Code.env.net_middle import Net
from DRL_Code.env.reward import reward_fun
from DRL_Code.env import *

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Env_8conv.py device is", device)

class env():
    """ Environment class for simulating movements on a grid with neural network-based decision making. """

    def __init__(self,  start=None, m1=None, m2=None, M=20, N=20,):
        self.M = M  # Number of rows in the grid
        self.N = N  # Number of columns in the grid
        self.action_space = 2  # Number of possible actions (left/right and up/down)

        self.start = start  # Start position x-coordinate
        self.m1 = m1  # Must-visit point 1 y-coordinate
        self.m2 = m2  # Must-visit point 2 y-coordinate

        # Load pre-trained weights for Q_model
        self.Q_model = Net()

        dir_path = os.path.dirname(os.path.realpath(__file__))

        Q_model_checkpoint_path = os.path.join(dir_path, "Q_Net.pt")
        Q_model_checkpoint = torch.load(Q_model_checkpoint_path, map_location=device).state_dict()
        self.Q_model.load_state_dict(Q_model_checkpoint)
        self.Q_model.to(device)

        self.state = "arc"  # Initial state of movement
        self.Q_norm = 1e6  # Normalization factor for Q


    def reset(self):
        """ Resets the environment to start a new episode. """
        # Initialize position markers and state indicators
        self.passed_must_visit_point1 = False
        self.passed_must_visit_point2 = False
        self.passed_end_point = False


        # Setting initial grid positions
        start_point_x = self.start
        m1 = self.m1
        m2 = self.m2
        self.start_point = np.array([start_point_x, 0])
        self.end_point = np.array([start_point_x, self.N - 1])
        self.must_visit_point_1 = np.array([0, m1])
        self.must_visit_point_2 = np.array([self.M - 1, m2])

        self.start_point_hg = np.zeros((self.M, self.N))
        self.end_point_hg = np.zeros((self.M, self.N))
        self.must_visit_point_1_hg = np.zeros((self.M, self.N))
        self.must_visit_point_2_hg = np.zeros((self.M, self.N))
        self.current_point_hg = np.zeros((self.M, self.N))

        # Set starting position in binary matrices
        self.start_point_hg[self.start_point[0], self.start_point[1]] = 1
        self.current_point_hg[self.start_point[0], self.start_point[1]] = 1
        self.end_point_hg[self.end_point[0], self.end_point[1]] = 1
        self.must_visit_point_1_hg[self.must_visit_point_1[0], self.must_visit_point_1[1]] = 1
        self.must_visit_point_2_hg[self.must_visit_point_2[0], self.must_visit_point_2[1]] = 1

        self.binary_matrix = np.zeros((self.M, self.N))
        self.binary_matrix_arc = np.zeros((self.M, self.N))
        self.binary_matrix_rect = np.zeros((self.M, self.N))
        self.binary_matrix[self.start_point[0], self.start_point[1]] = 1

        self.current_position = np.copy(self.start_point)

        # Set the initial state type for movement
        if self.state == "arc":
            self.binary_matrix_arc[self.current_position[0]][self.current_position[1]] = 1
        if self.state == "rect":
            self.binary_matrix_rect[self.current_position[0]][self.current_position[1]] = 1


        # Mirror and concatenate matrices for creating symmetric patterns
        matrix_half_trace_flip = np.flip(self.binary_matrix, axis=0)
        matrix_trace = np.concatenate((self.binary_matrix, matrix_half_trace_flip), axis=0)
        matrix_half_arc_flip = np.flip(self.binary_matrix_arc, axis=0)
        matrix_trace_arc = np.concatenate((self.binary_matrix_arc, matrix_half_arc_flip), axis=0)
        matrix_half_trace_flip = np.flip(self.binary_matrix_rect, axis=0)
        matrix_trace_rect = np.concatenate((self.binary_matrix_rect, matrix_half_trace_flip), axis=0)

        start_point_hg_flip = np.flip(self.start_point_hg, axis=0)
        start_point_g = np.concatenate((self.start_point_hg, start_point_hg_flip), axis=0)
        end_point_hg_flip = np.flip(self.end_point_hg, axis=0)
        end_point_g = np.concatenate((self.end_point_hg, end_point_hg_flip), axis=0)

        current_point_hg_flip = np.flip(self.current_point_hg, axis=0)
        current_point_g = np.concatenate((self.current_point_hg, current_point_hg_flip), axis=0)

        must_visit_point_1_hg_flip = np.flip(self.must_visit_point_1_hg, axis=0)
        must_visit_point_1_g = np.concatenate((self.must_visit_point_1_hg, must_visit_point_1_hg_flip), axis=0)
        must_visit_point_2_hg_flip = np.flip(self.must_visit_point_2_hg, axis=0)
        must_visit_point_2_g = np.concatenate((self.must_visit_point_2_hg, must_visit_point_2_hg_flip), axis=0)

        # Stack all observation layers
        obs = np.stack([matrix_trace, matrix_trace_arc, matrix_trace_rect, current_point_g,
                        start_point_g, end_point_g, must_visit_point_1_g, must_visit_point_2_g], axis=0)

        obs = np.expand_dims(obs, axis=0)

        return obs

    def position_bound(self):
        """ Ensure the current position is within grid boundaries. """
        if self.current_position[0] < 0:
            self.current_position[0] = 0
        if self.current_position[0] > self.M - 1:
            self.current_position[0] = self.M - 1
        if self.current_position[1] < 0:
            self.current_position[1] = 0
        if self.current_position[1] > self.N - 1:
            self.current_position[1] = self.N - 1

    def deterministic_direction(self, must_visit, action):
        """ Determine the direction of movement based on the current action and target location. """
        random = action
        if random == 1:
            if must_visit[0] > self.current_position[0]:
                direction = "left"
            elif must_visit[0] < self.current_position[0]:
                direction = "right"
            elif must_visit[1] > self.current_position[1]:
                direction = "up"
            elif must_visit[1] < self.current_position[1]:
                direction = "down"
            else:
                direction = None
        else:
            if must_visit[1] > self.current_position[1]:
                direction = "up"
            elif must_visit[1] < self.current_position[1]:
                direction = "down"
            elif must_visit[0] > self.current_position[0]:
                direction = "left"
            elif must_visit[0] < self.current_position[0]:
                direction = "right"
            else:
                direction = None
        return direction

    def step(self, action):
        """ Process a step in the environment given an action. """
        self.last_position = np.copy(self.current_position)

        # Determine the direction based on the state of must-visit points and the end point
        if self.passed_must_visit_point1 == False:
            direction = self.deterministic_direction(self.must_visit_point_1, action)
        elif self.passed_must_visit_point1 == True and self.passed_must_visit_point2 == False:
            direction = self.deterministic_direction(self.must_visit_point_2, action)
        elif self.passed_must_visit_point1 == True and self.passed_must_visit_point2 == True and self.passed_end_point == False:
            direction = self.deterministic_direction(self.end_point, action)
        else:
            direction = "None"

        # Update position based on the direction
        if direction == "up":
            self.current_position[1] += 1
            self.state = "rect"
        elif direction == "down":
            self.current_position[1] -= 1
            self.state = "rect"
        elif direction == "left":
            self.current_position[0] += 1
            self.state = "arc"
        elif direction == "right":
            self.current_position[0] -= 1
            self.state = "arc"
        else:
            self.state = "arc"

        # Ensure the position is within bounds
        self.position_bound()

        # Update the status of passing through points
        if np.array_equal(self.current_position, self.must_visit_point_1):
            self.passed_must_visit_point1 = True
        if np.array_equal(self.current_position, self.must_visit_point_2):
            self.passed_must_visit_point2 = True
        if np.array_equal(self.current_position, self.end_point):
            self.passed_end_point = True

        # Update the binary matrices
        self.binary_matrix[self.current_position[0]][self.current_position[1]] = 1
        if self.state == "arc":
            self.binary_matrix_arc[self.current_position[0]][self.current_position[1]] = 1
            if self.binary_matrix_rect[self.last_position[0]][self.last_position[1]] == 1:
                self.binary_matrix_arc[self.last_position[0]][self.last_position[1]] = 1
                self.binary_matrix_rect[self.last_position[0]][self.last_position[1]] = 0
        if self.state == "rect":
            self.binary_matrix_rect[self.current_position[0]][self.current_position[1]] = 1

        # Mirror and concatenate binary matrices
        matrix_half_trace_flip = np.flip(self.binary_matrix, axis=0)
        matrix_trace = np.concatenate((self.binary_matrix, matrix_half_trace_flip), axis=0)
        matrix_half_arc_flip = np.flip(self.binary_matrix_arc, axis=0)
        matrix_trace_arc = np.concatenate((self.binary_matrix_arc, matrix_half_arc_flip), axis=0)
        matrix_half_trace_flip = np.flip(self.binary_matrix_rect, axis=0)
        matrix_trace_rect = np.concatenate((self.binary_matrix_rect, matrix_half_trace_flip), axis=0)

        self.current_point_hg = np.zeros((self.M, self.N))
        self.current_point_hg[self.current_position[0], self.current_position[1]] = 1
        current_point_hg_flip = np.flip(self.current_point_hg, axis=0)
        current_point_g = np.concatenate((self.current_point_hg, current_point_hg_flip), axis=0)

        start_point_hg_flip = np.flip(self.start_point_hg, axis=0)
        start_point_g = np.concatenate((self.start_point_hg, start_point_hg_flip), axis=0)
        end_point_hg_flip = np.flip(self.end_point_hg, axis=0)
        end_point_g = np.concatenate((self.end_point_hg, end_point_hg_flip), axis=0)
        must_visit_point_1_hg_flip = np.flip(self.must_visit_point_1_hg, axis=0)
        must_visit_point_1_g = np.concatenate((self.must_visit_point_1_hg, must_visit_point_1_hg_flip), axis=0)
        must_visit_point_2_hg_flip = np.flip(self.must_visit_point_2_hg, axis=0)
        must_visit_point_2_g = np.concatenate((self.must_visit_point_2_hg, must_visit_point_2_hg_flip), axis=0)

        # input for surrogate model
        obs_3conv = np.stack([matrix_trace, matrix_trace_arc, matrix_trace_rect], axis=0)

        # input for RL agent
        obs = np.stack([matrix_trace, matrix_trace_arc, matrix_trace_rect, current_point_g,
                        start_point_g, end_point_g, must_visit_point_1_g, must_visit_point_2_g], axis=0)

        # Expand dimensions to add batch size of 1
        obs_3conv = np.expand_dims(obs_3conv, axis=0)
        obs = np.expand_dims(obs, axis=0)

        # Determine if the episode is done
        if self.passed_must_visit_point1 == True and self.passed_must_visit_point2 == True and self.passed_end_point == True:
            done = 1
        else:
            done = 0

        # Calculate reward and handle terminal state
        if done == 1:
            self.Q_model.eval()

            with torch.no_grad():
                input = torch.Tensor(obs_3conv).to(device)
                self.reward_Q = self.Q_model(input).cpu().detach().numpy()[0][0]
                self.Q = self.reward_Q * self.Q_norm

            if self.Q <= 0:
                error_flag = 1
            else:
                error_flag = 0

            reward = reward_fun(Q=self.Q, error_flag=error_flag)  # Calculate reward based on performance metrics

        else:
            reward = 0.0   # No reward if the episode is not done

        if math.isnan(reward):
            print("reward is nan.")

        info = None

        return obs, reward, done, info


if __name__ == "__main__":
    import random
    import matplotlib.pyplot as plt

    def setup_seed(seed):
        """ Set random seeds for reproducibility. """
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        np.random.seed(seed)
        random.seed(seed)
        torch.backends.cudnn.deterministic = True

    setup_seed(0)
    reward = 0
    env = env(start=5, m1=5, m2=5)  # Create environment instance
    obs = env.reset()   # Reset environment for a new episode
    for i in range(1000):
        act = random.randint(0, 1)  # Random action/policy
        obs, reward, done, info = env.step(act)
        if done == True:
            break   # Exit loop if the episode is done

    print("The predicted Q value of generated topology  using random policy is", round(reward*env.Q_norm, 3))

    # Visualize observation grids
    fig, ax = plt.subplots(1, 8, figsize=(10, 5))
    ax[0].set_title('Trace')
    ax[0].imshow(obs[0][0], cmap='gray', interpolation='nearest')
    ax[1].set_title('arc')
    ax[1].imshow(obs[0][1], cmap='gray', interpolation='nearest')
    ax[2].set_title('rect')
    ax[2].imshow(obs[0][2], cmap='gray', interpolation='nearest')
    ax[3].set_title('current')
    ax[3].imshow(obs[0][3], cmap='gray', interpolation='nearest')
    ax[4].set_title('start')
    ax[4].imshow(obs[0][4], cmap='gray', interpolation='nearest')
    ax[5].set_title('end')
    ax[5].imshow(obs[0][5], cmap='gray', interpolation='nearest')
    ax[6].set_title('m1')
    ax[6].imshow(obs[0][6], cmap='gray', interpolation='nearest')
    ax[7].set_title('m2')
    ax[7].imshow(obs[0][7], cmap='gray', interpolation='nearest')
    plt.show()

