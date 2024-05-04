from DRL_Code.D3QN.utils import *
from DRL_Code.D3QN.agent import *
from DRL_Code.D3QN.config import *

torch.set_default_tensor_type(torch.FloatTensor)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("D3QN_trainer.py device is", device)

class DQNTrainer():
    def __init__(self, env, config):
        self.config = merge_config(config, pytorch_config)
        self.learning_rate = self.config["learning_rate"]
        self.env_name = self.config['env_name']
        self.env = env

        self.eps = self.config['eps']
        self.memory = ReplayBuffer(self.config["memory_size"], (8, 40, 20))
        self.learn_start = self.config["learn_start"]
        self.batch_size = self.config["batch_size"]
        self.target_update_freq = self.config["target_update_freq"]
        self.clip_norm = self.config["clip_norm"]
        self.max_episode_length = self.config["max_episode_length"]
        self.seed = self.config["seed"]
        self.gamma = self.config["gamma"]

        self.act_dim = self.config["act_dim"]

        self.step_since_update = 0
        self.total_step = 0

        self.initialize_parameters()
        self.best_res = 0

        self.log = []

    def initialize_parameters(self):
        self.network = MyDQN().to(device)

        self.network.eval()
        self.network.share_memory()

        self.target_network = MyDQN().to(device)
        self.target_network.load_state_dict(self.network.state_dict())
        self.target_network.eval()

        self.optimizer = torch.optim.Adam(self.network.parameters(), lr=self.learning_rate)
        self.loss = nn.MSELoss()

    def evaluate(self, num_episodes=50):
        """Use the function you write to evaluate current policy.
        Return the mean episode reward of 50 episodes."""
        policy = lambda raw_state1: self.compute_action(self.process_state(raw_state1), eps=0.0)
        result = evaluate(env=self.env, policy=policy, num_episodes=num_episodes, seed=self.seed)
        return result

    def save(self, checkpoint_path):
        torch.save(self.network.state_dict(), checkpoint_path)

    def compute_values(self, processed_state):
        """Compute the value for each potential action. Note that you should NOT preprocess the state here."""
        output = self.network(processed_state)
        return output

    def compute_action(self, processed_state, eps=None):
        """Compute the action given the state. Note that the input
        is the processed state."""
        processed_state = processed_state.to(device)
        values = self.compute_values(processed_state)

        if eps is None:
            eps = self.eps

        #  the epsilon-greedy policy
        p = np.random.random()
        if p < eps:
            action = np.random.choice(self.act_dim)
        else:
            action = np.argmax(values.cpu().detach().numpy())
        return action


    def train(self, termtype, start, m1, m2):
        self.env.termtype = termtype
        self.env.start = start
        self.env.m1 = m1
        self.env.m2 = m2
        s = self.env.reset()

        processed_s = self.process_state(s)
        act = self.compute_action(processed_s)
        stat = {"loss": []}

        for t in range(self.max_episode_length):

            next_state, reward, done, _ = self.env.step(act)
            next_processed_s = self.process_state(next_state)

            self.memory.push(processed_s, act, reward, next_processed_s, done)

            processed_s = next_processed_s
            act = self.compute_action(next_processed_s)
            self.step_since_update += 1
            self.total_step += 1

            if done:
                self.log.append(reward)
                np.savetxt("RL_opt_record\\reward", self.log)
                break

            if (self.memory.mem_cntr) < self.learn_start:
                continue
            elif (self.memory.mem_cntr) == self.learn_start:
                pass
            else:
                if t % self.config["learn_freq"] != 0:
                    continue

                states, actions, rewards, states_,  terminal = self.memory.sample(self.batch_size)
                state_batch = torch.Tensor(states).to(device)
                action_batch = torch.Tensor(actions).to(device)
                reward_batch = torch.Tensor(rewards).to(device)
                next_state_batch = torch.Tensor(states_).to(device)
                done_batch = torch.Tensor(terminal).to(device)



                with torch.no_grad():
                    next_state_actions = torch.argmax(self.network(next_state_batch), dim=1)

                    Q_t_plus_one = self.target_network(next_state_batch).gather(1, next_state_actions.unsqueeze(1)).squeeze()

                    assert isinstance(Q_t_plus_one, torch.Tensor)
                    assert Q_t_plus_one.dim() == 1

                    Q_target = reward_batch.squeeze() + self.gamma * Q_t_plus_one * (1 - done_batch).squeeze()
                    assert Q_target.shape == (self.batch_size,)

                self.network.train()
                Q_t = self.network(state_batch).gather(1, action_batch.long().view(-1, 1)).squeeze()
                assert Q_t.shape == Q_target.shape

                self.optimizer.zero_grad()
                loss = self.loss(input=Q_t, target=Q_target)
                loss_value = loss.item()
                stat['loss'].append(loss_value)
                loss.backward()

                nn.utils.clip_grad_norm_(self.network.parameters(), self.clip_norm)

                self.optimizer.step()
                self.network.eval()

            if (self.memory.mem_cntr) >= self.learn_start and self.step_since_update > self.target_update_freq:
                self.step_since_update = 0

                self.target_network.load_state_dict(self.network.state_dict())

                self.target_network.eval()

        return reward, done

    def process_state(self, state):
        return torch.from_numpy(state).type(torch.float32)