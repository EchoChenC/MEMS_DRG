import matplotlib.pyplot as plt
import numpy as np

# Load the reward data
reward_DRL = np.loadtxt("RL_opt_record\\reward")
reward_RP = np.loadtxt("RL_opt_record\\reward_rp")

# Find maximum cumulative reward for DRL
max1 = 0
reward_DRL_reshape = []
for i in range(len(reward_DRL)):
    if reward_DRL[i] >= max1:
        max1 = reward_DRL[i]
    reward_DRL_reshape.append(max1)

# Smooth the DRL data
window_size = 50
smoothed_data_DRL = np.convolve(reward_DRL, np.ones(window_size)/window_size, mode='valid')

# Adjust the start of the smoothed data to align with the original data
smoothed_data_DRL = np.r_[np.full(window_size//2, np.nan), smoothed_data_DRL]

# Find maximum cumulative reward for Random Policy
max2 = 0
reward_RP_reshape = []
for i in range(len(reward_RP)):
    if reward_RP[i] >= max2:
        max2 = reward_RP[i]
    reward_RP_reshape.append(max2)

# Smooth the Random Policy data
smoothed_data_RP = np.convolve(reward_RP, np.ones(window_size)/window_size, mode='valid')

# Adjust the start of the smoothed data to align with the original data
smoothed_data_RP = np.r_[np.full(window_size//2, np.nan), smoothed_data_RP]

# Create a figure with two subplots
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 6))

# Define the common y-axis range based on the maximum observed reward
ymax = max(np.max(reward_DRL_reshape), np.max(reward_RP_reshape))*1.1

# Plotting DRL data
ax1.plot(reward_DRL, label="DRL")
ax1.plot(reward_DRL_reshape, label="DRL_best")
ax1.plot(smoothed_data_DRL, label="DRL_smooth")
ax1.set_title("Deep Reinforcement Learning")
ax1.set_xlabel("Episodes")
ax1.set_ylabel("Reward")
ax1.set_ylim(0, ymax)
ax1.legend()

# Plotting Random Policy data
ax2.plot(reward_RP, label="RP")
ax2.plot(reward_RP_reshape, label="RP_best")
ax2.plot(smoothed_data_RP, label="RP_smooth")
ax2.set_title("Random Policy")
ax2.set_xlabel("Episodes")
ax2.set_ylabel("Reward")
ax2.set_ylim(0, ymax)
ax2.legend()

# Show the plots
plt.show()
