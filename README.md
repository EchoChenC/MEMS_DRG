## Usage

- **To use the reinforcement learning algorithm:** Run `DRL_Code\\main_RL.py`
- **For a random strategy:** Run `DRL_Code\\main_random_policy.py`
- **To compare the convergence curves of the random strategy and the reinforcement learning:** Run `DRL_Code\\plt_reward.py`
- **To verify the surrogate model:** Run `Surrogate_Model_Code\\net.py`
- **To verify the mode identification model:** Run `Mode_Idt_Code\\MINN.py`


## Requirements

This code has been tested with the following package versions:

- Python: 3.8.5
- torch: 2.0.1+cu118
- tqdm: 4.50.2
- numpy: 1.24.4
- matplotlib: 3.7.3

### Installation

To install the required packages, run the following command:

```bash
pip install torch==2.0.1+cu118 tqdm==4.50.2 numpy==1.24.4 matplotlib==3.7.3
