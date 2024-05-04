Q_norm = 1000000

def reward_fun(Q, error_flag):
    """
    Calculate a normalized reward based on Q, bounded by Q_max.
    If an error is flagged (error_flag == 1), the reward is set to 0.

    Args:
    Q (int or float): The input value to normalize as a reward.
    error_flag (int): Indicator if an error occurred (1 for error, 0 for no error).

    Returns:
    float: The normalized reward, constrained between 0 and 1, or 0 if there is an error.
    """
    if error_flag == 1:
        return 0.0  # Return immediately if there's an error

    # Normalize Q with respect to Q_max
    reward_Q_part = min(max(Q, 0)/ Q_norm, 3)

    return reward_Q_part
