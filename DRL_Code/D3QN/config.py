pytorch_config = dict(
    env_name="Myenv",
    checkpoint_path="chkpt_dir",
    max_iteration=int(2e3),
    max_episode_length=300,
    evaluate_interval=200,
    gamma=0.99,
    eps=0.1,
    seed=0,
    learning_rate=1e-3,
    clip_norm=1.0,
    clip_gradient=True,
    memory_size=int(0.5e6),
    learn_start=int(5e3),
    batch_size=512,
    target_update_freq=int(1e3),
    learn_freq=3,
    act_dim=2,
    n=1
)