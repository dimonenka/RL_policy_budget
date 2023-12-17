class Config:
    hid_actor = 64
    hid_critic = 64
    entropy_start = 1e-3
    entropy_end = 1e-4
    lr_start = 3e-4
    lr_end = 1e-4
    lr_assignment = 2e-3
    gamma = 0.99
    batch_size = 64
    n_batches = 32
    warm_start_iters = 50
    mixing_coef = 0.95
    return_type = 'gae'  # 'gae', 'td', 'mc'
    ppo_update = True
    ppo_epochs = 10
    ppo_eps = 0.2
    gae_lam = 0.95
    n_hid_layers = 1
    rms = True
    adam_eps = 1e-5
    target_intervals = {"HalfCheetah-v4": (0, 4),
                        "Walker2d-v4": (0, 2.5),
                        "Ant-v4": (0, 3),
                        "Hopper-v4": (0, 2.5),
                        "resource-gathering-v1": (0, 4),}
    train = True
