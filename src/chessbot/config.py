from chessbot import SP_DIR, MODEL_DIR


class Config(object):
    """
    Central knobs. Keep simple; override from a dict or flags as needed.
    """

    # files
    run_tag = "deep_sim_test"
    selfplay_dir =  SP_DIR
    init_model = SP_DIR + "new_conv_net_run1/new_conv_net_run1_model.h5"

    # MCTS
    c_puct = 1.0

    # Simulation schedule
    sims_floor = 50
    sims_target = 1000
    sims_ceiling = 200

    # early stop
    use_sim_decision_model = False
    sim_decision_model_path = MODEL_DIR + "sim_decision-v1.dll"
    es_check_every = 60
    es_best_move_threshold = 0.85
    bs_best_move_threshold = 0.5
    prefer_top_q = False
    
    use_q_override = False
    q_override_vis_ratio = 0.8
    q_override_q_margin = 0.08
    q_override_min_vis = 800
    q_override_top_k = 2

    # Game stuff
    micro_batch_size = 4
    games_at_once = 256
    n_training_games = 1500
    
    move_limit = 200
    min_game_length = 10
    material_diff_cutoff = 25
    material_diff_cutoff_span = 30

    play_vs_sf_prob = 0.5
    sf_depth = 12

    # post hoc server
    run_post_hoc = True
    mine_bonus_data = True
    
    # game_probs = {
    #     "pre_opened": 0.20, "random_init": 0.20,
    #     "random_middle_game": 0.25, "random_endgame": 0.15,
    #     "piece_odds": 0.15, "piece_training": 0.05
    # }
    game_probs = {
        "pre_opened": 0.1, #"random_init": 0.20,
        "random_middle_game": 0.45, "random_endgame": 0.45
        #"piece_odds": 0.15, "piece_training": 0.05
    }
    
    # priors
    prior_clip_max = 0.75
    prior_clip_min = 0.001

    # root noise
    add_root_noise = True
    dirichlet_eps = 0.075
    dirichlet_alpha = 0.5

    # training
    training_queue_min = 4096
    fwd_batch = 1024

    target_y_weights = {'z': 0.25, 'z_taper':0.5, 'vwq':0.25}
    loss_weights = {"policy_winner": 1.75, "policy_loser": 1.25, "value_out": 1.5}
    draw_weight = 0.2

    def to_dict(self):
        return {
            k: getattr(self, k)
            for k in dir(self)
            if not k.startswith("_") and not callable(getattr(self, k))
        }
    
    def update(self, mapping=None, **kwargs):
        if mapping is not None:
            try:
                items = mapping.items()
            except AttributeError:
                items = mapping
            for k, v in items:
                if not hasattr(self, k):
                    raise AttributeError(f"Unknown config key: {k}")
                setattr(self, k, v)
        for k, v in kwargs.items():
            if not hasattr(self, k):
                raise AttributeError(f"Unknown config key: {k}")
            setattr(self, k, v)