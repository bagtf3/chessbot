from chessbot import SP_DIR, MODEL_DIR


class Config(object):
    """
    Central knobs. Keep simple; override from a dict or flags as needed.
    """

    # files
    run_tag = "new_conv_net_run0"
    selfplay_dir =  SP_DIR
    init_model = SP_DIR + "new_conv_net_run0/new_conv_net_run0_model.h5"
    #init_model = MODEL_DIR + "conv_64_token_12M_10.h5"

    # MCTS
    c_puct = 2.0

    # Simulation schedule
    sims_floor = 50
    sims_target = 540
    sims_ceiling = 200

    # early stop
    use_sim_decision_model = False
    sim_decision_model_path = MODEL_DIR + "sim_decision-v1.dll"
    es_check_every = 24
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
    n_training_games = 1000
    
    move_limit = 200
    material_diff_cutoff = 12
    material_diff_cutoff_span = 20

    play_vs_sf_prob = -1
    sf_depth = 2

    # post hoc server
    run_post_hoc = True
    mine_bonus_data = True
    
    game_probs = {
        "pre_opened": 0.175, "random_init": 0.25,
        "random_middle_game": 0.25, "random_endgame": 0.15,
        "piece_odds": 0.15, "piece_training": 0.025
    }
    
    # priors
    prior_clip_max = 0.65
    prior_clip_min = 0.001

    # root noise
    add_root_noise = True
    dirichlet_eps = 0.075
    dirichlet_alpha = 0.3

    # training
    training_queue_min = 3072
    fwd_batch = 1024

    target_y_weights = {'z': 0.2, 'z_taper':0.6, 'vwq':0.2}
    target_loss_weights = {"policy_winner": 2.0, "policy_loser": 1.0, "value_out": 1.5}
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