from chessbot import SP_DIR


class Config(object):
    """
    Central knobs. Keep simple; override from a dict or flags as needed.
    """

    # files
    run_tag = "conv_1000_test"
    selfplay_dir =  SP_DIR
    init_model = "c:\\Users\\Bryan\\Data\\chessbot_data\\models\\conv_small_init.h5"
    
    # MCTS
    c_puct = 1.5
    anytime_uniform_mix = 0.15
    endgame_uniform_mix = 0.2

    # Simulation schedule
    sims_target = 200
    micro_batch_size = 10

    # early stop
    es_min_sims = 150
    es_check_every = 25
    es_gap_frac = 0.8
    es_top_node_frac = 0.7
    
    # Q-override selection
    use_q_override = True
    q_override_vis_ratio = 0.80
    q_override_q_margin = 0.08
    q_override_min_vis = 1200
    q_override_top_k = 3
    
    # Game stuff
    games_at_once = 100
    n_training_games = 5000
    
    move_limit = 160
    material_diff_cutoff = 15
    material_diff_cutoff_span = 30

    play_vs_sf_prob = 0.5
    sf_depth = 10
    
    game_probs = {
        "pre_opened": 0.25, "random_init": 0.2,
        "random_middle_game": 0.2, "random_endgame": 0.1,
        "piece_odds": 0.1, "piece_training": 0.15
    }
    
    # boosts/penalize
    use_prior_boosts = True
    prior_clip_max = 0.35
    prior_clip_min = 0.001
    endgame_prior_adjustments = {
        "pawn_push":0.1, "capture":0.1, "repetition_penalty": 0.1
    }
    
    anytime_prior_adjustments = {"gives_check": 0.1, "repetition_penalty": 0.1}

    # TF
    training_queue_min = 4096
    fwd_batch = 1200
    vwq_blend = 0.5
    use_vwq_alpha_taper = True
    target_mean = 0.1
    additional_data_ratio = 1.0
    factorized_bins = (64, 64, 6, 4)

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