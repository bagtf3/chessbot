from chessbot import SP_DIR, MODEL_DIR


class Config(object):
    """
    Central knobs. Keep simple; override from a dict or flags as needed.
    """

    # files
    run_tag = "conv_stm_pov_test"
    selfplay_dir =  SP_DIR
    init_model = MODEL_DIR + "conv_stm_pov_test.h5"
    
    # MCTS
    c_puct = 2.0
    anytime_uniform_mix = 0.15
    endgame_uniform_mix = 0.2

    # Simulation schedule
    sims_floor = 50
    sims_target = 100
    sims_ceiling = 200
    micro_batch_size = 10

    # early stop
    use_sim_decision_model = False
    sim_decision_model_path = MODEL_DIR + "sim_decision-v1.dll"
    es_check_every = 250
    es_best_move_threshold = 0.85
    bs_best_move_threshold = 0.5
    prefer_top_q = False
    
    use_q_override = False
    q_override_vis_ratio = 0.8
    q_override_q_margin = 0.08
    q_override_min_vis = 800
    q_override_top_k = 2

    # Game stuff
    games_at_once = 100
    n_training_games = 1000
    
    move_limit = 160
    material_diff_cutoff = 15
    material_diff_cutoff_span = 30

    play_vs_sf_prob = 0.5
    sf_depth = 15
    
    game_probs = {
        "pre_opened": 0.25, "random_init": 0.25,
        "random_middle_game": 0.25, "random_endgame": 0.15,
        "piece_odds": 0.05, "piece_training": 0.05
    }
    
    # priors
    prior_clip_max = 0.35
    prior_clip_min = 0.001

    # TF
    training_queue_min = 512
    fwd_batch = 1024
    vwq_blend = 0.5
    use_vwq_alpha_taper = True
    target_mean = 0.1
    additional_data_ratio = 1.0

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