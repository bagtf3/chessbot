from chessbot import SP_DIR, MODEL_DIR
import os


class Config(object):
    """
    Central knobs. Keep simple; override from a dict or flags as needed.
    """

    is_validation_run = False
    validation_every = 10

    # files
    selfplay_dir = SP_DIR
    run_tag = "conv_net_flat_run2"
    init_model = SP_DIR + "conv_net_flat_run2/conv_net_flat_run2_model.h5"
    #run_tag = 'conv_12x512SE'
    #init_model = 'C:/Users/Bryan/Data/chessbot_data/models/conv_12x512SE_0.h5'
    previous_run_tag = None

    # MCTS
    c_puct = 1.5

    # Simulation schedule
    sf_move_sims = 250
    sims_floor = 1600
    sims_ceiling = 2200
    target_delta = 500

    # early stop
    use_sim_decision_model = False
    sim_decision_model_path = MODEL_DIR + "sim_decision-v1.dll"
    es_check_every = 50

    # Game stuff
    n_games = 64
    games_at_once = 64
    n_rounds = 51
    micro_batch = 8
    fwd_batch = 512

    max_game_length = 200
    min_game_length = 5
    material_diff_cutoff = 25
    material_diff_cutoff_span = 25
    use_syzygy = False

    play_vs_sf_prob = 0.25
    sf_depth = 12
    sf_config = {"Threads": 1, "Hash": 256}

    # post hoc server
    run_post_hoc = True
    mine_bonus_data = True

    game_probs = {
        "startpos":0.5, "pre_opened_mini": 0.22, "pre_opened": 0.22,
        "random_init": 0.03,
        "piece_odds": 0.02, "piece_training": 0.01
    }

    # priors
    prior_clip_max = 0.6
    prior_clip_min = 0.0125

    # randomness
    add_root_noise = True
    dirichlet_eps = 0.20
    dirichlet_alpha = 0.3
    sample_moves = True
    move_sample_temp_range = [1e-6, 1.2]

    target_y_weights = {'vwq': 0.3, 'z': 0.4, 'z_taper': 0.3}
    loss_weights = {"policy_winner": 1.5, "policy_loser": 1.5, "value_out": 1.0}
    vscale = 0.9
    draw_weight = 0.1

    def __init__(self):
        """
        Initialize runtime paths and derived config fields.

        Derive run_dir, game_dir, and various path attributes from
        class defaults and creates the directories.
        """
        # resolve run_dir from class attributes
        resolved_run_dir = os.path.join(self.selfplay_dir, self.run_tag)
        resolved_run_dir = os.path.abspath(resolved_run_dir)
        self.run_dir = resolved_run_dir
        Config.run_dir = resolved_run_dir

        # game dir
        game_dir = os.path.join(self.run_dir, "game_logs")
        self.game_dir = game_dir
        Config.game_dir = game_dir

        # index and progress paths
        self.game_index_file = os.path.join(self.run_dir, "game_index.json")
        Config.game_index_file = self.game_index_file

        self.progress_csv_path = os.path.join(self.run_dir, "eval_progress.csv")
        Config.progress_csv_path = self.progress_csv_path

        self.progress_plot_path = os.path.join(self.run_dir, "eval_progress.png")
        Config.progress_plot_path = self.progress_plot_path

        # model path default (can be overridden later)
        model_name = f"{self.run_tag}_model.h5"
        self.model_path = os.path.join(self.run_dir, model_name)
        Config.model_path = self.model_path

        # make directories right away so callers can write safely
        for d in (self.run_dir, self.game_dir):
            os.makedirs(d, exist_ok=True)

        # provide a small hook attribute so callers know this config was initialized
        self._paths_initialized = True

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
