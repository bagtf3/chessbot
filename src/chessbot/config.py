from chessbot import SP_DIR, MODEL_DIR
import os
import yaml

class Config(object):
    """
    Central knobs. Keep simple; override from a dict or flags as needed.
    """

    is_validation_run = False
    validation_every = 8

    # files
    selfplay_dir = SP_DIR
    run_tag = "conv_12x296_bootstrapped"
    init_model = MODEL_DIR + 'conv64_9x296_bootstrapped.h5'
    previous_run_tag = None

    # MCTS
    c_puct = [1.25, 1.5, 1.5, 2.0, 2.0, 2.25]

    # Simulation schedule
    sf_move_sims = 250
    sims_floor = 850
    sims_ceiling = 1000
    target_delta = 500

    # early stop
    use_sim_decision_model = False
    sim_decision_model_path = MODEL_DIR + "sim_decision-v1.dll"
    es_check_every = 50

    # Game stuff
    n_games = 96
    games_at_once = 96
    n_rounds = 50
    micro_batch = 4
    fwd_batch = 384

    max_game_length = 200
    min_game_length = 5
    material_diff_cutoff = 15
    material_diff_cutoff_span = 20
    use_syzygy = False

    play_vs_sf_prob = 0.33
    sf_depth = 12
    sf_config = {"Threads": 1, "Hash": 256}

    # post hoc server
    run_post_hoc = True
    mine_bonus_data = True

    game_probs = {
        "startpos":0.4, "pre_opened_mini": 0.22, "pre_opened": 0.27,
        "random_init": 0.08,
        "piece_odds": 0.02, "piece_training": 0.01
    }

    # priors
    prior_clip_max = 0.55
    prior_clip_min = 0.015

    # randomness
    add_root_noise = True
    dirichlet_eps = 0.2
    dirichlet_alpha = 0.3
    sample_moves = True
    move_sample_temp_range = [1e-6, 1.25]

    target_y_weights = {'vwq': 0.5, 'z': 0.5, 'z_taper': 0.0}
    loss_weights = {"policy_winner": 0.25, "policy_loser": 0.25, "value_out": 0.25}
    vscale = 0.9
    draw_weight = 0.1

    def __init__(self):
        self.init_paths()

    def init_paths(self):
        """
        Derive run_dir, game_dir and other path attrs from class defaults.
        Creates directories and sets class-level Config.* fields so callers
        can update class attrs and re-run this to recompute paths.
        """
        resolved_run_dir = os.path.join(self.selfplay_dir, self.run_tag)
        resolved_run_dir = os.path.abspath(resolved_run_dir)
        self.run_dir = resolved_run_dir
        Config.run_dir = resolved_run_dir

        game_dir = os.path.join(self.run_dir, "game_logs")
        self.game_dir = game_dir
        Config.game_dir = game_dir

        self.game_index_file = os.path.join(self.run_dir, "game_index.json")
        Config.game_index_file = self.game_index_file

        self.progress_csv_path = os.path.join(self.run_dir, "eval_progress.csv")
        Config.progress_csv_path = self.progress_csv_path

        self.progress_plot_path = os.path.join(self.run_dir, "eval_progress.png")
        Config.progress_plot_path = self.progress_plot_path

        model_name = f"{self.run_tag}_model.h5"
        self.model_path = os.path.join(self.run_dir, model_name)
        Config.model_path = self.model_path

        for d in (self.run_dir, self.game_dir):
            os.makedirs(d, exist_ok=True)

        # public flag; keep the old name too for backward compatibility
        self.paths_initialized = True
        self._paths_initialized = True

    @classmethod
    def from_json(cls, path, init=True):
        """
        Create Config from a JSON file, update fields, and optionally init paths.
        """
        with open(path, "r", encoding="utf-8") as fh:
            disk = json.load(fh)

        inst = cls()
        # update existing attrs from disk (raises on unknown keys)
        inst.update(disk)

        if init:
            # idempotent: safe to call even if __init__ already ran init_paths
            inst.init_paths()
        return inst

    @classmethod
    def from_yaml(cls, path, init=True):
        """
        Create Config from a YAML file, update fields, and optionally init paths.
        """
        with open(path, "r", encoding="utf-8") as fh:
            disk = yaml.safe_load(fh) or {}

        inst = cls()
        inst.update(disk)

        if init:
            inst.init_paths()
        return inst

    @classmethod
    def from_dict(cls, data, init=True):
        """
        Construct a Config from a plain dict and optionally init paths.
        """
        inst = cls()
        inst.update(data)
        if init:
            inst.init_paths()
        return inst

    def update_via(self, path, init=True):
        """
        Read path (yaml|yml|json), load into a dict, and apply via update().
        Returns self. If init True, run init_paths() after update.
        """
        if not os.path.exists(path):
            raise FileNotFoundError(path)

        ext = os.path.splitext(path)[1].lower()
        if ext in (".yaml", ".yml"):
            with open(path, "r", encoding="utf-8") as fh:
                data = yaml.safe_load(fh) or {}
        elif ext == ".json":
            with open(path, "r", encoding="utf-8") as fh:
                data = json.load(fh)
        else:
            raise ValueError(f"unsupported config file type: {ext}")

        self.update(data)

        if init:
            self.init_paths()
        return self

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

    def copy(self, init=False):
        """
        Return a fresh Config instance cloned from this one.
        By default this does not touch the filesystem; pass init=True
        to call init_paths() on the new instance.
        """
        # use the existing dict/ctor path so update() validation stays active
        cfg_dict = self.to_dict()
        new = self.__class__.from_dict(cfg_dict, init=False)

        # do not eagerly create directories unless requested
        if init:
            new.init_paths()
        return new
