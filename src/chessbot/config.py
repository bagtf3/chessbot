from chessbot import SP_DIR, MODEL_DIR
import os
import yaml, json

class Config(object):
    """
    Central knobs. Keep simple; override from a dict or flags as needed.
    """

    # files
    selfplay_dir = SP_DIR
    run_tag = "dummy"
    init_model = MODEL_DIR + 'dummy.h5'
    previous_run_tag = None

    # highest level params
    is_validation_run = False
    validation_every = 5
    n_workers = 2
    n_rounds = 51

    # per-game parameter sampling; resolved once per game in the main process.
    # each entry: param_name: [list of options to pick from].
    # only 1st-order scalar params are supported (no nested keys).
    sampleable = {}

    # MCTS
    c_puct = 2.0

    use_smart_pruning = True
    pruning_factor = 1.33
    
    # Simulation schedule
    sf_move_sims = 200
    sims_floor = 400
    sims_ceiling = 800

    # early stop
    es_check_every = 100
    min_top_visits = 300
    min_delta = 100
    use_robust = True
    robust_only_above = 2400
    es_jsd_thresh = 0.05    # JSD below this = converged; bounded [0, ln(2)~0.693]; can be a list
    es_jsd_n_stable = 3     # consecutive stable checks required for JSD stop
    es_jsd_min_delta = 100  # minimum delta_12 floor for JSD stop
    jsd_min_sims = 600      # sims before JSD stop is considered; RSC stop is active before this
    
    # Game stuff
    n_games = 128
    games_at_once = 128
    micro_batch = 4
    min_batch = 4
    fwd_batch = 256

    max_game_length = 200
    min_game_length = 5
    use_syzygy = False
    use_material_diff = True
    material_diff_cutoff = 9
    material_diff_cutoff_span = 20

    use_eval_draw = True
    eval_draw_min_plies = 50
    eval_draw_thresh = 0.1
    eval_draw_span = 15

    play_vs_sf_prob = 0.5
    sf_depth = 10
    sf_config = {"Threads": 1, "Hash": 256}
    sf_exclude = ["piece_training"]

    # rescoring config
    rescore_depth = 12
    rescore_analyze_batch = 30
    rescore_equiv_range = 25
    rescore_inaccuracy_cp = 75
    rescore_blunder_cp_loser = 90
    rescore_blunder_cp_winner = 200
    rescore_eviction_window = 500
    rescore_cache_size = 50000
    rescore_n_sf_threads = 1

    # blunder replay
    blunder_replay_min_ply = 20

    use_collar_rescoring = False
    collar_threshold_cp = 300
    collar_n_consec = 5
    collar_reset_cp = 50
    train_on_stockfish = True
    train_on_validation = False
    validation_min_training_depth = 9

    z_mix = 0.9  # Y = z_mix * Z_stm + (1 - z_mix) * Q

    rescore_kl_threshold = 0.75   # always include if KL exceeds this
    rescore_ce_threshold = 0.625  # always include if value CE exceeds this
    rescore_sample_floor = 0.2    # min probability for soft-include sampling

    KL_boost_threshold = 1.75
    KL_weight_boost = 1.0

    game_probs = {
        "startpos":0.4, "pre_opened_mini": 0.22, "pre_opened": 0.27,
        "random_init": 0.08, "piece_odds": 0.02, "piece_training": 0.01
    }

    # game type mix for validation rounds; same format as game_probs.
    # all games are paired (one as white, one as black) and must be unique positions.
    # if a type exhausts unique positions, UHO fills the remainder.
    validation_game_probs = {"UHO": 1.0}

    # priors
    uniform_eps = 0.25
    prior_clip_max = 0.75
    fpu_reduction = 0.1
    qema_span = 40
    qdelta_span = 100

    # randomness
    add_root_noise = True
    dirichlet_eps = 0.3
    dirichlet_alpha = 0.3
    reuse_tree = True
    sample_moves = True
    move_sample_temp_range = [0.000001, 1.25]
    
    learning_rate = 1e-4          # optimizer LR; applied fresh at every retrain
    policy_loss_weight = 0.25     # per-sample weight for policy head
    value_loss_weight = 0.25      # per-sample weight for value head (non-draw)
    draw_value_scale = 0.5        # multiplies value_loss_weight for drawn games
    vscale = 0.9
    contempt_flip_q  = -0.22
    contempt_fight_c = 0.06
    contempt_save_c  = 0.12
    retrain_batch_size = 512
    retrain_size = 10240
    training_queue_buffer = 30720

    # inference / retrain backend selection
    inference_backend = "tf_xla"  # "tf_xla" | "ort_trt"
    retrain_backend = "tf"        # "tf" | "pytorch"

    # PyTorch retrain (only used when retrain_backend = "pytorch")
    pytorch_model_path = ""

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

        self.pending_training_dir = os.path.join(self.run_dir, "pending_training")
        
        game_dir = os.path.join(self.run_dir, "game_logs")
        self.game_dir = game_dir

        self.game_index_file = os.path.join(self.run_dir, "game_index.json")
        self.progress_csv_path = os.path.join(self.run_dir, "eval_progress.csv")
        self.progress_plot_path = os.path.join(self.run_dir, "eval_progress.png")

        model_name = f"{self.run_tag}_model.h5"
        self.model_path = os.path.join(self.run_dir, model_name)

        if 'dummy' not in self.run_dir:
            for d in (self.run_dir, self.game_dir):
                os.makedirs(d, exist_ok=True)

        # ORT/TRT: fall back to env var if not set explicitly in YAML
        if self.inference_backend == "ort_trt" and not self.ort_trt_engine_cache_dir:
            self.ort_trt_engine_cache_dir = os.getenv("TRT_ENGINE_CACHE_DIR", "")

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
    def from_dict(cls, data, init=True, permissive=False):
        """
        Construct a Config from a plain dict and optionally init paths.
        """
        inst = cls()
        if permissive:
            inst.update_permissive(data)
        else:
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

    def update_permissive(self, mapping=None, **kwargs):
        if mapping is not None:
            try:
                items = mapping.items()
            except AttributeError:
                items = mapping
            for k, v in items:
                setattr(self, k, v)
        for k, v in kwargs.items():
            setattr(self, k, v)

    def copy(self, init=False):
        """
        Return a fresh Config instance cloned from this one.
        By default this does not touch the filesystem; pass init=True
        to call init_paths() on the new instance.
        """
        # use the existing dict/ctor path so update() validation stays active
        cfg_dict = self.to_dict()
        new = self.__class__.from_dict(cfg_dict, init=False, permissive=True)

        # do not eagerly create directories unless requested
        if init:
            new.init_paths()
        return new
