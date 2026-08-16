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
    init_model = MODEL_DIR + 'dummy.ts'
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
    es_jsd_thresh = 0.05    # JSD below this = converged; bounded [0, ln(2)~0.693]; can be a list
    es_jsd_n_stable = 3     # consecutive stable checks required for JSD stop
    es_jsd_min_delta = 100  # minimum delta_12 floor for JSD stop
    jsd_min_sims = 600      # sims before JSD stop is considered; RSC stop is active before this
    
    # Game stuff
    n_games = 128
    games_at_once = 128
    micro_batch = 4
    macro_batch = 256

    max_game_length = 200
    min_game_length = 5
    use_syzygy = False
    use_material_diff = True
    material_diff_cutoff = 9
    material_diff_cutoff_span = 20

    allow_resignation = False
    resign_threshold = 0.80
    resign_consecutive = 10
    resign_min_plies = 20

    use_eval_draw = True
    eval_draw_min_plies = 50
    eval_draw_thresh = 0.1
    eval_draw_span = 15

    play_vs_sf_prob = 0.5
    sf_depth = 10
    sf_config = {"Threads": 1, "Hash": 256, "UCI_ShowWDL": True}
    sf_exclude = ["piece_training"]

    # rescoring config
    # SF searches to whichever binds first: this time budget or a hardcoded
    # depth backstop. Under backlog it drops by a hardcoded 10ms.
    rescore_movetime_ms = 90
    rescore_analyze_batch = 30
    rescore_equiv_range = 25
    rescore_inaccuracy_cp = 75
    rescore_blunder_cp_loser = 90
    rescore_blunder_cp_winner = 200
    rescore_n_sf_threads = 1

    # target replay positions GameGenerator spreads across each round,
    # metered against real games queued/completions -- not a flood
    blunder_replay_per_round = 16384

    train_on_stockfish = True

    kl_boost_median_mult = 1.5   # pwht multiplier when KL > running EMA median
    kl_boost_p80_mult    = 3.0   # pwht multiplier when KL > running EMA 80th pctile
    kl_quantile_lr       = 0.0005 # step size for the online trackers (~1/lr, ~2000)

    inaccuracy_downweight = 0.5  # xc0 policy weight mult for inaccuracy-tier plies

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
    move_sample_temp_range = [0.000001, 2.0]
    move_sample_temp_plies = 20
    
    learning_rate = 1e-4          # optimizer LR; applied fresh at every retrain
    adam_beta2 = 0.9917           # Adam v-window: 1/(1-beta2) steps; 0.9917~=120, 0.999~=1000
    policy_loss_weight = 0.25     # per-sample weight for policy head
    value_loss_weight = 0.25      # per-sample weight for value head (non-draw)
    draw_value_scale = 1.0        # multiplies value_loss_weight for drawn games
    vscale = 0.9
    contempt_zero_q  = 0.0
    contempt_full_q  = 0.5
    contempt_fight_c = 0.0
    retrain_clip_norm = 10.0      # grad norm ceiling per retrain step
    retrain_batch_size = 512

    # LC0 distillation enrichment
    lc0_distill_model_name = ''   # e.g. 't1-large'; trt cache path via LC0_DISTILL_TRT_CACHE env
    lc0_distill_batch_size = 64   # positions per ORT inference call
    lc0_enrich_frac   = 0.0   # fraction of non-blunder accepted positions to enrich with lc0 (0=off, 1=all)
    lc0_enrich_weight = 0.75  # vwht/pwht multiplier applied to lc0-generated training samples

    # retrain-time validation against lc0-sourced samples is expensive and not
    # useful every retrain; only run it (and print/save the lc0 breakdown) on
    # every Nth retrain. Other retrains validate xc0 (incl. historic_seeded) only.
    lc0_validation_every = 10

    # Prior temperature scaling in C++ tree (0 = disabled)
    tempscale_entropy_target = 0.0   # normed entropy target; 0 = disabled
    tempscale_trigger_q      = -2.0  # STM-POV Q floor; -2 = always; 0.5 = winning only

    # board encoding fed to the model. Sole driver of the encoder on every path
    # (selfplay drain, training-data board field, retrain input). "xc0" = 64 int16
    # tokens; "lc0" = 112x8x8 lc0 planes. Selfplay and retrain MUST agree or the
    # model is fed garbage. Orthogonal to inference_backend (lc0_trt requires lc0).
    encoding_type            = "xc0"        # "xc0" | "xc0h" | "lc0"
    history_K                = 6            # xc0h only: history frame count

    # inference / retrain backend selection
    inference_backend        = "ort_trt"  # "ort_trt" | "lc0_trt"
    retrain_backend          = "pt_eager"  # "pt_eager"
    trt_model_name           = ""          # ort_trt: cache prefix (e.g. "xc0_precond")
    trt_cache                = ""          # ort_trt: path to TRT engine cache dir
    ort_trt_engine_cache_dir = ""          # legacy alias for trt_cache

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

        self.primary_buffer_dir = os.path.join(self.run_dir, "primary_buffer")
        self.replay_buffer_dir = os.path.join(self.run_dir, "replay_buffer")
        self.historic_dir = os.getenv("BOOTSTRAP_TFREC_DIR", "")

        game_dir = os.path.join(self.run_dir, "game_logs")
        self.game_dir = game_dir

        self.game_index_file = os.path.join(self.run_dir, "game_index.json")
        self.progress_csv_path = os.path.join(self.run_dir, "eval_progress.csv")
        self.progress_plot_path = os.path.join(self.run_dir, "eval_progress.png")

        model_name = f"{self.run_tag}_model.ts"
        self.model_path = os.path.join(self.run_dir, model_name)

        if 'dummy' not in self.run_dir:
            for d in (self.run_dir, self.game_dir,
                      self.primary_buffer_dir, self.replay_buffer_dir):
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
