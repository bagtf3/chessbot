from chessbot import SP_DIR, MODEL_DIR
from copy import deepcopy
import os
import yaml, json

# xc0h history frame count; fixed, never modulated.
XC0H_K = 6

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
    # validation fires every N retrains as a discrete, non-overlapping batch.
    # The ladder counts runs, not games, so overlapping batches would advance
    # depth roughly twice as fast.
    validate_every_n_retrains = 5
    validation_games_per_batch = 256
    # a batch still open after this long is closed and summarised as partial,
    # so games lost to a dead worker cannot block retraining forever
    validation_max_minutes = 120
    # epoch 0 only: hold the first validation batch until this many games
    # are done, so the quick selfplay games show up first
    validation_warmup_games = 1000
    n_workers = 2

    # lc0 teacher probe fleet. 0 disables it; the worker runs its own net on
    # its own queue and is counted separately from the selfplay workers.
    n_lc0_workers = 0
    telemetry_kind = "xc0"
    # xc0 blunder replay bulk rounds; False skips them entirely
    use_blunder_replay = True

    # per-game parameter sampling; resolved once per game in the main process.
    # each entry: param_name: [list of options to pick from]; None = nothing.
    # only 1st-order scalar params are supported (no nested keys).
    sampleable = None
    # set by scenario_cfg on each game's config: the names (not options) of the
    # params drawn for that game, which the looper records in its summary.
    # Runtime only, never set from yaml.
    sampled_params = None

    # MCTS
    c_puct = 2.0

    pruning_factor = 1.33
    
    # Simulation schedule. Each move the tree gets sims_min = sims_floor and
    # sims_max = sims_schedule at the current ply (the entry with the largest
    # start ply at or below it), else sims_ceiling; both times the effort scale.
    sf_move_sims = 400
    sims_floor = 400
    sims_ceiling = 800
    sims_schedule = None   # {start_ply: ceiling}

    # per-bin sims scaling; "" disables. effort_alpha is sampled per game.
    effort_scale_table = ""
    effort_alpha = 0.0

    # early stop -- tiered rule (C++, pyfastchess mcts.cpp evaluate_early_stop)
    es_check_every = 100
    use_robust = True       # fallback selector when a search hits "full" (ceiling)

    es_tier1_consec = 3         # "runaway": consecutive checkins with the same leader
    es_tier1_jsd_thresh = 0.005
    es_tier2_consec = 5         # "two-horse race": consecutive checkins, top-2 stable in top-3
    es_tier2_jsd_thresh = 0.0003
    es_tier1_dq12_floor = -0.25
    es_tier1_qema_veto = 0.2
    es_observe_only = False     # experiments: record would-stop, never stop
    es_record_all = False       # experiments: full check-in rows from the first
    rsc_top_n = 5
    rsc_min_visits = 60
    rsc_visit_share_floor = 0.15
    rsc_margin = 0.05

    # search controller (C++, pyfastchess controller.cpp; the rules are
    # documented at the ctrl params in mcts.hpp). ctrl_model: an exported
    # two-head <checkpoint>.ctrl dir, loaded once per worker; ctrl_enable needs
    # it and es_check_every 0. Off on Stockfish's turns in vs-SF games.
    ctrl_model = ""
    # "ctrl" or "old": switches every stop-rule setting together
    # (game_utils.STOP_RULE_ARMS) so the two can be sampled per game. None
    # leaves the config's own settings alone.
    stop_rules = None
    ctrl_enable = False
    ctrl_min_stop = 300         # earliest stop; sims_max is held at least this
    ctrl_s_thresh = 0.95
    ctrl_q_thresh = 0.98
    ctrl_agree_consec = 2
    ctrl_ext_max = 2            # extensions per move; 0 = locks are never extended
    ctrl_ext_sims = 2000        # sims per extension
    ctrl_ext_d_max = 0.9
    ctrl_ext_q0 = 0.7
    ctrl_ext_absq0 = 0.6
    ctrl_ext_slope = 0.6666667
    ctrl_ext_pruning = 0.5      # pruning_factor while extended; restored next move
    ctrl_record = False         # keep every check-in in the game log (ply["ctrl"])
    checkin_break = False       # collect_many_leaves returns at each check-in

    # Game stuff
    n_games = 128
    games_at_once = 128
    micro_batch = 4
    macro_batch = 256
    # 2x micro_batch past min(dynamic_mbs_min, half the sims ceiling)
    dynamic_mbs = False
    dynamic_mbs_min = None
    # {root_visits: micro_batch}; when set it replaces micro_batch/dynamic_mbs
    mbs_schedule = None

    max_game_length = 200
    min_game_length = 5
    use_syzygy = False
    use_material_diff = True

    allow_resignation = False
    resign_threshold = 0.80
    resign_consecutive = 10
    resign_min_plies = 20

    use_eval_draw = True
    eval_draw_min_plies = 50
    eval_draw_thresh = 0.1
    eval_draw_span = 15

    sf_validation_depth = 10
    sf_config = {"Threads": 1, "Hash": 256, "UCI_ShowWDL": True}

    # rescoring config
    # SF searches to whichever binds first: this time budget or a hardcoded
    # depth backstop. Under backlog it drops by a hardcoded 10ms.
    rescore_movetime_ms = 90
    rescore_movetime_validation_ms = 160
    rescore_analyze_batch = 30
    
    rescore_equiv_min = 10
    rescore_equiv_max = 60
    # expected-score-loss bars for an inaccuracy and a blunder
    rescore_esl_inaccuracy = 0.05
    rescore_esl_blunder = 0.12
    rescore_n_sf_workers = 1

    train_on_stockfish = True

    kl_boost_median_mult = 1.5   # pwht multiplier when KL > running EMA median
    kl_boost_p80_mult    = 3.0   # pwht multiplier when KL > running EMA 80th pctile
    kl_quantile_lr       = 0.0005 # step size for the online trackers (~1/lr, ~2000)

    game_probs = {
        "startpos":0.4, "pre_opened_mini": 0.22, "pre_opened": 0.27,
        "random_init": 0.08, "piece_odds": 0.02, "piece_training": 0.01
    }

    # game type mix for validation rounds; same format as game_probs.
    # all games are paired (one as white, one as black) and must be unique positions.
    # if a type exhausts unique positions, UHO fills the remainder.
    validation_game_probs = {"UHO": 1.0}

    # priors
    prior_uniform_eps = 0.25
    prior_clip_max = 0.75
    fpu_reduction = 0.1
    # sharp boost: per node, while its visits are inside [sharp_lo, sharp_hi],
    # captures and checks get prior bonus
    # sharp_boost * (capture + check) * (1 - N/sharp_M)
    sharp_lo = 100
    sharp_hi = 250
    sharp_M = 10
    sharp_boost = 0.0
    qema_span = 40
    qdelta_span = 100
    recent_visits_span = 400    # visit_share EMA span
    # qmm_force sends a visit to each new Qmm leader; qmm_gate is the completed
    # backups a child needs to join Qmm (0 = 1); qmm_gap_block vetoes
    # tier1/tier2 stops while the leader's STM Qmm - Q < -qmm_gap_block (0 = off)
    qmm_force = False
    qmm_gate = 0
    qmm_gap_block = 0.0
    qmm_min_visits = 3
    # tree mechanics
    tree_reuse = True
    tree_descent_cap = 1000     # descents per collect call; 0 = no cap
    tree_resort_thresholds = [250, 1500, 5000]
    # randomness. dirichlet_eps = 0 turns root noise off;
    # move_sample_temp_plies = 0 turns move sampling off
    dirichlet_eps = 0.3
    dirichlet_alpha = 0.3
    move_sample_temp_range = [0.000001, 2.0]
    move_sample_temp_plies = 20
    # sampling candidates: the most-visited move plus the next ones up to
    # rank sample_top_k, each within sample_max_q_delta (STM Q) of the leader
    sample_top_k = 7
    sample_max_q_delta = 0.4
    
    learning_rate = 1e-4          # optimizer LR; applied fresh at every retrain
    retrain_weight_decay = 0.005  # AdamW decoupled, 2D+ weights only
    adam_beta2 = 0.9917           # Adam v-window: 1/(1-beta2) steps; 0.9917~=120, 0.999~=1000
    draw_value_scale = 1.0        # per-sample value weight for drawn games (decisive = 1.0)
    vscale = 0.9
    # vscale applies only while raw |root Q| > vscale_gate_q (0 = always)
    vscale_gate_q = 0.9
    # value outcome before a sampled deviation = search WDL at that deviation
    value_bootstrap_at_sample = False
    contempt_zero_q  = 0.0
    contempt_full_q  = 0.5
    contempt_fight_c = 0.0
    contempt_min_completed = 100   # root results before contempt latches
    retrain_clip_norm = 10.0      # grad norm ceiling per retrain step
    retrain_batch_size = 512

    # LC0 distillation enrichment
    lc0_distill_model_name = ''   # e.g. 't1-large'; trt cache path via LC0_DISTILL_TRT_CACHE env
    lc0_distill_batch_size = 64   # positions per ORT inference call

    # Prior temperature scaling in C++ tree (0 = disabled)
    tempscale_entropy_target = 0.0   # normed entropy target; 0 = disabled
    tempscale_trigger_q      = -2.0  # STM-POV Q floor; -2 = always; 0.5 = winning only

    # board encoding fed to the model; selfplay and retrain must agree.
    encoding_type            = "xc0"        # "xc0" | "xc0h" | "lc0"

    # inference backend; retrain is always pt_eager, not a config knob.
    inference_backend        = "ort_trt"  # "ort_trt" | "lc0_trt"
    trt_model_name           = ""          # ort_trt: cache prefix (e.g. "xc0_precond")
    trt_cache                = ""          # ort_trt: path to TRT engine cache dir
    # xc0 sessions in one inference-server process; 0 = each worker its own
    infer_server_threads     = 0

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
        self.historic_dir = os.getenv("BOOTSTRAP_PKL_GZ", "")

        game_dir = os.path.join(self.run_dir, "game_logs")
        self.game_dir = game_dir

        self.game_index_file = os.path.join(self.run_dir, "game_index.json")
        self.game_index_staging_file = os.path.join(
            self.run_dir, "game_index_staging.json")
        self.progress_csv_path = os.path.join(self.run_dir, "eval_progress.csv")
        self.progress_plot_path = os.path.join(self.run_dir, "eval_progress.png")

        model_name = f"{self.run_tag}_model.ts"
        self.model_path = os.path.join(self.run_dir, model_name)

        if 'dummy' not in self.run_dir:
            for d in (self.run_dir, self.game_dir,
                      self.primary_buffer_dir, self.replay_buffer_dir):
                os.makedirs(d, exist_ok=True)

        self.paths_initialized = True

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
        Return a deep, independent clone: every value, class defaults included,
        is copied onto the new instance, so no dict or list is ever shared
        between configs. __init__ is skipped, so nothing touches the
        filesystem; pass init=True to call init_paths() on the clone.
        """
        new = self.__class__.__new__(self.__class__)
        new.__dict__.update(deepcopy(self.to_dict()))
        if init:
            new.init_paths()
        return new
