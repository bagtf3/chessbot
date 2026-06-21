import gc
import os
import time

import numpy as np


def make_trt_session(onnx_path, model_name, trt_cache, max_bs):
    """Create an ORT TRT session. Pre-compiled engines are reused via content-hash prefix."""
    import hashlib
    import tensorrt  # registers TRT DLLs with Windows before ORT loads its TRT provider
    import onnxruntime as ort

    h = hashlib.sha256()
    with open(onnx_path, 'rb') as fh:
        while True:
            chunk = fh.read(1024 * 1024)
            if not chunk:
                break
            h.update(chunk)
    prefix = f'{model_name}_{h.hexdigest()[:12]}'

    ort.set_default_logger_severity(3)
    sess_opts = ort.SessionOptions()
    sess_opts.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
    sess_opts.log_severity_level = 3

    trt_opts = {
        'trt_engine_cache_enable':  True,
        'trt_engine_cache_path':    trt_cache,
        'trt_engine_cache_prefix':  prefix,
        'trt_fp16_enable':          True,
        'trt_force_timing_cache':   True,
        'trt_max_workspace_size':   4 * 1024 * 1024 * 1024,
        'trt_profile_min_shapes':   'enc_in:1x64',
        'trt_profile_opt_shapes':   f'enc_in:{max_bs}x64',
        'trt_profile_max_shapes':   f'enc_in:{max_bs}x64',
        'trt_timing_cache_enable':  True,
        'trt_timing_cache_path':    trt_cache,
    }

    engines = [f for f in os.listdir(trt_cache)
               if f.startswith(prefix) and f.endswith('.engine')]
    if engines:
        print(f'[ort_trt] loading cached engine: {engines[0]}')
    else:
        print(f'[ort_trt] no cached engine for {prefix!r}, TRT will compile')

    providers = [('TensorrtExecutionProvider', trt_opts)]
    sess = ort.InferenceSession(onnx_path, sess_options=sess_opts, providers=providers)
    active = sess.get_providers()[0]
    if active != 'TensorrtExecutionProvider':
        raise RuntimeError(f'[ort_trt] expected TRT but got {active}')
    print(f'[ort_trt] provider: TensorrtExecutionProvider')
    return sess


def make_ort_trt_infer(onnx_path, model_name, trt_cache, max_bs):
    sess = make_trt_session(onnx_path, model_name, trt_cache, max_bs)

    def infer(pair):
        enc_np = np.asarray(pair[0], dtype=np.int64)
        pol_fp16, wdl_fp16 = sess.run(
            ['policy_logits', 'value_out'], {'enc_in': enc_np}
        )
        logits = pol_fp16.astype(np.float32)
        wdl_raw = wdl_fp16.astype(np.float32)
        e = np.exp(wdl_raw - wdl_raw.max(axis=-1, keepdims=True))
        wdl = e / e.sum(axis=-1, keepdims=True)
        return logits, wdl

    return sess, infer


def selfplay_trt_paths(cfg):
    """Derive (trt_dir, model_name, onnx_path) for the selfplay TRT cache."""
    trt_dir = cfg.trt_cache or os.path.join(cfg.run_dir, 'trt_cache')
    model_name = cfg.trt_model_name or f'{cfg.run_tag}_selfplay'
    onnx_path = os.path.join(trt_dir, f'{model_name}.onnx')
    return trt_dir, model_name, onnx_path


def prepare_trt(cfg, trt_dir, model_name):
    """
    Export current model to ONNX and compile a TRT engine in the calling process.
    Deletes stale .engine files but preserves .profile and .timing for fast recompile.
    Patches cfg so workers call make_trt_session and get a cache hit, not a recompile.
    """
    from chessbot.train_pytorch import export_ts_to_onnx

    os.makedirs(trt_dir, exist_ok=True)
    onnx_path = os.path.join(trt_dir, f'{model_name}.onnx')

    print(f'[trt] exporting {cfg.model_path} -> {onnx_path}')
    export_ts_to_onnx(cfg.model_path, onnx_path)

    for f in os.listdir(trt_dir):
        if f.startswith(model_name) and f.endswith('.engine'):
            os.remove(os.path.join(trt_dir, f))
            print(f'[trt] removed stale engine: {f}')

    print('[trt] compiling TRT engine...')
    t0 = time.time()
    sess = make_trt_session(onnx_path, model_name, trt_dir, cfg.macro_batch)
    dummy = np.zeros((min(cfg.macro_batch, 256), 64), dtype=np.int64)
    sess.run(['policy_logits', 'value_out'], {'enc_in': dummy})
    del sess
    gc.collect()
    print(f'[trt] TRT engine ready ({time.time() - t0:.0f}s)')

    cfg.inference_backend = 'ort_trt'
    cfg.model_path = onnx_path
    cfg.trt_model_name = model_name
    cfg.trt_cache = trt_dir
    return cfg


def recompile_selfplay_trt(cfg):
    """
    Re-export ONNX from the freshly trained model and compile a new TRT engine,
    reusing cached .timing and .profile for fast compile. Called from retrain_worker
    after training so workers pick up the new weights on unpause.
    Returns elapsed seconds.
    """
    from chessbot.train_pytorch import export_ts_to_onnx

    trt_dir, model_name, onnx_path = selfplay_trt_paths(cfg)
    os.makedirs(trt_dir, exist_ok=True)

    t0 = time.time()
    print('[retrain] exporting ONNX for TRT recompile...')
    export_ts_to_onnx(cfg.model_path, onnx_path)

    for f in os.listdir(trt_dir):
        if f.startswith(model_name) and f.endswith('.engine'):
            os.remove(os.path.join(trt_dir, f))
            print(f'[retrain] removed stale engine: {f}')

    print('[retrain] compiling TRT engine...')
    sess = make_trt_session(onnx_path, model_name, trt_dir, cfg.macro_batch)
    dummy = np.zeros((min(cfg.macro_batch, 256), 64), dtype=np.int64)
    sess.run(['policy_logits', 'value_out'], {'enc_in': dummy})
    del sess
    gc.collect()
    elapsed = time.time() - t0
    print(f'[retrain] TRT engine ready ({elapsed:.0f}s)')
    return elapsed
