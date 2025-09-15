import os
import chess
import chess.engine
import pandas as pd
import numpy as np
import random
import time
from collections import defaultdict
from copy import deepcopy

from IPython.display import display, clear_output, SVG
import chess.svg
import uuid

import tensorflow as tf
from tensorflow.keras import layers, Model
from tensorflow import keras

from chessbot.model import res_block
HE = "he_normal"

from chessbot.utils import random_init, format_time
import chessbot.encoding as cbe
from chessbot.encoding import (
    get_board_state, compute_move_priors, get_legal_labels, move_to_labels
)    

import chessbot.features as feats

import chessbot.utils as cbu
from chessbot.utils import mirror_move

from chessbot import SF_LOC, ENDGAME_LOC
import chess.syzygy

MODEL_DIR = "C:/Users/Bryan/Data/chessbot_data/models"                 

# -------------------------
# Core Transformer Stump
# -------------------------

class TransformerBlock(layers.Layer):
    def __init__(self, d_model, num_heads, ff_dim, rate=0.1, name=None, **kwargs):
        super().__init__(name=name)
        self.d_model = d_model
        self.num_heads = num_heads
        self.ff_dim = ff_dim
        self.rate = rate

        self.att = layers.MultiHeadAttention(num_heads=num_heads, key_dim=d_model)
        self.ffn = tf.keras.Sequential([
            layers.Dense(ff_dim, activation="gelu"),
            layers.Dense(d_model),
        ])
        self.norm1 = layers.LayerNormalization(epsilon=1e-6)
        self.norm2 = layers.LayerNormalization(epsilon=1e-6)
        self.drop1 = layers.Dropout(rate)
        self.drop2 = layers.Dropout(rate)

    def call(self, x, training=False):
        attn_output, attn_scores = self.att(x, x, return_attention_scores=True)
        x = self.norm1(x + self.drop1(attn_output, training=training))
        ffn_output = self.ffn(x)
        x = self.norm2(x + self.drop2(ffn_output, training=training))
        return x, attn_scores

    def get_config(self):
        config = super().get_config()
        config.update({
            "d_model": self.d_model,
            "num_heads": self.num_heads,
            "ff_dim": self.ff_dim,
            "rate": self.rate,
            "name": self.name,
        })
        return config

# -------------------------
# Heads (policy/value/aux)
# -------------------------
def value_head(trunk_vec, hidden=512, leak=0.05, name="value"):
    x = trunk_vec
    if hidden:
        # First hidden layer
        x = layers.Dense(hidden, name=f"{name}_dense1")(x)
        x = layers.LeakyReLU(alpha=leak, name=f"{name}_lrelu1")(x)

        # Second hidden layer at half size
        x = layers.Dense(hidden // 2, name=f"{name}_dense2")(x)
        x = layers.LeakyReLU(alpha=leak, name=f"{name}_lrelu2")(x)

    out = layers.Dense(1, activation="tanh", name=name)(x)  # [-1,1] White POV
    return out, "mse"


def policy_factor_head(trunk_vec, prefix, hidden=512, leak=0.05):
    """Factorized move logits (no capture)."""
    x = trunk_vec
    if hidden:
        # First hidden layer
        x = layers.Dense(hidden, name=f"{prefix}_dense1")(x)
        x = layers.LeakyReLU(alpha=leak, name=f"{prefix}_lrelu1")(x)

        # Second hidden layer at half size
        x = layers.Dense(hidden // 2, name=f"{prefix}_dense2")(x)
        x = layers.LeakyReLU(alpha=leak, name=f"{prefix}_lrelu2")(x)

    # Raw logits, no activation
    from_logits   = layers.Dense(64, name=f"{prefix}_from")(x)
    to_logits     = layers.Dense(64, name=f"{prefix}_to")(x)
    piece_logits  = layers.Dense(6,  name=f"{prefix}_piece")(x)
    promo_logits  = layers.Dense(5,  name=f"{prefix}_promo")(x)

    return [from_logits, to_logits, piece_logits, promo_logits]


def build_conv_trunk(input_shape=(8, 8, 25), width=128, n_blocks=6, leak=0.05):
    """Convolutional stump with residual blocks, returns feature map + pooled vector."""
    inputs = layers.Input(shape=input_shape, name="board")

    x = layers.Conv2D(width, 3, padding="same", use_bias=False, kernel_initializer=HE)(inputs)
    x = layers.BatchNormalization()(x)
    x = layers.LeakyReLU(alpha=leak)(x)

    for _ in range(n_blocks):
        x = res_block(x, width, leak=leak)

    # final compression
    x = layers.Conv2D(width // 2, 1, padding="same", use_bias=False)(x)
    x = layers.BatchNormalization()(x)
    x = layers.LeakyReLU(alpha=leak, name="trunk")(x)

    trunk_feat = x
    trunk_vec = layers.GlobalAveragePooling2D(name="trunk_vec")(trunk_feat)

    return Model(inputs, [trunk_feat, trunk_vec], name="conv_trunk")


def make_conv_model(input_shape=(8,8,25), width=128, n_blocks=6):
    trunk = build_conv_trunk(input_shape=input_shape, width=width, n_blocks=n_blocks)
    inputs = trunk.input
    trunk_feat, trunk_vec = trunk.output

    # Heads
    val_out, _ = value_head(trunk_vec, hidden=512)
    best_outputs = policy_factor_head(trunk_vec, prefix="best", hidden=512)

    model = Model(inputs, [val_out] + best_outputs, name="conv_factorized")
    return model


def evaluate_terminal(board):
    if board.is_checkmate():
        return 1 if board.result() == '1-0' else -1
    
    elif board.is_game_over(claim_draw=True):
        return 0
    
    else:
        raise Exception("Non terminal board found!")


def make_hybrid_model(input_shape=(8,8,25), width=128, n_blocks=6,
                      d_model=128, num_heads=4, ff_dim=512, n_transformers=3):
    # --- Conv trunk ---
    trunk = build_conv_trunk(input_shape=input_shape, width=width, n_blocks=n_blocks)
    inputs = trunk.input
    trunk_feat, trunk_vec = trunk.output  # (8,8,width//2), (vec,)

    # --- Reshape conv features into sequence ---
    seq = layers.Reshape((64, width // 2))(trunk_feat)   # (batch, 64, width//2)

    # Project channels to d_model so transformer sees right dimension
    if width // 2 != d_model:
        seq = layers.Dense(d_model, name="proj_to_dmodel")(seq)  # (batch, 64, d_model)

    # --- Transformer stack ---
    x = seq
    for i in range(n_transformers):
        x, _ = TransformerBlock(d_model=d_model,
                                num_heads=num_heads,
                                ff_dim=ff_dim,
                                rate=0.1,
                                name=f"tfblock_{i}")(x)

    tf_vec = layers.GlobalAveragePooling1D(name="tf_vec")(x)

    # --- Fuse conv + transformer vectors ---
    fused = layers.Concatenate(name="fused")([trunk_vec, tf_vec])
    fused = layers.Dense(256, activation="gelu")(fused)

    # --- Heads ---
    val_out, _ = value_head(fused, hidden=512)
    best_outputs = policy_factor_head(fused, prefix="best", hidden=512)

    model = Model(inputs, [val_out] + best_outputs, name="conv_trans_factorized")
    return model


def load_model(model_loc):
    custom = {"TransformerBlock": TransformerBlock}
    model = keras.models.load_model(model_loc, custom_objects=custom)
    return model
#%%
losses = {
    "value": "mse",
    "best_from": tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
    "best_to": tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
    "best_piece": tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
    "best_promo": tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)
}

opt = tf.keras.optimizers.Adam(learning_rate=1e-4)
loss_weights = {
    "value": 0.5,
    "best_from": 0.5,
    "best_to": 0.5,
    "best_piece": 0.5,
    "best_promo": 0.1
}

# model = make_hybrid_model(
#     input_shape=(8,8,25), width=256, n_blocks=6, d_model=128,
#     num_heads=4, ff_dim=512, n_transformers=4
# )
model = load_model("C:/Users/Bryan/Data/chessbot_data/models/conv_model_medium_v1000.h5")
model.compile(optimizer=opt, loss=losses, loss_weights=loss_weights)
model.summary()
#%%

def prepare_data_for_model(X, Y, model, W=None):
    inputs, outputs, weights = {}, {}, {}

    # Inputs
    for name, tensor in zip(model.input_names, model.inputs):
        vals = [x[name] for x in X]
        arr = np.array(vals, dtype=np.int32)
        inputs[name] = arr

    # Outputs
    for name, tensor in zip(model.output_names, model.outputs):
        vals = [y.get(name, None) for y in Y]
        # replace None with 0 (dummy class) if needed
        vals = [0 if v is None else v for v in vals]

        if name == "value":
            arr = np.array(vals, dtype=np.float32).reshape(-1, 1)
        else:
            arr = np.array(vals, dtype=np.int32)
        outputs[name] = arr

    # Sample weights (optional)
    if W is not None:
        for name, tensor in zip(model.output_names, model.outputs):
            vals = [w.get(name, 1.0) for w in W]   # default weight = 1.0
            arr = np.array(vals, dtype=np.float32)
            weights[name] = arr

    if W is not None:
        return inputs, outputs, weights
    else:
        return inputs, outputs


def prepare_X(X, model):
    if not isinstance(X, list):
        X = [X]
        
    inputs = {}

    # Inputs
    for name, tensor in zip(model.input_names, model.inputs):
        vals = [x[name] for x in X]
        arr = np.array(vals, dtype=np.int32)
        inputs[name] = arr
    
    return inputs

import json
from datetime import datetime

import pandas as pd
import matplotlib.pyplot as plt
import re


def load_eval_positions(path, max_positions=None):
    data = []
    with open(path, "r") as f:
        for i, line in enumerate(f):
            pos = json.loads(line)
            b = chess.Board(pos['fen'])
            pos['board'] = b
            pos['moves_list'] = list(b.legal_moves)
            pos['legal_labels'] = get_legal_labels(b, pos['moves_list'])
            data.append(pos)
            if max_positions and i >= max_positions - 1:
                break
            
    losses = []
    ranks = []
    n_moves = []
    for d in data:
        losses.append(np.mean([m['cp_loss'] for m in d['moves']]))
        ranks.append(np.mean([i+1 for i in range(len(d['moves']))]))
        n_moves.append(len(d['moves']))
    
    summary = {
        "expected_mean":np.mean(losses),
        "expected_rank":np.mean(ranks),
        "avg_moves": sum(n_moves)
    }
    
    return data, summary


def positions_to_inputs(positions):
    X = [get_board_state(chess.Board(p['fen'])) for p in positions]
    return np.stack(X, axis=0)


def evaluate_model(model, positions, batch_size=512, top_ns=(1, 3, 5)):
    X = positions_to_inputs(positions)
    raw_preds = model.predict(X, batch_size=batch_size, verbose=0)
    
    preds = []
    for i in range(X.shape[0]):
        p = {name: raw_preds[j][i] for j, name in enumerate(model.output_names)}
        preds.append(p)
    
    # assume outputs are dict of {head_name: logits}
    results = {
        "avg_cp_loss": 0.0,
        "avg_rank": 0.0,
        "top_hits": {n:0 for n in top_ns},
        "count": 0
    }
    
    for pos, pred in zip(positions, preds):
        # build move prior scores
        moves = pos["moves"]
        if not moves:
            continue

        uci_to_cp_loss = {m["uci"]: m["cp_loss"] for m in moves}
        losses = [[m['uci'], m['cp_loss']] for m in moves]
        losses = sorted(losses, key=lambda x: x[1])
        
        # model picks: take highest logit (from "best_*" heads combined)
        priors = compute_move_priors(pred, pos['legal_labels'])
        hardmax = np.argmax(priors)
        choice_uci = str(pos['moves_list'][hardmax])
        choice_cp_loss = uci_to_cp_loss[choice_uci]

        # metrics
        results["avg_cp_loss"] += choice_cp_loss

        # rank
        true_rank = [l[0] for l in losses].index(choice_uci)
        results["avg_rank"] += true_rank + 1

        # top-N
        for k in results['top_hits']:
            results['top_hits'][k] += 1*(true_rank <= k)

        results["count"] += 1

    # averages
    count = results["count"]
    results["avg_cp_loss"] /= count
    results["avg_rank"] /= count
    for n in top_ns:
        results["top_hits"][n] /= count

    return results


def run_gauntlet(models_paths, loader, eval_file, max_positions=None):
    positions, summary = load_eval_positions(eval_file, max_positions=max_positions)
    results = {}

    for path in models_paths:
        print(f"Evaluating {path}...")
        model = loader(path)
        res = evaluate_model(model, positions)
        results[path.split("/")[-1].split(".")[0]] = res
        print(res)

    return results, summary


def results_to_df(all_results):
    rows = []
    for name, metrics in all_results.items():
        # extract number after "v"
        match = re.search(r'v(\d+)', name)
        epoch = int(match.group(1)) if match else None

        rows.append({
            "model": name,
            "epoch": epoch,
            "avg_cp_loss": metrics["avg_cp_loss"],
            "avg_rank": metrics["avg_rank"],
            "top3": metrics["top_hits"][3]
        })
    df = pd.DataFrame(rows)
    df = df.sort_values("epoch")
    return df


def plot_all_metrics(df, random_baseline=None):
    if random_baseline is None:
        random_baseline = {"expected_mean": 288, "expected_rank": 15.6}

    fig, axes = plt.subplots(1, 3, figsize=(18, 5))

    # CP loss subplot
    axes[0].plot(df["epoch"], df["avg_cp_loss"], marker="o", label="Model")
    axes[0].axhline(random_baseline["expected_mean"],
                    color="red", linestyle="--",
                    label="Random baseline")
    axes[0].set_title("Average Centipawn Loss over Training")
    axes[0].set_xlabel("Epoch")
    axes[0].set_ylabel("Avg CP Loss")
    axes[0].legend()
    axes[0].grid(True)

    # Rank subplot
    axes[1].plot(df["epoch"], df["avg_rank"], marker="o",
                 color="orange", label="Model")
    axes[1].axhline(random_baseline["expected_rank"],
                    color="red", linestyle="--",
                    label="Random baseline")
    axes[1].set_title("Average Rank of Chosen Move")
    axes[1].set_xlabel("Epoch")
    axes[1].set_ylabel("Avg Rank (1=Best)")
    axes[1].legend()
    axes[1].grid(True)

    # Top-3 hit rate subplot
    axes[2].plot(df["epoch"], df["top3"], marker="o", color="green")
    axes[2].set_title("Top-3 Hit Rate over Training")
    axes[2].set_xlabel("Epoch")
    axes[2].set_ylabel("Top-3 Accuracy")
    axes[2].grid(True)

    plt.tight_layout()
    plt.show()



def get_score_against_sf(all_finished):
    res = {"win": 0, "lose": 0, "draw": 0}
    for af in all_finished:
        if af.outcome == 0:
            res['draw'] += 1
        if af.sf_plays and af.outcome == 1:
            res['lose'] += 1
        if af.sf_plays and af.outcome == -1:
            res['win'] += 1
        if not af.sf_plays and af.outcome == 1:
            res['win'] += 1
        if not af.sf_plays and af.outcome == -1:
            res['lose'] += 1
                
    print("Results against SF", res)


LOGFILE = "C:/Users/Bryan/Data/chessbot_data/self_play_stats.jsonl"
def log_game_stats(games, epoch=None):
    log_path = LOGFILE
    
    # Create/initialize empty file
    if not os.path.exists(log_path):
        os.makedirs(os.path.dirname(LOGFILE), exist_ok=True)
        with open(LOGFILE, "w") as f:
            pass
    
    stats = {
        "timestamp": datetime.utcnow().isoformat(),
        "epoch": epoch,
        "n_games": len(games),
        "white_wins": 0,
        "black_wins": 0,
        "draws": 0,
        "avg_length": 0,
    }
    lengths = []
    for g in games:
        if g.outcome > 0:
            stats["white_wins"] += 1
        elif g.outcome < 0:
            stats["black_wins"] += 1
        else:
            stats["draws"] += 1
        lengths.append(len(g.board.move_stack))

    if lengths:
        stats["avg_length"] = float(np.mean(lengths))

    # append to JSONL
    with open(log_path, "a") as f:
        f.write(json.dumps(stats) + "\n")

    return stats
#%%

class SelfPlayGame:
    def __init__(self, board=None, move_limit=90, **kwargs):
        self.game_id = str(uuid.uuid4())
        self.move_limit = move_limit
        
        n_init = kwargs.get("n_init", 2)
        self.board = board if board is not None else random_init(n_init)
        self.init_fen = self.board.fen()
        
        self.prevent_repetition = kwargs.get("prevent_repetition", True)
        self.pos_move_counter = defaultdict(int)
        self.pos_counter = defaultdict(int)
        self.move_counter = defaultdict(int)

        self.training_X = []
        self.training_Y = []
        self.sample_weights = []

        self.outcome = None
        self.game_complete = False

        self.mat_adv_counter = 0  # track 7-pt lead persistence
        self.draw_score = kwargs.get("draw_score", 0.0)
        
        self.sf_plays = kwargs.get("sf_plays", None)
        self.sample_scores = kwargs.get("sample_scores", False)

    def short_fen(self, fen=None):
        f = self.board.fen() if fen is None else fen
        return " ".join(f.split(" ")[:4])

    def check_termination(self):
        claim_draw = len(self.board.move_stack) > 70
        if self.board.is_game_over(claim_draw=claim_draw):
            print(self.game_id, "Game over detected:", self.board.result(), self.board.fen())
            self.outcome = evaluate_terminal(self.board)
            return True
    
        white_mat, black_mat = feats.get_piece_value_sum(self.board)
        mat_diff = 10*white_mat - 10*black_mat
        if abs(mat_diff) >= 12:
            self.mat_adv_counter += 1
        else:
            self.mat_adv_counter = 0
    
        if self.mat_adv_counter >= 28:
            print(self.game_id, "Material cutoff:", mat_diff)
            self.outcome = 1.0 if mat_diff > 0 else -1.0
            return True
        
        # Syzygy probe if few pieces
        if len(self.board.piece_map()) <= 5:
            # may not work so just go as normal
            try:
                outcomes = {-2: -1, -1:-1, 0:0, 1:1, 2:1}
                with chess.syzygy.open_tablebase(ENDGAME_LOC) as tablebase:
                    table_res = tablebase.probe_wdl(self.board)
                    
                table_res = table_res if self.board.turn else -1*table_res
                self.outcome = outcomes[table_res]
                print(self.game_id, "Endgame Solution Found:", self.outcome, self.board.fen())
                return True
            except:
                pass
            
        if self.board.fullmove_number > self.move_limit:
            print(self.game_id, "Move limit reached:", self.board.fullmove_number)
            self.outcome = 0.0
            return True
    
        return False

    def record_position(self):
        # STM-POV: mirror if Black to move
        to_score = self.board if self.board.turn else self.board.mirror()
        state = get_board_state(to_score)

        self.training_X.append({"board": state})
        
        # # labels filled in later (after outcome known)
        # self.training_Y.append({
        #     'value': None, 'best_from': None, 'best_to': None, 'best_piece': None,
        #     'best_promo': None
        # })
        
        # self.sample_weights.append({
        #     'value': 0.01, 'best_from': 0.0, 'best_to': 0.0, 'best_piece': 0.0,
        #     'best_promo': 0.0
        # })

    def get_premove_data(self):
        if self.check_termination():
            self.finalize_outcome()
            return [], []
        
        # always collect
        self.record_position()
        
        meta = [self.game_id]
        X = self.training_X[-1:]
        # this gets sent to the batch, move made when preds are returned
        return X, meta    
    
    def make_move(self, preds):
        """
        Select and push a move given model predictions for this position.
        raw_preds: list/array of outputs for ONE sample (from the batch)
        """
        
        short_fen = self.short_fen()
        self.pos_counter[short_fen] += 1
        
        to_score = self.board if self.board.turn else self.board.mirror()
        moves_list = list(to_score.legal_moves)
        
        # prevent threefold by storing pos + move and if that combo has been encountered
        if self.prevent_repetition:
            repeated = []
            for lm in list(self.board.legal_moves):
                if self.pos_move_counter[(short_fen, lm)] >= 2:
                    repeated.append(lm)
                    
                # # check the san too
                if self.move_counter[self.board.san(lm)] >= 2:
                    repeated.append(lm)
            
            # if there are no non-repeating moves, end game in draw
            if len(repeated) == len(moves_list):
                self.outcome = self.draw_score
                self.finalize_outcome()
                return
            
            if repeated:
                # flip and subset moves_list
                disallow = [m if self.board.turn else mirror_move(m) for m in repeated]
                moves_list = [m for m in moves_list if m not in disallow]
        
        # another guard rail
        if not moves_list:
            # Game is effectively over, finalize as draw if no moves
            self.outcome = self.draw_score
            self.finalize_outcome()
            return

        # use the prior finder to get best move
        legal_labels = get_legal_labels(to_score, moves_list)
        scores = compute_move_priors(preds, legal_labels)
        
        # pick the move
        if not self.sample_scores or (scores.max() > 0.2):
            top_idx = np.argmax(scores)
        else:
            top_idx = np.random.choice(len(scores), p=scores)
            
        chosen = moves_list[top_idx]
        chosen = chosen if self.board.turn else mirror_move(chosen)
        
        # push best move and update lists
        self._make_move(chosen, short_fen)
        return
    
    def _make_move(self, move, short_fen):
        self.move_counter[self.board.san(move)] += 1
        self.pos_move_counter[(short_fen, move)] += 1
        
        # record move info in the last Y slot
        # from_idx, to_idx, piece_idx, pr_idx = move_to_labels(self.board, move)
    
        # self.training_Y[-1].update({
        #     "from": from_idx,
        #     "to": to_idx,
        #     "piece": piece_idx,
        #     "promo":pr_idx,
        #     "move": move   # keep the actual move object too
        # })
        
        self.board.push(move)
        
    def force_move(self, move):
        """Use this to push an engine move or loading a starting pos """
        short_fen = self.short_fen()
        self.pos_counter[short_fen] += 1
        self._make_move(move, short_fen)
        return
    
    def finalize_outcome(self):
        # stm = True  # White POV
        # for x, y, w in zip(self.training_X, self.training_Y, self.sample_weights):
        #     # always set value from White’s perspective
        #     y["value"] = self.outcome if stm else -self.outcome
            
        #     if self.outcome == 0:
        #         w['value'] = 0.0
        #         # make sure everything else is 0
        #         w.update(
        #             {'best_from': 0.0, 'best_to': 0.0,
        #              'best_piece': 0.0,'best_promo': 0.0
        #         })
            
        #     if (self.outcome > 0 and stm) or (self.outcome < 0 and not stm):
        #         # winning side → good move
        #         y["best_from"] = y.pop("from")
        #         y["best_to"]   = y.pop("to")
        #         y["best_piece"]= y.pop("piece")
        #         y["best_promo"]  = y.pop("promo")
                
        #         # update weights and pay extra attention to promos
        #         w.update({'best_from': 0.1, 'best_to': 0.1, 'best_piece': 0.1})
        #         w['best_promo'] = 3.0 if y['best_promo'] > 0 else 0.005
                
        #     else:
        #         # remove the extra stuff
        #         for k in ["from", "to", "piece", "promo"]:
        #             y.pop(k, None)
    
        #     stm = not stm
        self.game_complete = True
        
    def show_board(self, flipped=False, sleep=0.0):
        clear_output(wait=True)
        display(SVG(chess.svg.board(board=self.board, flipped=flipped)))
        time.sleep(sleep)


def get_boosting_data(game, result, weights):
    # STM-POV: mirror if Black to move
    to_score = game.board if game.board.turn else game.board.mirror()
    move = result.move if game.board.turn else mirror_move(result.move)
    score = cbe.score_to_cp_white(result.info['score'])
    score = score if game.board.turn else -1*score
    
    # this was just calculated in get_premove_data
    moreX = game.training_X[-1]
        
    # add targets
    f, t, pi, pr = move_to_labels(to_score, move)
    moreY = {
        'value': score, 'best_to': t, 'best_from': f, 'best_piece': pi, 'best_promo': pr
    }
    
    moreW = deepcopy(weights)
    moreW['value'] = 0.75
    moreW['best_from'] = 1.2
    moreW['best_to'] = 1.2
    moreW['best_promo'] = 3.0 if moreY['best_promo'] > 0 else 0.01
    
    return moreX, moreY, moreW


import pickle
fen_path = "C:/Users/Bryan/Data/chessbot_data/fens2000.pkl"
with open(fen_path, "rb") as f:
    fen_list = pickle.load(f)

    
def self_play_batch(model, n_games=32, show=True, return_games=False, **kwargs):
    play_sf = kwargs.get("play_sf", False)
    engine = kwargs.get("engine", False)
    prevent_repetition = kwargs.get("prevent_repetition", True)
    sample_scores = kwargs.get("sample_scores", False)
    
    # this adds in the true best move for instant gradient boost
    instant_boosting = kwargs.get("instant_boosting", False)
    if instant_boosting:
        weights = deepcopy(model.compiled_loss._loss_weights)
        if isinstance(weights, list):
            weights = dict(zip(model.output_names, weights))
    
    # load games
    games = []
    for i in range(n_games):
        move_limit = 90
        n_init = kwargs.get("n_init", np.random.randint(2, 6))
        
        # half the games from the structured fens, half random inits
        if i < n_games/2:
            b = chess.Board(random.choice(fen_list))
        else:
            b = random_init(n_init)
            
        # optionally play white or black vs stockfish
        sf_plays = i % 2 == 0 if play_sf else None
        g = SelfPlayGame(
            board=b, move_limit=move_limit,
            sf_plays=sf_plays, prevent_repetition=prevent_repetition,
            sample_scores=sample_scores
        )
        
        games.append(g)
    
    all_finished = []
    train_X, train_Y, train_W = [], [], []
    while games:
        this_X, this_meta = [], []
        
        # get board data for GPU to batch
        for g in games:
            if not g.game_complete:
                X, meta = g.get_premove_data()
                this_X += X
                this_meta += meta
        
        # check if any games are newly finished
        finished = [g for g in games if g.game_complete]
        all_finished += finished
        # for g in finished:
        #     train_X += g.training_X
        #     train_Y += g.training_Y
        #     train_W += g.sample_weights
            
        games = [g for g in games if not g.game_complete]
        
        # if no data, we are done with this set
        if not this_X:
            break
        
        # predict and store
        if this_X:
            to_predict = prepare_X(this_X, model)
            raw_preds = model.predict(to_predict, batch_size=256, verbose=0)
            preds = {name: arr for name, arr in zip(model.output_names, raw_preds)}
            
            preds_lookup = {}
            for i, g_id in enumerate(this_meta):
                preds_lookup[g_id] = {k:v[i] for k, v in preds.items()}
        
        # make the moves
        for game in games:
            sf_turn = play_sf and (game.sf_plays == game.board.turn)
        
            result = None
            # run stockfish if it is needed
            if sf_turn or instant_boosting:
                result = engine.play(
                    game.board, limit=chess.engine.Limit(depth=3),
                    info=chess.engine.Info.ALL
                )
        
            # Boosting
            if instant_boosting:
                moreX, moreY, moreW = get_boosting_data(game, result, weights)
                train_X.append(moreX); train_Y.append(moreY); train_W.append(moreW)
        
            # Play the move
            if sf_turn:
                game.force_move(result.move)  # use result.move (guaranteed)
                
            else:
                game.make_move(preds_lookup[game.game_id])
        
        # show the game
        if show and games:
            game = games[0]
            flipped = False
            if play_sf:
                if game.sf_plays:
                    flipped = True
                    
            game.show_board(flipped=flipped, sleep=0.0)
        
        # drain the list once more
        finished = [g for g in games if g.game_complete]
        all_finished += finished
        # for g in finished:
        #     train_X += g.training_X
        #     train_Y += g.training_Y
        #     train_W += g.sample_weights
            
        games = [g for g in games if not g.game_complete]
    
    # we get here once its done with every game
    inputs, outputs, weights = prepare_data_for_model(train_X, train_Y, model, train_W)
    
    if play_sf:
        get_score_against_sf(all_finished)
        
    if return_games:
        return inputs, outputs, weights, all_finished
    
    else:
        return inputs, outputs, weights


### test games ###
i, o, w, g = self_play_batch(
    model, n_games=1, show=True, return_games=True,
    prevent_repetition=True
)

with chess.engine.SimpleEngine.popen_uci(SF_LOC) as engine:
    i, o, w, g = self_play_batch(
        model, n_games=1, show=True, return_games=True, n_init=0,
        play_sf=True, engine=engine, instant_boosting=False,
        prevent_repetition=False
    )
    
    i, o, w, g = self_play_batch(
        model, n_games=1, show=True, return_games=True, n_init=0,
        play_sf=False, engine=engine, instant_boosting=True,
        prevent_repetition=False
    )

#%%
all_evals = pd.DataFrame()
all_results = {}
loop_times = []
ol = 1001

MODEL_DIR = "C:/Users/Bryan/Data/chessbot_data/models/"
model_files = [f for f in os.listdir(MODEL_DIR) if 'conv_model_medium' in f]
model_paths = [os.path.join(MODEL_DIR, f) for f in model_files]

DATA_DIR = "C:/Users/Bryan/Data/chessbot_data/training_data"
eval_file = os.path.join(DATA_DIR, "tf_model_pos_eval_data.jsonl")

all_results, summary = run_gauntlet(model_paths, load_model, eval_file)
positions, summary = load_eval_positions(eval_file, max_positions=None)

#sf_results = evaluate_stockfish(positions, depth=1)
df = results_to_df(all_results)
plot_all_metrics(df, random_baseline=None)
#%%
while ol <= 2000:
    start_loop = time.time()
    show=ol % 7 == 0
    print(f"Outer Loop {ol}")
    if ol % 2 == 0:
        print("------- Playing vs Stockfish -------")
        with chess.engine.SimpleEngine.popen_uci(SF_LOC) as engine:
            inputs, outputs, weights, games = self_play_batch(
                model, n_games=192, show=show, return_games=True, n_init=3,
                play_sf=True, engine=engine, instant_boosting=True,
                prevent_repetition=False
            )
    
    else:
        print("------- Playing vs Self -------")
        with chess.engine.SimpleEngine.popen_uci(SF_LOC) as engine:
            inputs, outputs, weights, games = self_play_batch(
                model, n_games=96, show=show, return_games=True, n_init=3,
                instant_boosting=True, engine=engine, prevent_repetition=True
            )
        
    stats = log_game_stats(games, epoch=ol)
    print(stats)
    
    if ol % 5 == 0:
        eval_df = cbu.score_game_data(model, inputs, outputs)
        all_evals = pd.concat([all_evals, eval_df])
    
        if len(all_evals) > 2:
            cbu.plot_training_progress(all_evals)
    
    # train (no val)
    model.fit(
        inputs, outputs, sample_weight=weights,
        batch_size=256, epochs=2, shuffle=True
    )
    
    if ol % 25 == 0:
        # more frequent checkpointing
        model_out = os.path.join(MODEL_DIR, "tf_conv_medium_latest.h5")
        model.save(model_out)
        
    if ol % 100 == 0:
        model_out = os.path.join(MODEL_DIR, f"conv_model_medium_v{ol}.h5")
        model.save(model_out)
        
    stop_loop = time.time()
    loop_time = stop_loop - start_loop
    loop_times.append(loop_time)
    print(f"*** Loop {ol} completed in {format_time(loop_time)}")
    print(f"*** Avg loop time: {format_time(np.mean(loop_times))}")
    ol += 1
#%%