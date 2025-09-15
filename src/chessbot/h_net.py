import os, json
import chess
import chess.engine
import pandas as pd
import numpy as np

import matplotlib.pyplot as plt
from datetime import datetime
import random, time
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

from chessbot.utils import format_time
from chessbot.encoding import compute_move_priors
import chessbot.utils as cbu
from chessbot.utils import get_pre_opened_game

from chessbot import SF_LOC, ENDGAME_LOC
import chess.syzygy

from pyfastchess import Board as fastboard

MODEL_DIR = "C:/Users/Bryan/Data/chessbot_data/models"

from stockfish import Stockfish
stockfish = Stockfish(path=SF_LOC)
stockfish.update_engine_parameters({"Threads": 4})
stockfish.set_depth(3)


from chessbot.encoding import sf_top_moves_to_values, values_to_priors


def analyze_position(board, topk=4, temp=0.02):
    stockfish.set_fen_position(board.fen())

    legal = board.legal_moves()
    K = min(topk, len(legal))
    sf_moves = stockfish.get_top_moves(K)

    # raw values for SF-returned moves
    vals_sf = sf_top_moves_to_values(sf_moves)

    # start output in SF order with raw scores
    out = []
    vals = []
    for m, v in zip(sf_moves, vals_sf):
        d = dict(m)
        d["score"] = float(v)   # raw in [-1, 1]
        out.append(d)
        vals.append(v)
    
    STM = board.side_to_move() == 'w'
    min_val = -1.0 if STM else 1.0
    
    # append missing legal moves with worst raw score
    have = {m["Move"] for m in sf_moves}
    for uci in legal:
        if uci not in have:
            out.append(
                {"Move": uci, "Centipawn": None, "Mate": None, "score": float(min_val)}
            )
            vals.append(min_val)
    
    # flip it its black's turn. they will pick moves good them (low/negative eval)
    signed_vals = vals if STM else [-v for v in vals]
    
    # compute priors over the full list (same order)
    priors = values_to_priors(signed_vals, temp=temp, mix=0.025)
    for d, p in zip(out, priors):
        d["prior"] = float(p)

    return out
    
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
    promo_logits  = layers.Dense(4,  name=f"{prefix}_promo")(x)

    return [from_logits, to_logits, piece_logits, promo_logits]


def build_conv_trunk(input_shape=(8, 8, 90), width=256, n_blocks=8, leak=0.05):
    """Convolutional trunk with residual blocks, returns feature map + pooled vector."""
    inputs = layers.Input(shape=input_shape, name="board")

    # fat first conv
    x = layers.Conv2D(width, 3, padding="same", use_bias=False, kernel_initializer=HE)(inputs)
    x = layers.BatchNormalization()(x)
    x = layers.LeakyReLU(alpha=leak)(x)

    # residual tower
    for _ in range(n_blocks):
        x = res_block(x, width, leak=leak)
    
    trunk_feat = layers.LeakyReLU(alpha=leak, name="trunk")(x)

    # global pooling: (8,8,width) -> (width,)
    trunk_vec = layers.GlobalAveragePooling2D(name="trunk_vec")(trunk_feat)

    return Model(inputs, [trunk_feat, trunk_vec], name="conv_trunk")


def make_conv_model(input_shape=(8, 8, 90), width=256, n_blocks=8):
    trunk = build_conv_trunk(input_shape=input_shape, width=width, n_blocks=n_blocks)
    inputs = trunk.input
    trunk_feat, trunk_vec = trunk.output

    # Heads
    val_out, _ = value_head(trunk_vec, hidden=512)
    best_outputs = policy_factor_head(trunk_vec, prefix="best", hidden=512)

    model = Model(inputs, [val_out] + best_outputs, name="conv_factorized")
    return model


def evaluate_terminal(b):
    reason, result = b.is_game_over()

    if reason == "none":
        raise Exception("Non terminal board found!")

    if reason == "checkmate":
        loser = b.side_to_move()
        return -1 if loser == "w" else 1
    return 0


def load_model(model_loc):
    model = keras.models.load_model(model_loc)
    return model


def get_legal_labels(board, moves):
    # normalize to UCI strings
    ucis = [m.uci() if hasattr(m, "uci") else str(m) for m in moves]
    N = len(ucis)
    if N == 0:
        z = np.zeros(0, dtype=np.int32)
        return {"from": z, "to": z, "piece": z, "promo": z}

    # fast path: use C++ batch
    f, t, pc, pr = board.moves_to_labels(ucis)
    return {
        "from":  np.asarray(f, dtype=np.int32),
        "to":    np.asarray(t, dtype=np.int32),
        "piece": np.asarray(pc, dtype=np.int32),
        "promo": np.asarray(pr, dtype=np.int32),
    }
#%%
losses = {
    "value": tf.keras.losses.MeanSquaredError(),
    "best_from": tf.keras.losses.CategoricalCrossentropy(from_logits=True),
    "best_to":   tf.keras.losses.CategoricalCrossentropy(from_logits=True),
    "best_piece":tf.keras.losses.CategoricalCrossentropy(from_logits=True),
    "best_promo":tf.keras.losses.CategoricalCrossentropy(from_logits=True),
}

loss_weights = {
    "value": 0.5,
    "best_from": 0.5,
    "best_to": 0.5,
    "best_piece": 0.5,
    "best_promo": 0.1,
}

opt = tf.keras.optimizers.Adam(1e-4)

#model = make_conv_model(input_shape=(8,8,70), width=256, n_blocks=8)
model = load_model(MODEL_DIR + "/conv_model_big_latest.h5")
model.compile(optimizer=opt, loss=losses, loss_weights=loss_weights)
model.summary()
#%%
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
    

def summarize_sf_block(games, elo, epoch):
    """
    Count results from the MODEL's perspective, but respect which color SF played:
      - g.sf_plays == True  -> SF is White, model is Black
      - g.sf_plays == False -> SF is Black, model is White
    g.outcome in {+1 (model win), 0 (draw), -1 (model loss)}
    """
    n = len(games)
    wins = draws = losses = 0

    # by model color
    col = defaultdict(lambda: {"n":0,"wins":0,"draws":0,"losses":0})

    lengths = []
    for g in games:
        lengths.append(g.board.history_size())
        
        model_color = "black" if g.sf_plays else "white"
        cs = col[model_color]
        cs["n"] += 1
        
        if g.outcome == 0:
            cs['draws'] += 1
            draws += 1
        if g.sf_plays and g.outcome == 1:
            cs['losses'] += 1            
            losses += 1
        if g.sf_plays and g.outcome == -1:
            cs['wins'] += 1
            wins += 1
        if not g.sf_plays and g.outcome == 1:
            cs['wins'] += 1
            wins += 1
        if not g.sf_plays and g.outcome == -1:
            cs['losses'] += 1
            losses += 1

    score = wins + 0.5 * draws
    wr = wins / n if n else 0.0

    avg_len = float(np.mean(lengths)) if lengths else float("nan")
    med_len = float(np.median(lengths)) if lengths else float("nan")

    row = {
        "epoch": int(epoch),
        "elo": int(elo),
        "n_games": n,
        "wins": wins,
        "draws": draws,
        "losses": losses,
        "score": float(score),
        "winrate": float(wr),           # model score per game (1/0.5/0)
        "avg_ply": avg_len,
        "med_ply": med_len,
        "white_wins":   col["white"]["wins"]   if "white" in col else float("nan"),
        "white_draws":  col["white"]["draws"]  if "white" in col else float("nan"),
        "white_losses": col["white"]["losses"] if "white" in col else float("nan"),
        "black_wins":   col["black"]["wins"]   if "black" in col else float("nan"),
        "black_draws":  col["black"]["draws"]  if "black" in col else float("nan"),
        "black_losses": col["black"]["losses"] if "black" in col else float("nan"),
    }
    return row


def print_sf_summary(row):
    print("\n=== SF Probe Summary ===")
    print(f"epoch={row['epoch']}  ELO={row['elo']}  games={row['n_games']}")
    print(f"W/D/L = {row['wins']}/{row['draws']}/{row['losses']}  "
          f"score={row['score']:.1f}  winrate={row['winrate']*100:.1f}%")
    if not np.isnan(row["avg_ply"]):
        print(f"avg ply={row['avg_ply']:.1f}  med ply={row['med_ply']:.1f}")
    if not np.isnan(row.get("white_wins", np.nan)):
        ww = row["white_wins"]; wd = row["white_draws"]; wl = row["white_losses"]
        bw = row["black_wins"]; bd = row["black_draws"]; bl = row["black_losses"]
        print(f"White W/D/L: {ww}/{wd}/{wl}   Black W/D/L: {bw}/{bd}/{bl}")
    print("========================\n")


def adjust_elo(curr_elo, row, up_step=100, down_step=100, min_games=32,
               up_thresh=0.55, down_thresh=0.45, min_elo=400, max_elo=3000):
    """
    Simple ladder:
      - if winrate >= up_thresh and enough games -> bump up
      - if winrate <= down_thresh and enough games -> bump down
    """
    if row["n_games"] < min_games:
        return curr_elo, "hold"

    wr = row["winrate"]
    if wr >= up_thresh:
        return min(max_elo, curr_elo + up_step), "up"
    if wr <= down_thresh:
        return max(min_elo, curr_elo - down_step), "down"
    return curr_elo, "hold"


def plot_sf_probe_progress(df):
    df = df.sort_values("epoch")
    fig, ax1 = plt.subplots(figsize=(8,4))
    ax1.plot(df["epoch"], df["winrate"]*100, marker='o', linewidth=1)
    ax1.set_ylabel("Win rate vs SF (%)")
    ax1.set_xlabel("Epoch")
    ax1.grid(True, alpha=0.3)
    ax2 = ax1.twinx()
    ax2.plot(df["epoch"], df["elo"], color='orange', alpha=0.7)
    ax2.set_ylabel("SF ELO challenge")
    plt.title("Probe performance vs Stockfish")
    plt.tight_layout()
    plt.show()


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
        lengths.append(g.board.history_size())

    if lengths:
        stats["avg_length"] = float(np.mean(lengths))

    # append to JSONL
    with open(log_path, "a") as f:
        f.write(json.dumps(stats) + "\n")

    return stats

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
            arr = np.array(vals, dtype=np.float32)
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


def random_init(plies=5):
    b = fastboard()
    p = 0
    counter = 0
    limit = int(3*plies)
    while p < plies:
        # safety in case it gets caught in some weird corner
        counter += 1
        if counter > limit:
            # really bad luck, just try again
            return random_init(plies=plies)
        
        moves = b.legal_moves()
        while not len(moves):
            b.unmake()
            moves = b.legal_moves()
            p -= 1

        b.push_uci(random.choice(moves))
        p += 1
    return b
    
    
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

        self.mat_adv_counter = 0
        self.draw_score = kwargs.get("draw_score", 0.0)
        
        self.sf_plays = kwargs.get("sf_plays", None)
        self.sample_scores = kwargs.get("sample_scores", False)

    def turn(self, return_bool=True):
        stm = self.board.side_to_move()
        if return_bool:
            return stm == 'w'
        else:
            return stm
    
    def short_fen(self, fen=None):
        return self.board.fen(include_counters=False)

    def check_termination(self):
        reason, result = self.board.is_game_over()
        if reason != 'none':
            print(self.game_id, "Game over detected:", reason, self.board.fen())
            self.outcome = evaluate_terminal(self.board)
            return True
    
        mat_diff = self.board.material_count()
        if abs(mat_diff) >= 12:
            self.mat_adv_counter += 1
        else:
            self.mat_adv_counter = 0
    
        if self.mat_adv_counter >= 28:
            print(self.game_id, "Material cutoff:", mat_diff)
            self.outcome = 1.0 if mat_diff > 0 else -1.0
            return True
        
        # Syzygy probe if few pieces
        if self.board.piece_count() <= 5:
            # may not work so just go as normal
            try:
                outcomes = {-2: -1, -1:-1, 0:0, 1:1, 2:1}
                with chess.syzygy.open_tablebase(ENDGAME_LOC) as tablebase:
                    # gotta flip back to python chess here
                    chess_board = chess.Board(self.board.fen())
                    table_res = tablebase.probe_wdl(chess_board)
                    
                table_res = table_res if chess_board.turn else -1*table_res
                self.outcome = outcomes[table_res]
                print(self.game_id, "Endgame Solution Found:", self.outcome, self.board.fen())
                return True
            except:
                pass
           
        hs = self.board.history_size()
        if hs > self.move_limit:
            print(self.game_id, "Move limit reached:", hs)
            self.outcome = 0.0
            return True
    
        return False
        
    def record_position(self):
        self.training_X.append({"board": self.board.stacked_planes(5)})
        
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
        
        moves_list = self.board.legal_moves()
        
        # prevent threefold by storing pos + move and if that combo has been encountered
        if self.prevent_repetition:
            repeated = []
            for lm in moves_list:
                if self.pos_move_counter[(short_fen, lm)] >= 1:
                    repeated.append(lm)
                    
                # # check the san too
                if self.move_counter[lm] >= 1:
                    repeated.append(lm)
            
            # if there are no non-repeating moves, end game in draw
            if len(repeated) == len(moves_list):
                self.outcome = self.draw_score
                self.finalize_outcome()
                return
            
            if repeated:
                moves_list = [m for m in moves_list if m not in repeated]
        
        # another guard rail
        if not moves_list:
            # Game is effectively over, finalize as draw if no moves
            self.outcome = self.draw_score
            self.finalize_outcome()
            return

        # use the prior finder to get best move
        legal_labels = get_legal_labels(self.board, moves_list)
        scores = compute_move_priors(preds, legal_labels)
        
        # pick the move
        if not self.sample_scores or (scores.max() > 0.15):
            top_idx = np.argmax(scores)
        else:
            top_idx = np.random.choice(len(scores), p=scores)
            
        chosen = moves_list[top_idx]
        
        # push best move and update lists
        self._make_move(chosen, short_fen)
        return
    
    def _make_move(self, move, short_fen):
        self.move_counter[move] += 1
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
        
        self.board.push_uci(move)
        
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
        show_board = chess.Board(self.board.fen())
        clear_output(wait=True)
        display(SVG(chess.svg.board(board=show_board, flipped=flipped)))
        time.sleep(sleep)


def _normalize(v, eps=1e-12):
    s = float(v.sum())
    if s <= 0:
        return v
    return (v / (s + eps)).astype(np.float32)


def get_boosting_data(game, result, weights):
    """
    game: has .board (pyfastchess Board) and training_X[-1]
    result: list of dicts like {'Move': 'e2e4', 'score': raw_value[-1..1], 'prior': prob}
            If 'prior' missing, we'll build priors from all 'score's.
    weights: base loss weights dict
    """
    # moves and priors in the order you built `result`
    all_moves = [r['Move'] for r in result]
    priors = [r.get('prior') for r in result]

    # value target = raw value of top move (your current convention)
    value_target = float(result[0]['score'])

    # map legal moves -> head indices
    f, t, pc, pr = game.board.moves_to_labels(all_moves)
    f = np.asarray(f); t = np.asarray(t); pc = np.asarray(pc); pr = np.asarray(pr)

    # aggregate priors into head distributions
    p_from  = np.zeros(64, dtype=np.float64)
    p_to    = np.zeros(64, dtype=np.float64)
    p_piece = np.zeros(6,  dtype=np.float64)
    p_promo = np.zeros(4,  dtype=np.float64)

    np.add.at(p_from,  f,  priors)
    np.add.at(p_to,    t,  priors)
    np.add.at(p_piece, pc, priors)
    np.add.at(p_promo, pr, priors)

    # normalize each head to sum to 1 (only seen indices have mass)
    p_from  = _normalize(p_from)
    p_to    = _normalize(p_to)
    p_piece = _normalize(p_piece)
    p_promo = _normalize(p_promo)

    # inputs
    moreX = game.training_X[-1]

    # targets (soft)
    moreY = {
        'value': np.array([value_target], dtype=np.float32),
        'best_from': p_from,
        'best_to': p_to,
        'best_piece': p_piece,
        'best_promo': p_promo,
    }

    # weights: keep your scheme; boost promo only if any nonzero mass on 1..3
    moreW = deepcopy(weights)
    moreW['value'] = 0.3
    moreW['best_from'] = 0.5
    moreW['best_to'] = 0.5
    moreW['best_piece'] = 0.5
    moreW['best_promo'] = 7.0 if p_promo[1:].sum() > 0 else 0.01

    return moreX, moreY, moreW

    
def self_play_batch(model, n_games=32, show=True, return_games=False, **kwargs):
    play_sf = kwargs.get("play_sf", False)
    prevent_repetition = kwargs.get("prevent_repetition", True)
    sample_scores = kwargs.get("sample_scores", False)
    topk = kwargs.get("topk", 4)
    
    # this adds in the true best move for instant gradient boost
    instant_boosting = kwargs.get("instant_boosting", False)
    if instant_boosting:
        weights = deepcopy(model.compiled_loss._loss_weights)
        if isinstance(weights, list):
            weights = dict(zip(model.output_names, weights))
    
    # load games
    games = []
    for i in range(n_games):
        move_limit = 120
        n_init = kwargs.get("n_init", np.random.randint(2, 6))
        # # half the games from the structured fens, half random inits
        if i < n_games/2:
            b = random_init(n_init)
        else:
            b = get_pre_opened_game()
            
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
            sf_turn = play_sf and (game.sf_plays == game.turn())
            result = None
            # run stockfish if it is needed
            if sf_turn or instant_boosting:
                result = analyze_position(game.board, topk=topk)
        
            # Boosting
            if instant_boosting:
                moreX, moreY, moreW = get_boosting_data(game, result, weights)
                train_X.append(moreX); train_Y.append(moreY); train_W.append(moreW)
        
            # Play the move
            if sf_turn:
                game.force_move(result[0]['Move'])
                
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
    model, n_games=8, show=True, return_games=True,
    prevent_repetition=True, instant_boosting=True
)

i, o, w, g = self_play_batch(
    model, n_games=1, show=True, return_games=True, n_init=0,
    play_sf=True, instant_boosting=False, prevent_repetition=False
)

i, o, w, g = self_play_batch(
    model, n_games=1, show=True, return_games=True, n_init=0,
    play_sf=False, instant_boosting=True, prevent_repetition=False, topk=6
)

#%%
all_evals = pd.DataFrame()
all_results = {}
loop_times = []
ol = 500

model_tag = 'conv_model_big'
MODEL_DIR = "C:/Users/Bryan/Data/chessbot_data/models/"
model_files = [f for f in os.listdir(MODEL_DIR) if model_tag in f]
model_paths = [os.path.join(MODEL_DIR, f) for f in model_files]
curr_elo_challenge = 400
#%%
while ol <= 1500:
    start_loop = time.time()
    show = False #ol % 9 == 0
    topk = 2 + ol % 5
    print(f"Outer Loop {ol}")
    if ol % 2 == 0:
        print("------- Playing vs Stockfish -------")
        inputs, outputs, weights, games = self_play_batch(
            model, n_games=192, show=show, return_games=True, n_init=2,
            play_sf=True, instant_boosting=True,
            prevent_repetition=False, topk=topk
        )
    
    else:
        print("------- Playing vs Self -------")
        inputs, outputs, weights, games = self_play_batch(
            model, n_games=96, show=show, return_games=True, n_init=2, topk=topk,
            instant_boosting=True, prevent_repetition=True, sample_scores=False
        )
        
    stats = log_game_stats(games, epoch=ol)
    print(stats)
    
    if ol % 5 == 0:
        eval_df = cbu.score_game_data(model, inputs, outputs)
        all_evals = pd.concat([all_evals, eval_df])
    
        if len(all_evals) > 2:
            cbu.plot_training_progress(all_evals.iloc[1:, :])
    
    # train (no val)
    model.fit(
        inputs, outputs, sample_weight=weights,
        batch_size=256, epochs=1, shuffle=True
    )
    
    if ol % 25 == 0:
        # more frequent checkpointing
        model_out = os.path.join(MODEL_DIR, f"{model_tag}_latest.h5")
        model.save(model_out)
        
        # --- Probe block vs SF ---
        stockfish.set_elo_rating(curr_elo_challenge)
        stockfish.set_depth(1)
        print(f"------- Playing vs Stockfish ({curr_elo_challenge} ELO) -------")
        inputs, outputs, weights, games = self_play_batch(
            model, n_games=64, show=True, return_games=True,
            play_sf=True, instant_boosting=False, prevent_repetition=True
        )
        
        # summarize & log
        row = summarize_sf_block(games, elo=curr_elo_challenge, epoch=ol)
        print_sf_summary(row)
        
        # stash to running logs
        all_results.setdefault("sf_probe_log", [])
        all_results["sf_probe_log"].append(row)
        
        # keep a DataFrame too
        if "sf_probe_df" not in all_results:
            all_results["sf_probe_df"] = pd.DataFrame([row])
        else:
            all_results["sf_probe_df"] = pd.concat(
                [all_results["sf_probe_df"], pd.DataFrame([row])], ignore_index=True
            )
        
        sf_df = all_results["sf_probe_df"]
        if len(sf_df) > 1:
            cbu.plot_sf_simple(sf_df)
            
        # ladder: adjust elo
        # new_elo, action = adjust_elo(
        #     curr_elo_challenge, row, min_games=48, up_thresh=0.55, down_thresh=0.40
        # )
        
        # if new_elo != curr_elo_challenge:
        #     print(f"→ ELO ladder: {curr_elo_challenge} → {new_elo} ({action})")
        # curr_elo_challenge = new_elo
        
        # restore engine params for learning
        stockfish = Stockfish(path=SF_LOC)
        stockfish.update_engine_parameters({"Threads": 4, "Contempt": 30})
        stockfish.set_depth(7)
        
    if ol % 100 == 0:
        model_out = os.path.join(MODEL_DIR, f"{model_tag}_v{ol}.h5")
        model.save(model_out)
        
    stop_loop = time.time()
    loop_time = stop_loop - start_loop
    loop_times.append(loop_time)
    print(f"*** Loop {ol} completed in {format_time(loop_time)}")
    print(f"*** Avg loop time: {format_time(np.mean(loop_times))}")
    ol += 1

#%%


    
