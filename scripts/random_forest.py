import numpy as np
import pandas as pd
pd.set_option('display.width', None)
pd.set_option('display.max_columns', None)

import matplotlib.pyplot as plt
plt.ion()

import chess
import chess.engine

import math, random
import os, pickle

from chessbot.utils import random_init
from IPython.display import display, clear_output, SVG
import chess.svg
def show_board(board, flipped=False):
    clear_output(wait=True)
    display(SVG(chess.svg.board(board=board, flipped=flipped)))
    
#%%


board = random_init(6)
show_board(board)

import numpy as np
import chess


def board_to_vec(board):
    arr = np.zeros(64)
    piece_count = np.zeros(12)
    for color in [1, 0]:
        value = 1 if color else -1
        for piece in [1, 2, 3, 4, 5, 6]:
            pieces = list(board.pieces(piece, color))
            if pieces:
                arr[pieces] = value * piece
                if color:
                    piece_count[piece-1] = len(pieces)
                else:
                    piece_count[5+piece] = len(pieces)
    
    meta = [
        1.0*board.turn,
        1.0*board.has_kingside_castling_rights(chess.WHITE),
        1.0*board.has_queenside_castling_rights(chess.WHITE),
        1.0*board.has_kingside_castling_rights(chess.BLACK),
        1.0*board.has_queenside_castling_rights(chess.BLACK),
        len(board.move_stack)
    ]
    
    ep = np.zeros(8)
    if board.ep_square is not None:
        ep[board.ep_square // 8] = 1.0
    
    return np.concatenate([arr, piece_count, np.array(meta), ep], dtype=np.float32)


names  = [f'square{s}' for s in range(64)]
names += [f'N_White_piece{n}' for n in [1, 2, 3, 4, 5, 6]]
names += [f'N_Black_piece{n}' for n in [1, 2, 3, 4, 5, 6]]
names += ['turn', 'KW', 'QW', 'KB', 'QB', 'N_moves']
names += [f'ep{r}' for r in range(8)]

#%%
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import train_test_split

# Create and fit model
model = RandomForestRegressor(n_jobs=-1)
X = board_to_vec(chess.Board()).reshape(1, -1)
model.fit(X, [0])
all_MSE = []
#%%
from IPython.display import display, clear_output, SVG
import chess.svg, chess.pgn, random
import chessbot.utils as cbu
from chessbot.encoding import score_to_cp_white
SF_LOC = "C://Users/Bryan/stockfish-windows-x86-64-avx2/stockfish/stockfish-windows-x86-64-avx2.exe"
bot_color = np.random.uniform() < 0.5

def plot_preds(y_true, y_pred):
    plt.scatter(y_true, y_pred, alpha=0.8)
    plt.plot([min(y_true), max(y_true)], [min(y_true), max(y_true)], color='red', linestyle='--', label='Perfect Prediction')
    plt.xlabel("True Values")
    plt.ylabel("Predicted Values")
    plt.title("True vs Predicted")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()
    

def train_loop(): 
    board = chess.Board()
    all_X = []
    all_y = []
    with chess.engine.SimpleEngine.popen_uci(SF_LOC) as engine:
        while not board.is_game_over():
            sf_eval = engine.analyse(
                board, multipv=10, limit=chess.engine.Limit(depth=3),
                info=chess.engine.Info.ALL
            )
            
            moves, these_X, these_y = [], [], []
            for mv in sf_eval:
                board.push(mv['pv'][0])
                moves.append(mv['pv'][0])
                these_X.append(board_to_vec(board))
                these_y.append(score_to_cp_white(mv['score']))
                board.pop()

            board.push(random.choice(list(board.legal_moves)))
            all_y += these_y
            all_X += these_X
    

    return all_X, all_y

all_X, all_y = [], []
for _ in range(100):
    X, y = train_loop()
    all_X += X
    all_y += y
    
    
X = np.stack(all_X, axis=0)
y = np.stack(all_y, axis=0)
model.fit(X, y)

test_X, test_y = [], []
for _ in range(40):
    X, y = train_loop()
    test_X += X
    test_y += y

test_X = np.stack(test_X, axis=0)
test_y = np.stack(test_y, axis=0)
print("predicting ...")
preds = model.predict(test_X)
plot_preds(test_y, preds)
MSE = np.mean((preds - test_y)**2)
print(f"MSE {MSE}")

1 - (np.mean((preds > 0) & (test_y < 0)) + np.mean((preds < 0) & (test_y > 0)))

# plt.plot(range(1, len(all_MSE)+1), all_MSE, marker='o')
# plt.title("MSE Over Time")
# plt.xlabel("Epoch")
# plt.ylabel("MSE")
# plt.grid(True)
# plt.tight_layout()
# plt.show()
#%%
importances = model.feature_importances_

# assuming you have feature names

importances = model.feature_importances_
indices = np.argsort(importances)[::-1]  # descending order

print("🔝 Top 20 Most Important Features:")
for i in range(20):
    print(f"{names[indices[i]]}: {importances[indices[i]]:.4f}")


#%%

#MCTS
import chess
from collections import defaultdict
import random
import numpy as np

def softmax(x):
    x = np.array(x)
    exps = np.exp(x - np.max(x))  # subtract max for numerical stability
    return exps / np.sum(exps)


class MCTSNode:
    def __init__(self, board, parent=None, move=None):
        self.board = board
        self.parent = parent
        self.move = move
        self.children = {}
        self.visits = 0
        self.value_sum = 0.0

    def is_fully_expanded(self):
        return len(self.children) == len(list(self.board.legal_moves))

    def average_value(self):
        return self.value_sum / self.visits if self.visits > 0 else 0.0

    def expand(self):
        for move in self.board.legal_moves:
            if move not in self.children:
                new_board = self.board.copy()
                new_board.push(move)
                self.children[move] = MCTSNode(new_board, parent=self, move=move)

    def best_child(self):
        return max(self.children.values(), key=lambda c: c.average_value())


def simulate(model, root, max_depth=5):
    path = []
    node = root
    depth = 0

    # SELECTION + EXPANSION
    while depth < max_depth:
        path.append(node)

        if not node.children:
            node.expand()
            break  # expand only 1 node per simulation

        # Pick a random child (or softmax/PUCT later)
        node = random.choice(list(node.children.values()))
        depth += 1

    # LEAF EVALUATION
    leaf_node = node
    position_tensor = get_board_state(leaf_node.board).reshape(1, 8, 8, 9)
    value = model.predict(position_tensor, verbose=0)
    value = value[model.output_names.index('value')].item()

    # BACKPROPAGATION
    for i, node in enumerate(reversed(path)):
        node.visits += 1
        sign = -1 if i % 2 == 1 else 1  # flip value each ply
        node.value_sum += sign * value
        

def choose_move(board, n_sims=500):
    root = MCTSNode(chess.Board())
    
    print(f"Running {n_sims} simulations...")
    for _ in range(n_sims):
        simulate(model, root, max_depth=5)
    
    return max(root.children.items(), key=lambda x: x[1].visits)[0]



best_move = choose_move(random_board())



from chessbot.utils import get_all_board_features, random_init

def test_encoder(funcs):
    start = time.time()
    for _ in range(5000):
        b = random_init(5)
        for f in funcs:
            f(b)
    stop = time.time()
    return stop - start

f1 = [board_to_vec]
f2 = [get_board_state]
f3 = [get_all_board_features]
f4 = [encode_board, get_all_board_features]
f5 = [encode_board]

print("f1", test_encoder(f1))
print("f2", test_encoder(f2))
print("f3", test_encoder(f3))
print("f4", test_encoder(f4))
print("f5", test_encoder(f5))


def test(func, m, r1, r2, batch_size=None):
    start = time.time()
    for a in range(r1):
        X = []
        for aa in range(r2):
            b = random_init(5)
            X.append(func(b))
        if batch_size is None:
            preds = m.predict(np.stack(X, axis=0))
        else:
            preds = m.predict(np.stack(X, axis=0), batch_size=batch_size)
    stop = time.time()
    return stop - start


net = tf.keras.models.load_model("C:/Users/Bryan/Data/chessbot_data/models/chess_model_multihead_v240_3.keras")
from chessbot.model import load_model
net2 = load_model("C:/Users/Bryan/Data/chessbot_data/models/value_policy_model_v450.h5")
test(board_to_vec, model, 100, 50)
test(get_board_state, net, 100, 50, batch_size=50)
test(encode_board, net2, 100, 50, batch_size=50)
#%%
from IPython.display import display, clear_output, SVG
import chess.svg, chess.pgn, random
import chessbot.utils as cbu

bot_color = np.random.uniform() < 0.5
board = chess.Board()
board = cbu.random_init(4)
root = MCTSNode(board)
data = []
with chess.engine.SimpleEngine.popen_uci(SF_LOC) as engine:
    #v = predict_value_cached_priors(root, engine)
    while not board.is_game_over():
        clear_output(wait=True)
        display(SVG(chess.svg.board(board=board, flipped=not bot_color)))
        
        sf_eval = engine.analyse(
            board, multipv=10, limit=chess.engine.Limit(depth=1),
            info=chess.engine.Info.ALL
        )
        
        if board.turn == bot_color:
            best_move, df, root = choose_move(root, engine, model, board, max_sims=100)
            data.append(df)
            root = advance_root(root, best_move)
            print("\nStockfish top 3 moves: ")
            moves = sf_eval[:3]
            for i, mv in enumerate(moves):
                san = board.san(mv['pv'][0])
                sc = cbe.pawns_to_winprob(cbe.score_to_cp_white(mv['score']))
                
                print(f"{i+1}. {san}: {np.round(sc, 3)}")
                
            print()
            board.push(best_move)
            
        else:
            time.sleep(0.5)
            sf_move = sf_eval[0]['pv'][0]
            san = board.san(sf_move)
            print(f"\nStockfish plays {san}\n")
            #board.push(random.choice(list(board.legal_moves)))
            board.push(sf_move)
            root = advance_root(root, sf_move)

#%%
PREDS_CACHE = {}
#%%
class MoveNode:
    def __init__(self, board, move=None, parent=None):
        self.board = board
        self.move = move
        self.parent = parent
        self.children = []
        self.temp_children = {}  # move: MoveNode
        self.predicted_value = None


    def is_leaf(self):
        return len(self.children) == 0

    def collapse(self, cache: dict):
        if self.cached_score is not None:
            return self.cached_score

        if self.is_leaf():
            key = self.board.fen()
            self.cached_score = cache[key]
            return self.cached_score

        scores = [child.collapse(cache) for child in self.children]
        self.cached_score = max(scores) if self.board.turn == chess.WHITE else min(scores)
        return self.cached_score
    
    def expand_candidate(self, move, board):
        self.temp_children[move] = MoveNode(board, move=move, parent=self)

    def _collapse_minimax(self):
        if not self.children:
            return self.predicted_value

        child_scores = [child._collapse_minimax() for child in self.children]
        # flip for negamax
        scores = [-s for s in child_scores]
        best = max(scores)
        self.predicted_value = best
        return best
    
    
class MoveTree:
    def __init__(self, board):
        self.root = MoveNode(board)

    def expand_with_beam_search(self, model, encoder, beam_width, max_depth=4):
        frontier = [self.root]

        for depth in range(max_depth):
            next_frontier = []

            # 1. Expand all children as temp
            for node in frontier:
                legal_moves = list(node.board.legal_moves)
                for mv in legal_moves:
                    new_board = node.board.copy(stack=False)
                    new_board.push(mv)
                    node.expand_candidate(mv, new_board)

                next_frontier.extend(node.temp_children.values())

            if not next_frontier:
                break  # no more moves anywhere

            # 2. Batch eval
            arrays = []
            to_predict = []
            for child in next_frontier:
                child_fen = child.board.fen()
                if child_fen not in PREDS_CACHE:
                    arrays.append(encoder(child.board))
                    to_predict.append(child)
                else:
                    child.predicted_value = PREDS_CACHE[child_fen]
                    
            print(f"{len(arrays)} boards on depth {depth}")
            # X = np.stack([board_to_vec(b) for b in boards])
            # preds = model.predict(X)
            if arrays:
                X = np.stack(arrays, axis=0)
                scores = model.predict(X, batch_size=512)
                scores = scores[model.output_names.index('value')]
            else:
                scores = []

            # 3. Assign predictions
            for child, score in zip(to_predict, scores):
                fen = child.board.fen()
                child.predicted_value = score.item()
                PREDS_CACHE[fen] = score.item()

            # 4. Select top-k children per node
            new_frontier = []
            for node in frontier:
                candidates = list(node.temp_children.values())
                # Correct sorting based on side to move
                reverse = node.board.turn  # True = high-to-low (White), False = low-to-high (Black)
                sorted_kids = sorted(candidates, key=lambda x: x.predicted_value, reverse=reverse)
                node.children = sorted_kids[:beam_width[depth]]
                new_frontier.extend(node.children)
                node.temp_children = {}  # clear temp
            
            # 5. Set frontier for next layer
            frontier = new_frontier
            for node in frontier + [n for p in frontier for n in p.children]:
                frontier.append(node)


    def collapse(self):
        return self.root._collapse_minimax()

    def build(self):
        self._expand(self.root, self.depth)

    def _expand(self, node, depth):
        if node.board.is_game_over():
            result = node.board.result()
            if result == '1-0':
                node.predicted_value = 1
            elif result == '0-1':
                node.predicted_value = -1
            else:
                node.predicted_value = 0
            return
    
        if depth == 0:
            return
    
        if node.children:
            for child_node in node.children:
                self._expand(child_node, depth - 1)
    
        for move in list(node.board.legal_moves):
            board_copy = node.board.copy()
            board_copy.push(move)
            child = MoveNode(board_copy, move, node)
            node.children.append(child)
            self._expand(child, depth - 1)

    def get_leaf_nodes(self):
        leaves = []

        def dfs(node):
            if node.is_leaf():
                leaves.append(node)
            else:
                for child in node.children:
                    dfs(child)

        dfs(self.root)
        return leaves

    def evaluate(self, model, encoder):
        leaves = self.get_leaf_nodes()
        boards = [leaf.board for leaf in leaves]
        X = np.stack([encoder(b) for b in boards], axis=0)
        scores = model.predict(X, batch_size=512)
        scores = scores[model.output_names.index('value')]
        self.prediction_cache = {
            b.fen(): s for b, s in zip(boards, scores)
        }

    def get_best_move(self):
        self.root.collapse(self.prediction_cache)
        selecter = max if self.root.board.turn else min
        best_child = selecter(self.root.children, key=lambda child: child.cached_score)
        return best_child.move, best_child.cached_score


import time
def make_move(board, net, encoder):
    print("Thinking ...")
    start = time.time()
    tree = MoveTree(board)
    tree.expand_with_beam_search(net, get_board_state, beam_width=[16, 8, 4], max_depth=3)
    tree.collapse()
    stop=time.time()
    print(f"Searched for {stop-start: <.3} seconds")
    candidates = sorted(tree.root.children, key=lambda c: c.predicted_value)
    if board.turn:
        candidates = candidates[: : -1]
        
    for c in candidates[:3]:
        san = board.san(c.move)
        print(san, f"{c.predicted_value: <.4}")
    
    return candidates[0].move
    
    
#%%

from IPython.display import display, clear_output, SVG
import chess.svg, chess.pgn, random
import chessbot.utils as cbu
import chessbot.encoding as cbe

bot_color = np.random.uniform() < 0.5
board = chess.Board()
#board = cbu.random_init(4)
with chess.engine.SimpleEngine.popen_uci(SF_LOC) as engine:
    while not board.is_game_over():
        clear_output(wait=True)
        display(SVG(chess.svg.board(board=board, flipped=not bot_color)))
        
        sf_eval = engine.analyse(
            board, multipv=10, limit=chess.engine.Limit(depth=1),
            info=chess.engine.Info.ALL
        )
        
        if board.turn == bot_color:
            best_move = make_move(board, net, get_board_state)

            print("\nStockfish top 3 moves: ")
            moves = sf_eval[:3]
            for i, mv in enumerate(moves):
                san = board.san(mv['pv'][0])
                sc = cbe.pawns_to_winprob(cbe.score_to_cp_white(mv['score']))
                
                print(f"{i+1}. {san}: {np.round(sc, 3)}")
                
            print()
            board.push(best_move)
            
        else:
            time.sleep(0.5)
            sf_move = random.choice(sf_eval[:2])['pv'][0]
            san = board.san(sf_move)
            print(f"\nStockfish plays {san}\n")
            board.push(sf_move)

    

