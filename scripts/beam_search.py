import numpy as np
import pandas as pd
pd.set_option('display.width', None)
pd.set_option('display.max_columns', None)

import matplotlib.pyplot as plt
plt.ion()

import chess
import chess.engine

import random
from joblib import Parallel, delayed

from chessbot.utils import random_init
from IPython.display import display, clear_output, SVG
import chess.svg

import io
from PIL import Image
import time

import chessbot.encoding as cbe
from chessbot.encoding import get_board_state, encode_worker

def show_board(board, flipped=False):
    clear_output(wait=True)
    display(SVG(chess.svg.board(board=board, flipped=flipped)))
    

SF_LOC = "C://Users/Bryan/stockfish-windows-x86-64-avx2/stockfish/stockfish-windows-x86-64-avx2.exe"
bot_color = np.random.uniform() < 0.5

from chessbot.model import load_model
net = load_model("C:/Users/Bryan/Data/chessbot_data/models/chess_model_multihead_v240_3.keras")
#import tensorflow as tf
#print("Built with CUDA?", tf.test.is_built_with_cuda())
#print("GPUs available:", tf.config.list_physical_devices('GPU'))
#net = tf.keras.models.load_model("C:/Users/Bryan/Data/chessbot_data/models/chess_model_multihead_v240_3.keras")

PREDS_CACHE = {}


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
            uncached = []
            for child in next_frontier:
                fen = child.board.fen()
                if fen not in PREDS_CACHE:
                    uncached.append((fen, child))
                else:
                    child.predicted_value = PREDS_CACHE[fen]
            
            if len(uncached) > 1024:
                results = Parallel(n_jobs=8, backend="loky")(
                    delayed(encode_worker)(fen) for fen, _ in uncached
                )
                fen_to_arr = dict(results)
                arrays = [fen_to_arr[fen] for fen, _ in uncached]
                to_predict = [child for fen, child in uncached]
            else:
                arrays = [encoder(child.board) for fen, child in uncached]
                to_predict = [child for fen, child in uncached]
                    
            print(f"{len(arrays)} boards on depth {depth}")
            # X = np.stack([board_to_vec(b) for b in boards])
            # preds = model.predict(X)
            if arrays:
                X = np.stack(arrays, axis=0)
                scores = model.predict(X, batch_size=512)
                scores = scores[model.output_names.index('value')]
            else:
                scores = []
            
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


def make_move(board, net, encoder):
    print("Thinking ...")
    start = time.time()
    tree = MoveTree(board)
    tree.expand_with_beam_search(net, get_board_state, beam_width=[8, 4, 4, 4], max_depth=4)
    tree.collapse()
    stop=time.time()
    print(f"Searched for {stop-start: <.3} seconds\n")
    candidates = sorted(tree.root.children, key=lambda c: c.predicted_value)
    if board.turn:
        candidates = candidates[: : -1]
        
    for c in candidates[:3]:
        san = board.san(c.move)
        print(san, f"{c.predicted_value: <.4}")
    
    return candidates[0].move


def board_to_array(board, flipped=False):
    """Convert a chess.Board into a numpy image array."""
    svg = chess.svg.board(board=board, flipped=flipped)
    # Convert SVG → PNG in memory using Pillow
    # (works without Cairo because Pillow can handle SVG strings)
    try:
        import cairosvg
        png_bytes = cairosvg.svg2png(bytestring=svg.encode("utf-8"))
        img = Image.open(io.BytesIO(png_bytes)).convert("RGBA")
    except ImportError:
        raise RuntimeError("You need cairosvg installed: pip install cairosvg")
    return np.array(img)


def play_game():
    #bot_color = np.random.uniform() < 0.5
    bot_color = True
    board = chess.Board()
    
    # Setup matplotlib window
    fig, ax = plt.subplots(figsize=(6,6))
    fig.canvas.manager.set_window_title("ChessBot Game Viewer")
    plt.axis("off")
    with chess.engine.SimpleEngine.popen_uci(SF_LOC) as engine:
        while not board.is_game_over():
            # Draw board
            arr = board_to_array(board, flipped=not bot_color)
            ax.clear()
            ax.imshow(arr)
            ax.axis("off")            
            plt.pause(0.5)
            plt.show()
            
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
                sf_move = random.choice(sf_eval[:2])['pv'][0]
                san = board.san(sf_move)
                print(f"\nStockfish plays {san}\n")
                board.push(sf_move)

    
if __name__ == "__main__":
    play_game()









    

