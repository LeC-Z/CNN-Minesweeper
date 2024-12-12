import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import numpy as np
import random

# ----------------------------
# Data generation utilities
# ----------------------------

def generate_minesweeper_board(height=16, width=30, num_mines=99):
    """
    Generate a full Minesweeper board:
    - Place `num_mines` mines randomly on a `height`x`width` grid.
    - Compute adjacent numbers for non-mine cells.
    Returns:
        board: 2D array of shape (height, width). For mines: -1, else number of adjacent mines 0-8.
    """
    # Initialize empty board
    board = np.zeros((height, width), dtype=int)
    # Place mines
    mine_positions = random.sample(range(height * width), num_mines)
    for pos in mine_positions:
        r = pos // width
        c = pos % width
        board[r, c] = -1  # -1 represents a mine

    # Compute adjacent mine counts
    directions = [(-1, -1), (-1, 0), (-1, 1),
                  (0, -1),           (0, 1),
                  (1, -1),  (1, 0),  (1, 1)]
    for r in range(height):
        for c in range(width):
            if board[r, c] == -1:
                continue
            count = 0
            for dr, dc in directions:
                nr, nc = r + dr, c + dc
                if 0 <= nr < height and 0 <= nc < width:
                    if board[nr, nc] == -1:
                        count += 1
            board[r, c] = count

    return board


def simulate_open_cells(board, max_opens=60):
    """
    Simulate opening some cells on the board. To make the training data helpful,
    we try to open cells that reveal some structure.
    
    Strategy:
    1. Randomly pick a cell that is not a mine and has a low number (like 0 or 1) to open first.
    2. Perform a pseudo "flood fill" of safe cells to simulate a more realistic opened scenario.
    3. Possibly open additional random safe cells to ensure we have enough revealed info.

    Returns:
        revealed_mask: boolean array (height, width), True if that cell is revealed
    """
    height, width = board.shape
    is_mine = (board == -1)
    
    # Choose a safe cell with a low number as a starting point
    candidates = [(r, c) for r in range(height) for c in range(width) 
                  if board[r,c] != -1 and board[r,c] <= 2]  # prioritize low numbers
    if len(candidates) == 0:
        # fallback: choose any safe cell
        candidates = [(r, c) for r in range(height) for c in range(width) if board[r,c] != -1]

    start = random.choice(candidates)
    
    revealed = set()
    stack = [start]
    directions = [(-1,0),(1,0),(0,-1),(0,1),(-1,-1),(-1,1),(1,-1),(1,1)]
    
    # Flood fill like reveal (for zero cells and their neighbors)
    while stack and len(revealed) < max_opens:
        r,c = stack.pop()
        if (r,c) in revealed:
            continue
        revealed.add((r,c))
        if board[r,c] == 0:
            # reveal neighbors
            for dr,dc in directions:
                nr,nc = r+dr,c+dc
                if 0 <= nr < height and 0 <= nc < width:
                    if board[nr,nc] != -1 and (nr,nc) not in revealed:
                        stack.append((nr,nc))
    
    # If we haven't revealed enough, randomly reveal more safe cells
    if len(revealed) < max_opens:
        safe_cells = [(r,c) for r in range(height) for c in range(width) 
                      if board[r,c] != -1 and (r,c) not in revealed]
        extra_opens = random.sample(safe_cells, min(len(safe_cells), max_opens-len(revealed)))
        revealed.update(extra_opens)
    
    revealed_mask = np.zeros((height, width), dtype=bool)
    for (r,c) in revealed:
        revealed_mask[r,c] = True
    
    return revealed_mask


def generate_training_example(height=16, width=30, num_mines=99, max_opens=60):
    """
    Generate one training example:
    Input channels:
        Channel 0: revealed mask (1 if revealed, else 0)
        Channel 1: revealed numbers (0-8 if revealed, else 0)
    Output:
        Label: binary mask of mines (1 if mine, else 0)
    """
    board = generate_minesweeper_board(height, width, num_mines)
    revealed_mask = simulate_open_cells(board, max_opens=max_opens)
    is_mine = (board == -1).astype(np.float32)
    
    # Prepare input
    # Channel 0: revealed_mask
    # Channel 1: numbers if revealed, else 0
    input_data = np.zeros((2, height, width), dtype=np.float32)
    input_data[0] = revealed_mask.astype(np.float32)
    number_layer = np.copy(board)
    number_layer[number_layer == -1] = 0  # mines become 0 here since not revealed anyway
    number_layer[~revealed_mask] = 0      # hide unrevealed
    input_data[1] = number_layer.astype(np.float32)
    
    label = is_mine  # shape (height, width)
    return input_data, label


# ----------------------------
# Dataset and Model
# ----------------------------

class MinesweeperDataset(Dataset):
    def __init__(self, samples=1000, height=16, width=30, num_mines=99, max_opens=60):
        self.samples = samples
        self.height = height
        self.width = width
        self.num_mines = num_mines
        self.max_opens = max_opens

    def __len__(self):
        return self.samples

    def __getitem__(self, idx):
        inp, lbl = generate_training_example(self.height, self.width, self.num_mines, self.max_opens)
        return torch.from_numpy(inp), torch.from_numpy(lbl)


class CNNMinePredictor(nn.Module):
    def __init__(self, in_channels=2, out_channels=1):
        super(CNNMinePredictor, self).__init__()
        # A simple CNN architecture
        self.net = nn.Sequential(
            nn.Conv2d(in_channels, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(64, out_channels, kernel_size=3, padding=1),
            # output is (1, H, W)
            nn.Sigmoid()  # predict probability of mine
        )

    def forward(self, x):
        return self.net(x)


# ----------------------------
# Training loop (example)
# ----------------------------

if __name__ == "__main__":
    # Hyperparameters
    batch_size = 8
    lr = 1e-3
    epochs = 2  # For demonstration, use more epochs in real training
    dataset_size = 200  # small dataset for demonstration
    train_dataset = MinesweeperDataset(samples=dataset_size, height=16, width=30, num_mines=99, max_opens=80)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

    model = CNNMinePredictor()
    optimizer = optim.Adam(model.parameters(), lr=lr)
    criterion = nn.BCELoss()

    model.train()
    for epoch in range(epochs):
        total_loss = 0
        for inputs, targets in train_loader:
            # inputs: [B,2,H,W], targets: [B,H,W]
            optimizer.zero_grad()
            outputs = model(inputs)  # [B,1,H,W]
            outputs = outputs.squeeze(1)  # [B,H,W]
            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

        avg_loss = total_loss / len(train_loader)
        print(f"Epoch {epoch+1}/{epochs}, Loss: {avg_loss:.4f}")

    print("Training complete.")
