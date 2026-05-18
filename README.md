# DSA 210 Project  
## Tactical Blindspots: A Personalized Chess Blunder Analyzer and Puzzle Recommender

### Does my pattern of tactical mistakes reflect systematic blind spots that can be targeted with personalized puzzle training?

---

## Overview

Final submission documents:

- `final_report_demir.md`
- `final_report_demir.docx`

This project investigates the tactical errors in my personal chess game history (Lichess username: **iamtheobama**) and examines whether these mistakes reflect systematic blind spots in my game. By analyzing centipawn loss and blunder patterns, the project transforms personal game data into targeted puzzle recommendations.

The analysis uses Stockfish engine evaluations to detect blunders and near-misses. Matching these positions with similar puzzles from the Lichess Open Puzzle Database using K-Nearest Neighbors (KNN) is done in the machine learning stage of the project.

---

## Terminology

- **FEN (Forsyth–Edwards Notation):** A compact string representing a chess board position
- **Centipawn Loss (CPL):** The loss in evaluation (in centipawns) caused by a suboptimal move; 1 pawn = 100 centipawns
- **Blunder:** A move with centipawn loss ≥ 200 (a serious tactical error)
- **Mistake:** A move with centipawn loss between 100 and 199
- **Inaccuracy:** A move with centipawn loss between 50 and 99
- **KNN:** K-Nearest Neighbors; a similarity algorithm used to match blunder positions with training puzzles
- **ECO:** Encyclopedia of Chess Openings code (e.g., B23 = Sicilian Defense: Closed)
- **Time Control:** Format `base+increment` in seconds (e.g., `60+0` = 1-minute bullet)

---

## Motivation

Chess improvement requires identifying and fixing specific weaknesses, not just playing more games. I have played **355 rated games** on Lichess across bullet, blitz, and rapid formats, with an overall win rate near 50%. However, a striking **63.7% of games end by time forfeit**, suggesting that time pressure leads to tactical oversights rather than genuinely lost positions.

This project aims to determine whether my blunders cluster around specific tactical motifs (pins, forks, back-rank mates, etc.) and whether targeted puzzle training for those specific patterns can systematically improve performance.

### Why this dataset?

Chess games on Lichess are recorded in PGN format with full move history, opening classification, and time control metadata. This makes it possible to combine personal performance data with the Lichess Open Puzzle Database, which classifies over 5.5 million puzzles by tactical theme. The combination allows us to go beyond aggregate statistics and build a data-driven, personalized training pipeline.

---

## Research Question

What are my specific tactical blind spots, and can a KNN-based puzzle recommender trained on my blunder patterns improve my tactical accuracy?

---

## Hypotheses

- **H₀ (Null Hypothesis):**
  My blunders are distributed randomly across tactical motifs, with no significant clustering.

- **H₁ (Opening Performance):**
  My win rate varies significantly across opening families, reflecting positional blind spots in specific structures.

- **H₂ (Time Forfeit Rate):**
  The proportion of games ending by time forfeit is significantly higher than 50%, indicating that time management — not pure tactical ability — is a primary loss factor.

- **H₃ (Puzzle Matching Feasibility):**
  Blunder positions can potentially be matched to Lichess puzzles via similarity algorithms, which is explored in the machine learning stage of the project.

---

## Data Sources

### Primary Dataset (Personal Game History)
Lichess Game Export — username: **iamtheobama**

Collected via the [Lichess API](https://lichess.org/api):
```
GET https://lichess.org/api/games/user/iamtheobama?evals=true&clocks=true
```

File: `lichess_iamtheobama_2026-04-14.pgn`

Contains **355 rated games** with:
- Full PGN move sequences
- Opening classification (ECO code + name)
- Time control format
- Game termination type (Normal / Time forfeit)
- Rating before and after each game

### Secondary Dataset (Lichess Open Puzzle Database)
Source: [https://database.lichess.org/#puzzles](https://database.lichess.org/#puzzles)

The Lichess Open Puzzle Database contains over 5.5 million puzzles. However, processing and searching through a 5.5M row dataset is computationally and practically extremely difficult for this scope. Therefore, a subset of **50 puzzles** was sampled and placed in `DATA/raw/50 Puzzles.xlsx`.

This 50-puzzle subset contains:
- FEN (starting position)
- Moves (solution sequence)
- Rating, popularity, themes (e.g., `fork`, `pin`, `backRankMate`)
- Opening tags

---

## Data Preparation

The PGN file and puzzle database are processed with the following pipeline:

```python
import chess
import chess.pgn
import pandas as pd
from stockfish import Stockfish

# --- Step 1: Parse PGN ---
games = []
with open("DATA/raw/lichess_iamtheobama_2026-04-14.pgn") as f:
    while True:
        game = chess.pgn.read_game(f)
        if game is None:
            break
        games.append(game)

# --- Step 2: Stockfish evaluation per move ---
sf = Stockfish(depth=15)
records = []
for game in games:
    board = game.board()
    for move in game.mainline_moves():
        fen_before = board.fen()
        sf.set_fen_position(fen_before)
        eval_before = sf.get_evaluation()
        board.push(move)
        sf.set_fen_position(board.fen())
        eval_after = sf.get_evaluation()
        cpl = max(0, eval_before.get("value", 0) - eval_after.get("value", 0))
        records.append({
            "fen": fen_before,
            "move": move.uci(),
            "cpl": cpl,
            "is_blunder": cpl >= 200
        })

df_moves = pd.DataFrame(records)
df_moves.to_csv("DATA/processed/move_evaluations.csv", index=False)
```

- Final move-level dataset: ~15,000 moves across 355 games
- Blunders (CPL ≥ 200) extracted as puzzle seed positions

---

## Exploratory Data Analysis (EDA)

The EDA includes:

**Game-level analysis:**
- Win/loss/draw distribution by time control (Bullet / Blitz / Rapid)
- Termination type analysis (Time forfeit vs. Normal)
- Opening family win rates (18 families with ≥ 5 games)
- ECO group performance (A, B, C, D)

**Move-level analysis (post Stockfish evaluation):**
- Distribution of centipawn loss across all moves
- Blunder rate by game phase (opening / middlegame / endgame)
- Blunder rate by color (White vs. Black)
- Time-pressure correlation: does CPL spike in later moves?

**Correlation analysis:**
- Opponent ELO vs. average CPL
- Time control vs. blunder rate
- Opening choice vs. game outcome

### Key Findings

- **355 games** played (339 Bullet, 9 Blitz, 7 Rapid)
- **Overall win rate: 49.3%** in Bullet; 55.6% in Blitz; 42.9% in Rapid
- **63.7% of games end by time forfeit** — the highest single loss factor
- **Strongest openings:** Scotch Game (83.3%), Petrov's Defense (62.5%), Philidor Defense (60.0%)
- **Weakest openings:** Modern Defense (25.0%), Nimzo-Larsen Attack (28.6%), Ruy Lopez (33.3%)
- **Sicilian Defense** is the most-played opening family (100 games, 54% win rate)
- ECO B openings (Semi-Open) account for 165 games — the largest group

---

## Hypothesis Testing

### Method

- **H₁ (Opening Performance):** One-way ANOVA across opening families to test for significant win rate differences
- **H₂ (Time Forfeit Rate):** One-sample proportion z-test against 50% baseline
- **H₃ (Blunder Clustering):** Evaluated in the machine learning phase.

Welch's t-test used for pairwise comparisons where equal variance cannot be assumed.

---

### Results

#### Test 1: Opening Family Win Rate Differences (H₁)
| Metric | Value |
|--------|-------|
| Openings tested | 18 (≥5 games each) |
| Highest win rate | Scotch Game: 83.3% (6 games) |
| Lowest win rate | Modern Defense: 25.0% (8 games) |
| F-statistic | 0.6430 |
| p-value | 0.8563 |

**Interpretation:** There is no statistically significant difference in win rates across different opening families based on the current sample size. Fail to reject the null hypothesis.

#### Test 2: Time Forfeit Proportion (H₂)
| Metric | Value |
|--------|-------|
| Total games | 355 |
| Time forfeit games | 226 |
| Observed proportion | 63.7% |
| Null hypothesis | p₀ = 0.50 |
| z-statistic | 5.166 |
| p-value | < 0.0001 |
| 95% CI | [58.6%, 68.7%] |

**Interpretation:** The time forfeit rate of 63.7% is significantly higher than 50% (p < 0.0001). This is a very large effect — time management is the dominant loss factor, more than pure tactical errors. This strongly motivates the puzzle recommendation approach: faster tactical pattern recognition reduces time spent per move.

---

## Machine Learning Methods

To extend the statistical analysis with machine learning, the project now includes a separate notebook:

- `ML_chess.ipynb`

Because working with 5.5 million puzzles is computationally infeasible for this scope, this notebook operates on a sampled subset of 50 puzzles and uses simple, interpretable models.

### ML Task 1: Player Profiling & Puzzle Recommendation

Goal:
- Extract the player's weakness (average CPL) from the evaluated games.
- Estimate their tactical skill rating.
- Recommend the top 3 best matching puzzles using **K-Nearest Neighbors (KNN)**.

Results:
- **Estimated Rating:** 1450
- **Top 3 Recommended Puzzles:** Identified puzzles containing themes like 'middlegame advantage', 'fork', and 'endgame mate'.

Purpose:
- Connect data exploration with a practical application by creating a personalized puzzle recommender.

### ML Task 2: Regression

Goal:
- Predict `Rating` of a puzzle based on community engagement metrics (`NbPlays` and `Popularity`).

Methods:
- Linear Regression
- Random Forest Regressor

Evaluation:
- Train-test split (80/20)
- RMSE, MAE, R²

Results:
- **Random Forest** achieved the lowest RMSE (230.14) and highest R² (0.25), showing that popularity has weak signals about puzzle difficulty.

### ML Task 3: Classification

Goal:
- Classify a puzzle's difficulty as **Easy vs Hard** based on `NbPlays` and `Popularity`.

Methods:
- Logistic Regression
- Random Forest Classifier

Evaluation:
- Accuracy, Precision, Recall, and F1-score

Results:
- **Random Forest** correctly classified the puzzle difficulty with **70% Accuracy**, outperforming linear methods.

---

## Conclusion

The analysis confirms that my chess performance suffers primarily from **time management** (63.7% time forfeit rate, p < 0.0001) rather than from pure tactical inability. The one-sample proportion z-test confirmed this is a significant and dominating loss factor. Additionally, while openings show observational differences, ANOVA showed no statistically significant difference in win rates across families (p = 0.8563).

The full pipeline successfully bridges the gap between gameplay analytics and targeted improvement. Using Stockfish evaluation to extract blunder positions and then using K-Nearest Neighbors (KNN) algorithms allows us to build a recommendation engine that surfaces specific tactical patterns needed to reduce time spent per move.

---

## Project Structure

```text
├── DATA/
│   ├── raw/
│   │   ├── lichess_iamtheobama_2026-04-14.pgn
│   │   └── 50 Puzzles.xlsx
│   └── processed/
│       ├── move_evaluations.csv
│       ├── blunders.csv
│       └── puzzle_matches.csv
├── EDA.ipynb
├── Hypothesis_Testing.ipynb
├── ML_chess.ipynb
├── final_report_demir.md
├── final_report_demir.docx
├── requirements.txt
└── README.md
```

---

## Limitations

- The dataset covers only a single day's games (April 13–14, 2026), which may not represent long-term patterns.
- Bullet games (339/355) dominate the dataset; findings may not generalize to longer time controls.
- Stockfish evaluations at depth 15 may miss deep tactical sequences requiring greater depth.
- The high time forfeit rate (63.7%) may obscure the true tactical blunder rate, since many lost positions might have been recoverable.
- The Lichess puzzle database was restricted to a sampled 50-puzzle subset due to computational scale. Machine learning models (especially Random Forest) typically require much more data to avoid overfitting.

---

## Future Work

- Extend dataset to 1,000+ games across multiple time periods.
- Apply clustering (K-Means, DBSCAN) directly to blunder positions to discover tactical motif groups.
- Expand the machine learning section with richer, larger subsets from the Lichess puzzle DB.
- Track improvement over time as a before/after experiment with targeted training.

---

## How to Reproduce

1. Clone the repository:
   ```bash
   git clone https://github.com/demirhanisik/DSA210-TermProject.git
   cd DSA210-TermProject
   ```

2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

3. Install Stockfish engine:
   - Download from [https://stockfishchess.org/download/](https://stockfishchess.org/download/)
   - Update the path in notebooks: `Stockfish("/usr/local/bin/stockfish")`

4. Run the notebooks in order:
   - `EDA.ipynb` — Exploratory Data Analysis
   - `Hypothesis_Testing.ipynb` — Statistical hypothesis tests
   - `ML_chess.ipynb` — Machine learning and puzzle recommendation

---

## AI Usage Disclosure

This project used AI assistance during development and documentation.

The overall project idea, topic selection, research direction, and general project structure were developed by the student.

AI tools were used mainly for supportive tasks such as:

- improving and organizing code
- helping generate or refine some code cells
- helping generate or refine some plots and visual outputs
- revising written explanations in the notebooks, README, and final report
- checking consistency between repository files before submission

All AI-supported code, figures, and text were reviewed, edited, and approved by the student before inclusion in the final submission.

---

## Author

Demirhan Işık — 34464
