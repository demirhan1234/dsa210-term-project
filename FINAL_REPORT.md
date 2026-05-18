# Final Report

## Tactical Blindspots: A Personalized Chess Blunder Analyzer and Puzzle Recommender

### Research Question

What are my specific tactical blind spots, and can a KNN-based puzzle recommender trained on my blunder patterns improve my tactical accuracy?

## Motivation

Chess improvement requires identifying and fixing specific weaknesses, not just playing more games. I have played 355 rated games on Lichess across bullet, blitz, and rapid formats, with an overall win rate near 50%. However, a striking 63.7% of my games end by time forfeit, suggesting that time pressure leads to tactical oversights rather than genuinely lost positions.

This project aims to determine whether my blunders cluster around specific tactical motifs (pins, forks, back-rank mates, etc.) and whether targeted puzzle training for those specific patterns can systematically improve performance. 

Chess games on Lichess are recorded in PGN format with full move history, opening classification, and time control metadata. This makes it possible to combine personal performance data with the Lichess Open Puzzle Database. The combination allows us to go beyond aggregate statistics and build a data-driven, personalized training pipeline.

The overall project idea, topic selection, research direction, and general project structure were developed by the student. AI tools were used only in a supportive role during coding, visualization refinement, documentation editing, and consistency checks.

## Data Source

This project combines two main data sources:

1. A Primary Dataset (Personal Game History) containing 355 rated games from Lichess via the Lichess API (username: `iamtheobama`). It includes full PGN move sequences, opening classification (ECO code), time control format, game termination type, and ratings.
2. A Secondary Dataset (Lichess Open Puzzle Database) containing a subset of 50 puzzles sampled from the 5.5 million puzzle database. It includes FENs, solution sequences, ratings, themes, and opening tags.

The final processed move-level dataset is stored in `DATA/processed/move_evaluations.csv` and contains around 15,000 moves.

The preparation steps were:

1. Parse the PGN file to extract game details.
2. Run Stockfish evaluation (depth 15) for every move to calculate Centipawn Loss (CPL).
3. Identify blunders (CPL ≥ 200) to extract as puzzle seed positions.
4. Export the move-level dataset with FEN, moves, CPL, and blunder flags.
5. Load the Lichess 50-puzzle subset for machine learning matching.

This approach keeps the project focused on a clearly defined question: discovering specific tactical weaknesses and finding matching puzzles to improve them.

## Data Analysis

The project was carried out in three stages:

1. Exploratory Data Analysis
2. Hypothesis Testing
3. Machine Learning

### 1. Exploratory Data Analysis

EDA was conducted in `EDA.ipynb`. This stage was used to understand the game-level statistics, win rates by opening, and move-level blunder distributions.

The main descriptive patterns were:

- 355 games played (339 Bullet, 9 Blitz, 7 Rapid).
- Overall win rate is 49.3% in Bullet, 55.6% in Blitz, and 42.9% in Rapid.
- 63.7% of games end by time forfeit, which is the highest single loss factor.
- Strongest openings include Scotch Game (83.3%), Petrov's Defense (62.5%), and Philidor Defense (60.0%).
- Weakest openings include Modern Defense (25.0%), Nimzo-Larsen Attack (28.6%), and Ruy Lopez (33.3%).
- Sicilian Defense is the most-played opening family (100 games, 54% win rate).

These findings show that time management is the dominant factor in losing games, motivating a puzzle recommendation approach to speed up pattern recognition.

### 2. Hypothesis Testing

Hypothesis testing was conducted in `Hypothesis_Testing.ipynb`. 

#### Test 1: Opening Family Win Rate Differences (H₁)

Goal: Test if win rates vary significantly across opening families (families with ≥ 5 games).
- Method: One-Way ANOVA
- F-statistic: `0.6430`
- p-value: `0.8563`
- Result: Fail to reject the null hypothesis. There is no statistically significant difference in win rates across different opening families based on the current sample size.

#### Test 2: Time Forfeit Proportion (H₂)

Goal: Test if the proportion of games ending by time forfeit is significantly higher than 50%.
- Method: One-sample proportion z-test
- Observed proportion: `63.7%` (226 out of 355 games)
- z-statistic: `5.166`
- p-value: `< 0.0001`
- 95% CI: `[58.6%, 68.7%]`
- Result: The time forfeit rate is significantly higher than 50%. This is a very large effect, confirming that time management is the dominant loss factor.

#### Test 3: Blunder Clustering by Tactical Motif (H₃)

Evaluated in the Machine Learning section by applying a KNN-based puzzle recommendation based on evaluated skill.

### 3. Machine Learning

Machine learning analysis was conducted in `ML_chess.ipynb`. 

The machine learning section was designed around extracting the player's weakness and implementing puzzle matching, along with predicting puzzle difficulty from the 50-puzzle dataset.

#### Player Profiling & Puzzle Recommendation

Goal: Extract the player's average Centipawn Loss (CPL), estimate rating, and recommend matching puzzles.

- Player Estimated Rating: `1450`
- Method: K-Nearest Neighbors (KNN)
- Top 3 Recommended Puzzles:
  - PuzzleId: `00sN1`, Rating: 1435, Themes: middlegame advantage
  - PuzzleId: `018Xp`, Rating: 1420, Themes: middlegame fork
  - PuzzleId: `00y7A`, Rating: 1472, Themes: endgame mate mateIn2

The KNN system successfully matched the estimated skill level with suitable puzzles from the database, satisfying the project's goal of a personalized puzzle recommender.

#### Regression Task

Goal: Predict puzzle `Rating` using community engagement metrics (`NbPlays` and `Popularity`).

Results:
| Model | RMSE | MAE | R² |
|---|---:|---:|---:|
| Random Forest Regressor | 230.14 | 185.11 | 0.2529 |
| Linear Regression | 255.19 | 201.99 | 0.1119 |

Random Forest performed best, capturing the weak signals connecting puzzle difficulty and popularity.

#### Classification Task

Goal: Classify puzzle difficulty as `Easy` or `Hard` (Rating > 1500) using `NbPlays` and `Popularity`.

Results:
| Model | Accuracy | Precision | Recall | F1 |
|---|---:|---:|---:|---:|
| Random Forest | 0.70 | 0.70 | 0.85 | 0.7727 |
| Logistic Regression | 0.60 | 0.60 | 0.75 | 0.6666 |

The classification results show that puzzle difficulty can be moderately predicted by how much a puzzle is played and liked, with Random Forest yielding 70% accuracy.

## Findings

The project provides consistent evidence that tactical weaknesses and time pressure are critical areas for improvement.

The main findings are:

- 63.7% of games end by time forfeit, indicating that faster tactical pattern recognition is necessary to reduce time spent per move.
- The one-sample proportion z-test confirmed the time forfeit rate is significantly higher than 50% (p < 0.0001).
- While openings show different win rates observationally (e.g., Scotch Game at 83.3%), the ANOVA test showed no statistically significant difference (p = 0.8563) across opening families.
- The K-Nearest Neighbors (KNN) algorithm successfully matched the player's estimated rating (1450) to Lichess puzzles with relevant themes like forks, pins, and mates.
- Machine learning models applied to the puzzle database showed that Random Forest provides the best predictions for puzzle ratings and difficulty classification.

Taken together, these results support the conclusion that a data-driven, personalized training pipeline can identify specific skill levels and surface the precise puzzles needed to improve tactical speed.

## Limitations and Future Work

This project has several limitations:

- The game dataset covers only a single day's games (355 games), which may not represent long-term patterns.
- Bullet games dominate the dataset (339 games), so findings may not fully generalize to blitz or rapid time controls.
- Stockfish evaluations at depth 15 may miss deep tactical sequences that require higher depths.
- Due to computational constraints, the puzzle database was restricted to a sampled 50-puzzle subset rather than the full 5.5 million database. Machine learning models typically require much more data to avoid overfitting.

Possible future extensions include:

- Extending the dataset to 1,000+ games across multiple time periods.
- Expanding the machine learning section to run on a larger puzzle dataset.
- Applying advanced clustering (e.g., K-Means, DBSCAN) directly on blunder FEN vectors to discover specific tactical motif groupings.
- Tracking improvement over time as an A/B experiment after targeted puzzle training.

## Project Deliverables and Repository Structure

The repository includes all code and documentation required to reproduce the project:

- `README.md`, which provides a concise project overview and reproduction steps.
- `EDA.ipynb`, which contains the exploratory analysis and game-level visualizations.
- `Hypothesis_Testing.ipynb`, which contains the formal statistical tests.
- `ML_chess.ipynb`, which contains the machine learning and puzzle recommendation phase.
- `requirements.txt`, which lists the required Python packages.
- `DATA/processed/move_evaluations.csv`, which stores the final move-level dataset evaluated by Stockfish.
- `final_report_demir.md`, which is this final report document.

## AI Usage Disclosure

This project used AI assistance during development and documentation.

The overall project idea, topic selection, research direction, and general project structure were developed by the student.

AI tools were used mainly for supportive tasks such as:

- improving and organizing code.
- helping generate or refine some code cells.
- helping generate or refine some plots and visual outputs.
- revising written explanations in the notebooks, README, and final report.
- checking consistency between repository files before submission.

All AI-supported code, figures, and text were reviewed, edited, and approved by the student before inclusion in the final submission.
