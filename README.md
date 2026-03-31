# dsa210-term-project
Student Name: Demirhan Isik
Student ID: 34464
Course: DSA 210 Introduction to Data Science

Project Proposal
Project Title: Tactical Blindspots: A Personalized Chess Blunder Analyzer and Puzzle Recommender

Motivation
This project addresses a practical challenge in personal chess improvement by applying data science methodologies. The goal is to analyze my personal chess game history to identify specific tactical mistakes, transform these errors into targeted training puzzles, and develop a recommendation engine to suggest mathematically similar puzzles. By doing so, I aim to systematically correct my tactical blind spots and improve my overall game performance.

Data Sources and Collection
I will work with two datasets. The primary dataset consists of my personal chess match history, which I will collect using the official Lichess and Chess.com APIs to extract PGN files of my games. This dataset will contain approximately 1,000 matches, including features such as board states (FEN), Stockfish engine evaluations, centipawn loss metrics, and time controls. To enrich this personal data, I will integrate the publicly available Lichess Open Puzzle Database, which contains over 5.5 million categorized chess puzzles. This public dataset will provide tactical tags and FEN structures to augment my personal blunder data.

Data Analysis and Planned Methods
I will evaluate my personal games using the Stockfish engine in Python to isolate moves with significant centipawn loss. These mistakes will be converted into FEN representations, which will then be matched with similar puzzles from the Lichess Open Puzzle Database using similarity algorithms such as K-Nearest Neighbors. Additionally, exploratory data analysis (EDA) will be conducted to visualize patterns in my errors, including distributions of centipawn loss, common tactical motifs, and positional contexts. This analysis will guide the selection of relevant puzzles for targeted training.
