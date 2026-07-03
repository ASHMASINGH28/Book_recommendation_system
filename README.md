# Book Recommendation System (C++)

A C++17 port of the original Python/Surprise prototype: a personalised book
recommender that predicts user ratings with **SVD-based collaborative
filtering** and serves **Top-N recommendations**.

## How it maps to the original project

| Python (scikit-surprise) prototype        | C++ port                                   |
|---------------------------------------------|---------------------------------------------|
| `pandas` for loading/cleaning CSVs          | `CsvReader` (`src/csv_reader.cpp`)          |
| `Reader` + `Dataset.load_from_df`           | `Rating` structs + user/item indices        |
| `surprise.model_selection.train_test_split` | `trainTestSplit()` in `main.cpp`            |
| `SVD()` model, SGD-trained                  | `SvdModel` (`src/svd_model.cpp`)            |
| `accuracy.rmse`                             | `rmse()` in `evaluation.cpp`                |
| `get_top_n`, `precision_recall_at_k`        | `getTopN()`, `precisionRecallAtK()`         |
| ISBN → title/author lookup via `books` df   | `BookCatalog` (`src/book_catalog.cpp`)      |

## The algorithm

The model is a **biased matrix factorization** (the same family scikit-surprise's
`SVD()` implements), trained with stochastic gradient descent:

```
r_hat(u, i) = mu + b_u + b_i + <p_u, q_i>
```

- `mu`   — global mean rating
- `b_u`  — per-user bias
- `b_i`  — per-item (book) bias
- `p_u`, `q_i` — learned `nFactors`-dimensional latent vectors for the user and item

Each SGD step nudges `b_u`, `b_i`, `p_u`, `q_i` to reduce the prediction error
`e = r - r_hat`, with L2 regularization to prevent overfitting. Default
hyperparameters (`include/svd_model.h`) mirror `surprise.SVD()`'s defaults:
100 factors, 20 epochs, learning rate 0.005, regularization 0.02.

## Project layout

```
book_recommender_cpp/
├── include/            # public headers
│   ├── book_catalog.h
│   ├── csv_reader.h
│   ├── evaluation.h
│   └── svd_model.h
├── src/                # implementation + main driver
│   ├── book_catalog.cpp
│   ├── csv_reader.cpp
│   ├── evaluation.cpp
│   ├── main.cpp
│   └── svd_model.cpp
├── data/               # put Ratings.csv / Books.csv here
├── Makefile
└── README.md
```

## Build

Requires a C++17 compiler (g++/clang++). No external dependencies.

```bash
make            # builds ./book_recommender
```

## Get the data

This uses the [Book-Crossing dataset](https://www.kaggle.com/datasets/arashnic/book-recommendation-dataset)
(same dataset as the original notebook). Download `Ratings.csv` and
`Books.csv` and place them in `data/`.

## Run

```bash
./book_recommender                                   # expects data/Ratings.csv, data/Books.csv
./book_recommender path/to/Ratings.csv path/to/Books.csv   # explicit paths
```

The program will:

1. Load and clean the ratings (drop rating <= 0)
2. Sample 10,000 ratings and do an 80/20 train/test split (seeded, reproducible)
3. Train the SVD model
4. Print **RMSE**, **MAE**, **Precision@5**, and **Recall@5** on the held-out test set
5. Print Top-5 recommendations (title/author/year/publisher) for a sample user
6. Prompt you to enter any `User-ID` from the sample to get its own Top-5 recommendations
   (blank line to quit)

## Notes / possible extensions

- `nFactors`, `nEpochs`, `lrAll`, `regAll` are all adjustable in `SvdParams`
  (`include/svd_model.h`) if you want to tune accuracy vs. training time.
- The current model falls back to the global mean for unseen users/items
  (a simple, standard cold-start baseline) — swapping in a popularity-based
  fallback would be a natural next step.
- `data/` is currently untouched by version control other than a placeholder
  README — add your own `.gitignore` entry if you don't want to commit the
  raw dataset.
