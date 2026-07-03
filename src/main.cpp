// Book Recommendation System (C++ port)
//
// Re-implements the original Python/Surprise prototype:
//   1. Load and clean Ratings.csv
//   2. Sample + train/test split
//   3. Train an SVD (biased matrix factorization) model via SGD
//   4. Evaluate with RMSE and Precision@K / Recall@K
//   5. Serve Top-N recommendations for a user, resolved to book details
//
#include <algorithm>
#include <iostream>
#include <random>
#include <string>
#include <unordered_set>
#include <vector>

#include "book_catalog.h"
#include "csv_reader.h"
#include "evaluation.h"
#include "svd_model.h"

namespace {

std::vector<Rating> loadRatings(const std::string& path) {
    std::unordered_map<std::string, size_t> headerIndex;
    std::vector<std::vector<std::string>> rows;
    CsvReader::readWithHeader(path, ';', headerIndex, rows);

    auto need = [&](const char* name) -> size_t {
        auto it = headerIndex.find(name);
        if (it == headerIndex.end()) {
            throw std::runtime_error(std::string("Ratings.csv missing column: ") + name);
        }
        return it->second;
    };

    size_t userCol = need("User-ID");
    size_t isbnCol = need("ISBN");
    size_t ratingCol = need("Rating");

    std::vector<Rating> ratings;
    ratings.reserve(rows.size());

    for (const auto& row : rows) {
        if (row.size() <= userCol || row.size() <= isbnCol || row.size() <= ratingCol) continue;
        try {
            double r = std::stod(row[ratingCol]);
            if (r <= 0) continue;  // drop unrated / implicit entries, same as the notebook
            ratings.push_back({row[userCol], row[isbnCol], r});
        } catch (...) {
            continue;  // skip malformed rows
        }
    }
    return ratings;
}

std::vector<Rating> sampleRatings(const std::vector<Rating>& ratings, size_t n, unsigned int seed) {
    if (ratings.size() <= n) return ratings;

    std::vector<size_t> idx(ratings.size());
    for (size_t i = 0; i < idx.size(); ++i) idx[i] = i;

    std::mt19937 rng(seed);
    std::shuffle(idx.begin(), idx.end(), rng);
    idx.resize(n);

    std::vector<Rating> sample;
    sample.reserve(n);
    for (size_t i : idx) sample.push_back(ratings[i]);
    return sample;
}

void trainTestSplit(const std::vector<Rating>& data,
                     double testSize,
                     unsigned int seed,
                     std::vector<Rating>& trainSet,
                     std::vector<Rating>& testSet) {
    std::vector<size_t> idx(data.size());
    for (size_t i = 0; i < idx.size(); ++i) idx[i] = i;

    std::mt19937 rng(seed);
    std::shuffle(idx.begin(), idx.end(), rng);

    size_t nTest = static_cast<size_t>(data.size() * testSize);
    testSet.clear();
    trainSet.clear();
    testSet.reserve(nTest);
    trainSet.reserve(data.size() - nTest);

    for (size_t i = 0; i < idx.size(); ++i) {
        if (i < nTest) testSet.push_back(data[idx[i]]);
        else trainSet.push_back(data[idx[i]]);
    }
}

void printTopNForUser(const std::string& targetUser,
                       const std::vector<Rating>& sampledRatings,
                       const SvdModel& model,
                       const BookCatalog& catalog,
                       int n) {
    std::unordered_set<std::string> ratedBooks;
    std::unordered_set<std::string> allBooks;
    for (const auto& r : sampledRatings) {
        allBooks.insert(r.isbn);
        if (r.userId == targetUser) ratedBooks.insert(r.isbn);
    }

    std::vector<std::pair<std::string, double>> userPredictions;
    userPredictions.reserve(allBooks.size());
    for (const auto& isbn : allBooks) {
        if (ratedBooks.count(isbn)) continue;
        userPredictions.emplace_back(isbn, model.predict(targetUser, isbn));
    }

    std::sort(userPredictions.begin(), userPredictions.end(),
              [](const auto& a, const auto& b) { return a.second > b.second; });
    if (static_cast<int>(userPredictions.size()) > n) userPredictions.resize(n);

    std::cout << "\nTop " << n << " book recommendations for user " << targetUser << ":\n";
    for (const auto& [isbn, est] : userPredictions) {
        const BookInfo* info = catalog.lookup(isbn);
        std::cout.precision(3);
        if (info) {
            std::cout << "  " << info->title << " by " << info->author
                       << " (" << info->year << ", " << info->publisher << ")"
                       << " | Predicted Rating: " << est << "\n";
        } else {
            std::cout << "  ISBN: " << isbn << " | Predicted Rating: " << est << "\n";
        }
    }
}

}  // namespace

int main(int argc, char** argv) {
    std::string ratingsPath = "data/Ratings.csv";
    std::string booksPath = "data/Books.csv";
    if (argc >= 2) ratingsPath = argv[1];
    if (argc >= 3) booksPath = argv[2];

    const size_t SAMPLE_SIZE = 10000;
    const double TEST_SIZE = 0.2;
    const unsigned int SEED = 42;
    const int TOP_N = 5;
    const double RELEVANCE_THRESHOLD = 4.0;

    try {
        std::cout << "Loading ratings from " << ratingsPath << " ...\n";
        std::vector<Rating> ratings = loadRatings(ratingsPath);
        std::cout << "Loaded " << ratings.size() << " rated (rating > 0) entries.\n";

        std::vector<Rating> sampled = sampleRatings(ratings, SAMPLE_SIZE, SEED);
        std::cout << "Sampled " << sampled.size() << " ratings for training.\n";

        std::vector<Rating> trainSet, testSet;
        trainTestSplit(sampled, TEST_SIZE, SEED, trainSet, testSet);
        std::cout << "Train: " << trainSet.size() << " | Test: " << testSet.size() << "\n";

        std::cout << "Training SVD model...\n";
        SvdModel model;
        model.fit(trainSet);

        std::vector<Prediction> predictions = generatePredictions(model, testSet);

        double rmseValue = rmse(predictions);
        double maeValue = mae(predictions);
        std::cout.precision(4);
        std::cout << "RMSE: " << rmseValue << "\n";
        std::cout << "MAE:  " << maeValue << "\n";

        PrecisionRecall pr = precisionRecallAtK(predictions, TOP_N, RELEVANCE_THRESHOLD);
        std::cout << "Precision@" << TOP_N << ": " << pr.precision << "\n";
        std::cout << "Recall@" << TOP_N << ":    " << pr.recall << "\n";

        std::cout << "\nLoading book details from " << booksPath << " ...\n";
        BookCatalog catalog;
        catalog.load(booksPath);
        std::cout << "Loaded " << catalog.size() << " unique books.\n";

        // Recommend for the first user in the sample, then let the user
        // look up any other User-ID interactively.
        std::string firstUser = sampled.front().userId;
        printTopNForUser(firstUser, sampled, model, catalog, TOP_N);

        std::cout << "\nEnter a User-ID to get recommendations (blank to quit): ";
        std::string targetUser;
        while (std::getline(std::cin, targetUser)) {
            if (targetUser.empty()) break;

            bool userExists = std::any_of(sampled.begin(), sampled.end(),
                                           [&](const Rating& r) { return r.userId == targetUser; });
            if (!userExists) {
                std::cout << "User-ID " << targetUser << " not found in the sampled dataset.\n";
            } else {
                printTopNForUser(targetUser, sampled, model, catalog, TOP_N);
            }
            std::cout << "\nEnter a User-ID to get recommendations (blank to quit): ";
        }

    } catch (const std::exception& e) {
        std::cerr << "Error: " << e.what() << "\n";
        std::cerr << "\nMake sure Ratings.csv and Books.csv (the Book-Crossing dataset) "
                     "are placed in the data/ directory, or pass their paths as arguments:\n"
                     "  ./book_recommender path/to/Ratings.csv path/to/Books.csv\n";
        return 1;
    }

    return 0;
}
