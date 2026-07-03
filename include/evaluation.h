#pragma once
#include <string>
#include <vector>
#include <unordered_map>
#include "svd_model.h"

// One (user, item, trueRating, estimatedRating) test-set prediction.
struct Prediction {
    std::string userId;
    std::string isbn;
    double trueRating;
    double estRating;
};

// Run the model over every (user, item, rating) in testSet and collect
// the resulting predictions.
std::vector<Prediction> generatePredictions(const SvdModel& model,
                                             const std::vector<Rating>& testSet);

// Root Mean Squared Error over a set of predictions.
double rmse(const std::vector<Prediction>& predictions);

// Mean Absolute Error, provided alongside RMSE for completeness.
double mae(const std::vector<Prediction>& predictions);

// For each user, the top-N (isbn, estRating) pairs sorted by estRating desc.
std::unordered_map<std::string, std::vector<std::pair<std::string, double>>>
getTopN(const std::vector<Prediction>& predictions, int n = 5);

// Precision@K and Recall@K, averaged over users that have at least one
// "relevant" item (trueRating >= threshold) in the test set.
struct PrecisionRecall {
    double precision = 0.0;
    double recall = 0.0;
};

PrecisionRecall precisionRecallAtK(const std::vector<Prediction>& predictions,
                                    int k = 5,
                                    double threshold = 4.0);
