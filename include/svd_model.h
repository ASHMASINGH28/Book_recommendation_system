#pragma once
#include <string>
#include <vector>
#include <unordered_map>

// A single (user, item, rating) observation.
struct Rating {
    std::string userId;
    std::string isbn;
    double rating;
};

// Hyperparameters mirror scikit-surprise's default SVD() configuration,
// which is what the original Python prototype used.
struct SvdParams {
    int   nFactors   = 100;
    int   nEpochs    = 20;
    double lrAll     = 0.005;   // learning rate
    double regAll    = 0.02;    // L2 regularization
    double ratingMin = 1.0;
    double ratingMax = 10.0;
    unsigned int seed = 42;
};

// Funk-SVD / biased matrix factorization trained with stochastic gradient
// descent:
//     r_hat(u, i) = mu + b_u + b_i + q_i . p_u
// This is the same model family scikit-surprise's SVD() implements.
class SvdModel {
public:
    explicit SvdModel(SvdParams params = SvdParams());

    // Fit the model on a training set. Builds internal user/item indices.
    void fit(const std::vector<Rating>& trainSet);

    // Predict a rating for (userId, isbn). Falls back to the global mean
    // (baseline) for users/items not seen during training, mirroring
    // scikit-surprise's behaviour for unknown users/items.
    double predict(const std::string& userId, const std::string& isbn) const;

    bool knownUser(const std::string& userId) const;
    bool knownItem(const std::string& isbn) const;

    const std::vector<std::string>& itemIds() const { return itemIds_; }

private:
    SvdParams params_;
    double globalMean_ = 0.0;

    std::unordered_map<std::string, int> userIndex_;
    std::unordered_map<std::string, int> itemIndex_;
    std::vector<std::string> itemIds_;

    std::vector<double> bu_;                 // user biases
    std::vector<double> bi_;                 // item biases
    std::vector<std::vector<double>> pu_;    // user factors
    std::vector<std::vector<double>> qi_;    // item factors

    double clip(double value) const;
};
