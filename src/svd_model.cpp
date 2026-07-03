#include "svd_model.h"
#include <algorithm>
#include <numeric>
#include <random>
#include <cmath>

SvdModel::SvdModel(SvdParams params) : params_(params) {}

bool SvdModel::knownUser(const std::string& userId) const {
    return userIndex_.find(userId) != userIndex_.end();
}

bool SvdModel::knownItem(const std::string& isbn) const {
    return itemIndex_.find(isbn) != itemIndex_.end();
}

double SvdModel::clip(double value) const {
    return std::min(params_.ratingMax, std::max(params_.ratingMin, value));
}

void SvdModel::fit(const std::vector<Rating>& trainSet) {
    // --- Build user/item indices -------------------------------------
    userIndex_.clear();
    itemIndex_.clear();
    itemIds_.clear();

    for (const auto& r : trainSet) {
        if (userIndex_.find(r.userId) == userIndex_.end()) {
            userIndex_[r.userId] = static_cast<int>(userIndex_.size());
        }
        if (itemIndex_.find(r.isbn) == itemIndex_.end()) {
            itemIndex_[r.isbn] = static_cast<int>(itemIndex_.size());
            itemIds_.push_back(r.isbn);
        }
    }

    const size_t nUsers = userIndex_.size();
    const size_t nItems = itemIndex_.size();

    // --- Global mean ----------------------------------------------------
    double sum = 0.0;
    for (const auto& r : trainSet) sum += r.rating;
    globalMean_ = trainSet.empty() ? 0.0 : sum / trainSet.size();

    // --- Initialize parameters ------------------------------------------
    std::mt19937 rng(params_.seed);
    std::normal_distribution<double> factorInit(0.0, 0.1);

    bu_.assign(nUsers, 0.0);
    bi_.assign(nItems, 0.0);
    pu_.assign(nUsers, std::vector<double>(params_.nFactors));
    qi_.assign(nItems, std::vector<double>(params_.nFactors));

    for (size_t u = 0; u < nUsers; ++u)
        for (int f = 0; f < params_.nFactors; ++f)
            pu_[u][f] = factorInit(rng);

    for (size_t i = 0; i < nItems; ++i)
        for (int f = 0; f < params_.nFactors; ++f)
            qi_[i][f] = factorInit(rng);

    // --- SGD training loop ------------------------------------------------
    std::vector<int> order(trainSet.size());
    std::iota(order.begin(), order.end(), 0);

    const double lr = params_.lrAll;
    const double reg = params_.regAll;

    for (int epoch = 0; epoch < params_.nEpochs; ++epoch) {
        std::shuffle(order.begin(), order.end(), rng);

        for (int idx : order) {
            const Rating& r = trainSet[idx];
            int u = userIndex_[r.userId];
            int i = itemIndex_[r.isbn];

            // Prediction and error for this observation.
            double dot = 0.0;
            for (int f = 0; f < params_.nFactors; ++f) dot += qi_[i][f] * pu_[u][f];
            double pred = globalMean_ + bu_[u] + bi_[i] + dot;
            double err = r.rating - pred;

            // Bias updates.
            bu_[u] += lr * (err - reg * bu_[u]);
            bi_[i] += lr * (err - reg * bi_[i]);

            // Factor updates (classic Funk-SVD SGD step).
            for (int f = 0; f < params_.nFactors; ++f) {
                double puf = pu_[u][f];
                double qif = qi_[i][f];
                pu_[u][f] += lr * (err * qif - reg * puf);
                qi_[i][f] += lr * (err * puf - reg * qif);
            }
        }
    }
}

double SvdModel::predict(const std::string& userId, const std::string& isbn) const {
    auto uit = userIndex_.find(userId);
    auto iit = itemIndex_.find(isbn);

    bool userKnown = uit != userIndex_.end();
    bool itemKnown = iit != itemIndex_.end();

    if (!userKnown && !itemKnown) {
        return clip(globalMean_);
    }

    double bu = userKnown ? bu_[uit->second] : 0.0;
    double bi = itemKnown ? bi_[iit->second] : 0.0;

    double dot = 0.0;
    if (userKnown && itemKnown) {
        const auto& p = pu_[uit->second];
        const auto& q = qi_[iit->second];
        for (int f = 0; f < params_.nFactors; ++f) dot += p[f] * q[f];
    }

    return clip(globalMean_ + bu + bi + dot);
}
