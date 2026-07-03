#include "evaluation.h"
#include <algorithm>
#include <cmath>

std::vector<Prediction> generatePredictions(const SvdModel& model,
                                             const std::vector<Rating>& testSet) {
    std::vector<Prediction> predictions;
    predictions.reserve(testSet.size());

    for (const auto& r : testSet) {
        double est = model.predict(r.userId, r.isbn);
        predictions.push_back({r.userId, r.isbn, r.rating, est});
    }
    return predictions;
}

double rmse(const std::vector<Prediction>& predictions) {
    if (predictions.empty()) return 0.0;
    double sumSq = 0.0;
    for (const auto& p : predictions) {
        double diff = p.trueRating - p.estRating;
        sumSq += diff * diff;
    }
    return std::sqrt(sumSq / predictions.size());
}

double mae(const std::vector<Prediction>& predictions) {
    if (predictions.empty()) return 0.0;
    double sumAbs = 0.0;
    for (const auto& p : predictions) {
        sumAbs += std::fabs(p.trueRating - p.estRating);
    }
    return sumAbs / predictions.size();
}

std::unordered_map<std::string, std::vector<std::pair<std::string, double>>>
getTopN(const std::vector<Prediction>& predictions, int n) {
    std::unordered_map<std::string, std::vector<std::pair<std::string, double>>> topN;

    for (const auto& p : predictions) {
        topN[p.userId].emplace_back(p.isbn, p.estRating);
    }

    for (auto& [uid, items] : topN) {
        std::sort(items.begin(), items.end(),
                  [](const auto& a, const auto& b) { return a.second > b.second; });
        if (static_cast<int>(items.size()) > n) {
            items.resize(n);
        }
    }
    return topN;
}

PrecisionRecall precisionRecallAtK(const std::vector<Prediction>& predictions,
                                    int k,
                                    double threshold) {
    struct UserData {
        std::vector<bool> relevant;                       // true rating >= threshold
        std::vector<std::pair<std::string, double>> preds; // (isbn, est)
    };
    std::unordered_map<std::string, UserData> perUser;

    for (const auto& p : predictions) {
        auto& ud = perUser[p.userId];
        ud.relevant.push_back(p.trueRating >= threshold);
        ud.preds.emplace_back(p.isbn, p.estRating);
    }

    std::vector<double> precisions, recalls;

    for (auto& [uid, ud] : perUser) {
        // Sort (isbn, est) together with the parallel "relevant" flags by
        // descending predicted rating, mirroring the notebook's approach
        // of sorting predicted scores and checking relevance by rank.
        std::vector<size_t> order(ud.preds.size());
        for (size_t i = 0; i < order.size(); ++i) order[i] = i;
        std::sort(order.begin(), order.end(), [&](size_t a, size_t b) {
            return ud.preds[a].second > ud.preds[b].second;
        });

        int topK = std::min<int>(k, static_cast<int>(order.size()));
        int relevantAndRecommended = 0;
        for (int i = 0; i < topK; ++i) {
            if (ud.relevant[order[i]]) relevantAndRecommended++;
        }

        int relevantTotal = 0;
        for (bool r : ud.relevant) if (r) relevantTotal++;

        if (relevantTotal > 0) {
            precisions.push_back(static_cast<double>(relevantAndRecommended) / k);
            recalls.push_back(static_cast<double>(relevantAndRecommended) / relevantTotal);
        }
    }

    PrecisionRecall result;
    if (!precisions.empty()) {
        double sumP = 0.0, sumR = 0.0;
        for (double v : precisions) sumP += v;
        for (double v : recalls) sumR += v;
        result.precision = sumP / precisions.size();
        result.recall = sumR / recalls.size();
    }
    return result;
}
