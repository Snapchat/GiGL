#pragma once

#include <torch/torch.h>

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <vector>

namespace gigl {

constexpr int32_t kTypedPPRBestScoreIndex = 0;
constexpr int32_t kTypedPPRGlobalHopProximityIndex = 1;
constexpr int32_t kTypedPPRNumGlobalFeatures = 2;     // best score + global hop proximity
constexpr int32_t kTypedPPRNumPerChannelFeatures = 3; // score + hop proximity + presence

// Owns the packed edge_attr layout used while merging typed PPR channels.
//
// Typed extraction first accumulates one dense feature vector per selected node,
// then copies those vectors into the final tensor. Keeping the offsets here
// keeps addTypedPPRSeedFeaturesAndCandidates focused on merge policy instead of
// repeating manual column math.
//
// The width is 2 + (3 * numChannels):
//   - 2 global columns:
//       [best_score, hop_proximity]
//     These preserve the regular PPR edge_attr contract at the front of the row.
//   - 3 columns per typed channel:
//       [(channel_score, channel_hop_proximity, channel_presence), ...]
//     The per-channel score supports channel attribution/ranking, the per-channel
//     hop proximity gives models a bounded closeness signal, and the presence bit
//     is the explicit channel-reachability mask.
//
// Hop proximity is 1 / (1 + hop) when a channel reaches the node:
// anchor=1.0, 1-hop=0.5, 2-hop ~= 0.333, and so on. Missing channels remain 0,
// which is finite, bounded, and never looks closer than a reached channel. For
// present channels, callers can recover the original hop count as
// (1 - proximity) / proximity; use the presence bit before applying this
// inverse because proximity 0 means the channel is missing.
class TypedPPRFeatureLayout {
public:
    explicit TypedPPRFeatureLayout(int32_t numChannels) : _numChannels(numChannels) {
        TORCH_CHECK(numChannels > 0, "Typed PPR feature layout requires at least one channel.");
    }

    [[nodiscard]] int32_t numFeatures() const {
        return kTypedPPRNumGlobalFeatures + (kTypedPPRNumPerChannelFeatures * _numChannels);
    }

    // New node feature vectors start empty. Scores, proximities, and presence
    // bits all use 0 as their absent-channel value.
    [[nodiscard]] std::vector<double> makeFeatureVector() const {
        return std::vector<double>(numFeatures(), 0.0);
    }

    // Merge one channel's emitted score/proximity into an existing node feature row.
    //
    // A node can be seen in multiple typed channels. The scalar score column keeps
    // the best emitted PPR score for downstream consumers that expect one weight,
    // while the global proximity keeps the closest discovered distance.
    void updateScores(std::vector<double>& features, int32_t channelIndex, double score, double hopProximity) const {
        validateFeatureVector(features);
        validateChannelIndex(channelIndex);

        // Repeated observations keep the strongest score and closest proximity.
        // Current extraction emits at most one row per node per channel, but the
        // monotonic merge keeps this helper safe if that changes.
        features[kTypedPPRBestScoreIndex] = std::max(features[kTypedPPRBestScoreIndex], score);
        features[kTypedPPRGlobalHopProximityIndex] = std::max(features[kTypedPPRGlobalHopProximityIndex], hopProximity);
        features[channelScoreIndex(channelIndex)] = std::max(features[channelScoreIndex(channelIndex)], score);
        features[channelHopProximityIndex(channelIndex)] =
            std::max(features[channelHopProximityIndex(channelIndex)], hopProximity);
        features[channelPresenceIndex(channelIndex)] = 1.0;
    }

private:
    [[nodiscard]] int32_t channelBaseIndex(int32_t channelIndex) const {
        return kTypedPPRNumGlobalFeatures + (kTypedPPRNumPerChannelFeatures * channelIndex);
    }
    [[nodiscard]] int32_t channelScoreIndex(int32_t channelIndex) const {
        return channelBaseIndex(channelIndex);
    }
    [[nodiscard]] int32_t channelHopProximityIndex(int32_t channelIndex) const {
        return channelBaseIndex(channelIndex) + 1;
    }
    [[nodiscard]] int32_t channelPresenceIndex(int32_t channelIndex) const {
        return channelBaseIndex(channelIndex) + 2;
    }

    void validateFeatureVector(const std::vector<double>& features) const {
        TORCH_CHECK(features.size() == static_cast<std::size_t>(numFeatures()),
                    "Typed PPR feature row width must be ",
                    numFeatures(),
                    ", got ",
                    features.size(),
                    ".");
    }

    void validateChannelIndex(int32_t channelIndex) const {
        TORCH_CHECK(channelIndex >= 0 && channelIndex < _numChannels,
                    "Typed PPR channel index ",
                    channelIndex,
                    " out of range [0, ",
                    _numChannels,
                    ").");
    }

    // The channel count is the only stored layout state. Every offset is derived
    // from it, so feature width cannot drift from the per-channel triples.
    const int32_t _numChannels;
};

} // namespace gigl
