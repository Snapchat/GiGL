// Microbenchmark for the PPR Forward Push kernel.
//
// Drives PPRForwardPush through the same call sequence the Python sampler uses
// (construct, then drainQueue / neighbor fetch / pushResiduals until the queue
// is empty, then extractTopKWithResidualTopUp) against a synthetic power-law
// graph held in memory. The neighbor fetch is served from a local CSR instead
// of RPC and is timed separately, so the reported drain, push, and extract
// times are kernel time only.
//
// Usage:
//   ppr_forward_push_benchmark [--seeds N] [--batches N] [--nodes N] ...
//   Run with --help for every flag and its default.
//
// --dump-output PATH writes every seed's extracted (node, score, hop proximity)
// rows. --compare-output PATH reads a file written by --dump-output and reports
// how far this build's output differs from it, which checks that a kernel
// change keeps results the same.

#include "sampling/ppr_forward_push.h"

#include <torch/torch.h>

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <fstream>
#include <iostream>
#include <map>
#include <random>
#include <string>
#include <vector>

namespace {

struct BenchmarkOptions {
    int64_t numNodes = 2'000'000;
    double meanDegree = 60.0;
    double degreeExponent = 2.1;
    int64_t maxDegree = 20'000;
    int32_t maxNeighborsPerFetch = 1000;
    int32_t numSeeds = 1500;
    int32_t numBatches = 10;
    int32_t warmupBatches = 1;
    double alpha = 0.5;
    double eps = 1e-4;
    int32_t maxPPRNodes = 50;
    bool enableResidualTopUp = true;
    uint64_t randomSeed = 42;
    std::string dumpOutputPath;
    std::string compareOutputPath;
};

void printUsage() {
    BenchmarkOptions defaults;
    std::cout << "Flags (defaults in parentheses):\n"
              << "  --nodes N                 graph node count (" << defaults.numNodes << ")\n"
              << "  --mean-degree X           target mean out-degree (" << defaults.meanDegree << ")\n"
              << "  --degree-exponent X       power-law exponent of the degree distribution ("
              << defaults.degreeExponent << ")\n"
              << "  --max-degree N            degree cap of the generated graph (" << defaults.maxDegree << ")\n"
              << "  --max-neighbors N         neighbors returned per fetched node (" << defaults.maxNeighborsPerFetch
              << ")\n"
              << "  --seeds N                 seeds per batch (" << defaults.numSeeds << ")\n"
              << "  --batches N               timed batches (" << defaults.numBatches << ")\n"
              << "  --warmup N                untimed batches before timing (" << defaults.warmupBatches << ")\n"
              << "  --alpha X                 restart probability (" << defaults.alpha << ")\n"
              << "  --eps X                   push threshold (" << defaults.eps << ")\n"
              << "  --max-ppr-nodes N         nodes extracted per seed (" << defaults.maxPPRNodes << ")\n"
              << "  --no-residual-top-up      disable residual top-up in extraction\n"
              << "  --random-seed N           graph and seed RNG seed (" << defaults.randomSeed << ")\n"
              << "  --dump-output PATH        write extracted rows for later comparison\n"
              << "  --compare-output PATH     compare extracted rows against a dump\n";
}

BenchmarkOptions parseOptions(int argc, char** argv) {
    BenchmarkOptions options;
    for (int argIndex = 1; argIndex < argc; ++argIndex) {
        std::string flag = argv[argIndex];
        auto nextValue = [&]() -> std::string {
            if (argIndex + 1 >= argc) {
                std::cerr << "Missing value for " << flag << "\n";
                std::exit(2);
            }
            ++argIndex;
            return argv[argIndex];
        };
        if (flag == "--help") {
            printUsage();
            std::exit(0);
        } else if (flag == "--nodes") {
            options.numNodes = std::stoll(nextValue());
        } else if (flag == "--mean-degree") {
            options.meanDegree = std::stod(nextValue());
        } else if (flag == "--degree-exponent") {
            options.degreeExponent = std::stod(nextValue());
        } else if (flag == "--max-degree") {
            options.maxDegree = std::stoll(nextValue());
        } else if (flag == "--max-neighbors") {
            options.maxNeighborsPerFetch = std::stoi(nextValue());
        } else if (flag == "--seeds") {
            options.numSeeds = std::stoi(nextValue());
        } else if (flag == "--batches") {
            options.numBatches = std::stoi(nextValue());
        } else if (flag == "--warmup") {
            options.warmupBatches = std::stoi(nextValue());
        } else if (flag == "--alpha") {
            options.alpha = std::stod(nextValue());
        } else if (flag == "--eps") {
            options.eps = std::stod(nextValue());
        } else if (flag == "--max-ppr-nodes") {
            options.maxPPRNodes = std::stoi(nextValue());
        } else if (flag == "--no-residual-top-up") {
            options.enableResidualTopUp = false;
        } else if (flag == "--random-seed") {
            options.randomSeed = std::stoull(nextValue());
        } else if (flag == "--dump-output") {
            options.dumpOutputPath = nextValue();
        } else if (flag == "--compare-output") {
            options.compareOutputPath = nextValue();
        } else {
            std::cerr << "Unknown flag " << flag << "\n";
            printUsage();
            std::exit(2);
        }
    }
    return options;
}

// Synthetic graph in CSR form. Degrees follow a truncated power law, and edge
// destinations are drawn with a bias toward low node IDs, so a small set of
// hub nodes appears in many neighbor lists, as in social graphs.
struct SyntheticGraph {
    std::vector<int64_t> rowOffsets;
    std::vector<int32_t> columnIndices;
    torch::Tensor degrees; // int32 [numNodes]
};

SyntheticGraph buildGraph(const BenchmarkOptions& options) {
    std::mt19937_64 generator(options.randomSeed);
    std::uniform_real_distribution<double> unit(0.0, 1.0);

    // Pareto-distributed degree with minimum chosen so the uncapped mean is
    // meanDegree: mean = minDegree * (exponent - 1) / (exponent - 2).
    const double exponent = options.degreeExponent;
    const double minDegree = options.meanDegree * (exponent - 2.0) / (exponent - 1.0);
    std::vector<int32_t> degreeValues(static_cast<size_t>(options.numNodes));
    for (auto& degree : degreeValues) {
        double sampled = minDegree * std::pow(1.0 - unit(generator), -1.0 / (exponent - 1.0));
        degree = static_cast<int32_t>(std::min<double>(std::max(1.0, sampled), static_cast<double>(options.maxDegree)));
    }

    SyntheticGraph graph;
    graph.rowOffsets.resize(static_cast<size_t>(options.numNodes) + 1, 0);
    for (int64_t nodeId = 0; nodeId < options.numNodes; ++nodeId) {
        graph.rowOffsets[nodeId + 1] = graph.rowOffsets[nodeId] + degreeValues[nodeId];
    }
    graph.columnIndices.resize(static_cast<size_t>(graph.rowOffsets.back()));
    // Destination = numNodes * u^3 puts about 46% of edge endpoints on the
    // lowest 10% of node IDs.
    for (auto& destination : graph.columnIndices) {
        double u = unit(generator);
        destination = static_cast<int32_t>(static_cast<double>(options.numNodes - 1) * u * u * u);
    }
    graph.degrees = torch::tensor(degreeValues, torch::kInt);
    return graph;
}

// Builds the pushResiduals() input for a drained frontier, capping each row at
// maxNeighborsPerFetch like the distributed one-hop sampler.
gigl::NeighborFetchMap fetchNeighbors(const SyntheticGraph& graph,
                                      const std::unordered_map<int32_t, torch::Tensor>& frontier,
                                      int32_t maxNeighborsPerFetch,
                                      int64_t& fetchedNodeCount) {
    gigl::NeighborFetchMap fetched;
    for (const auto& [edgeTypeId, nodeIds] : frontier) {
        auto nodeAccessor = nodeIds.accessor<int64_t, 1>();
        const int64_t numFrontierNodes = nodeIds.size(0);
        fetchedNodeCount += numFrontierNodes;
        std::vector<int64_t> counts(static_cast<size_t>(numFrontierNodes));
        int64_t totalNeighbors = 0;
        for (int64_t index = 0; index < numFrontierNodes; ++index) {
            int64_t nodeId = nodeAccessor[index];
            int64_t degree = graph.rowOffsets[nodeId + 1] - graph.rowOffsets[nodeId];
            counts[index] = std::min<int64_t>(degree, maxNeighborsPerFetch);
            totalNeighbors += counts[index];
        }
        auto flatNeighbors = torch::empty({totalNeighbors}, torch::kLong);
        int64_t* flatNeighborData = flatNeighbors.data_ptr<int64_t>();
        int64_t writeOffset = 0;
        for (int64_t index = 0; index < numFrontierNodes; ++index) {
            int64_t rowStart = graph.rowOffsets[nodeAccessor[index]];
            for (int64_t neighborIndex = 0; neighborIndex < counts[index]; ++neighborIndex) {
                flatNeighborData[writeOffset++] = graph.columnIndices[rowStart + neighborIndex];
            }
        }
        fetched.emplace(
            edgeTypeId,
            gigl::NeighborFetchTensors{nodeIds, flatNeighbors, torch::tensor(counts, torch::kLong), std::nullopt});
    }
    return fetched;
}

struct BatchTiming {
    double constructMs = 0.0;
    double drainMs = 0.0;
    double fetchMs = 0.0;
    double pushMs = 0.0;
    double extractMs = 0.0;
    int32_t iterations = 0;
    int64_t fetchedNodes = 0;
    int64_t extractedRows = 0;

    [[nodiscard]] double kernelMs() const {
        return constructMs + drainMs + pushMs + extractMs;
    }
};

using Clock = std::chrono::steady_clock;

double elapsedMs(Clock::time_point start) {
    return std::chrono::duration<double, std::milli>(Clock::now() - start).count();
}

struct BatchResult {
    BatchTiming timing;
    gigl::PPRExtractTensors extracted;
};

BatchResult runBatch(const SyntheticGraph& graph, const torch::Tensor& seeds, const BenchmarkOptions& options) {
    BatchResult result;
    BatchTiming& timing = result.timing;

    auto start = Clock::now();
    gigl::PPRForwardPush state(seeds,
                               /*seedNodeTypeId=*/0,
                               options.alpha,
                               options.alpha * options.eps,
                               /*nodeTypeToEdgeTypeIds=*/{{0}},
                               /*edgeTypeToDstNtypeId=*/{0},
                               {graph.degrees});
    timing.constructMs = elapsedMs(start);

    while (true) {
        start = Clock::now();
        auto frontier = state.drainQueue();
        timing.drainMs += elapsedMs(start);
        if (!frontier.has_value()) {
            break;
        }
        ++timing.iterations;

        start = Clock::now();
        auto fetched = fetchNeighbors(graph, frontier.value(), options.maxNeighborsPerFetch, timing.fetchedNodes);
        timing.fetchMs += elapsedMs(start);

        start = Clock::now();
        state.pushResiduals(fetched);
        timing.pushMs += elapsedMs(start);
    }

    start = Clock::now();
    auto extractedByNodeType = state.extractTopKWithResidualTopUp(options.maxPPRNodes, options.enableResidualTopUp);
    timing.extractMs = elapsedMs(start);
    result.extracted = extractedByNodeType.at(0);
    timing.extractedRows = result.extracted.ids.size(0);
    return result;
}

// One extracted row, keyed by (batch, seed) when dumped.
struct OutputRow {
    int64_t nodeId;
    double score;
    double hopProximity;
};
using OutputRows = std::map<std::pair<int32_t, int32_t>, std::vector<OutputRow>>;

void collectRows(int32_t batchIndex, const gigl::PPRExtractTensors& extracted, OutputRows& rows) {
    auto ids = extracted.ids.accessor<int64_t, 1>();
    auto weights = extracted.weights.accessor<double, 2>();
    auto validCounts = extracted.validCounts.accessor<int64_t, 1>();
    int64_t rowIndex = 0;
    for (int32_t seedIndex = 0; seedIndex < static_cast<int32_t>(extracted.validCounts.size(0)); ++seedIndex) {
        auto& seedRows = rows[{batchIndex, seedIndex}];
        for (int64_t count = 0; count < validCounts[seedIndex]; ++count, ++rowIndex) {
            seedRows.push_back(OutputRow{ids[rowIndex], weights[rowIndex][0], weights[rowIndex][1]});
        }
    }
}

void dumpRows(const OutputRows& rows, const std::string& path) {
    std::ofstream out(path);
    out.precision(17);
    for (const auto& [key, seedRows] : rows) {
        for (const auto& row : seedRows) {
            out << key.first << ' ' << key.second << ' ' << row.nodeId << ' ' << row.score << ' ' << row.hopProximity
                << '\n';
        }
    }
    std::cout << "wrote " << path << "\n";
}

// Compares this run's rows against a dump. Node IDs are matched per seed as a
// set, because summation order can change the last bits of a score and so the
// order of near-tied rows. Scores are compared for node IDs present in both.
// Returns false when any seed's node set or any score differs beyond tolerance.
bool compareRows(const OutputRows& rows, const std::string& path) {
    OutputRows reference;
    std::ifstream in(path);
    int32_t batchIndex = 0;
    int32_t seedIndex = 0;
    OutputRow row{};
    while (in >> batchIndex >> seedIndex >> row.nodeId >> row.score >> row.hopProximity) {
        reference[{batchIndex, seedIndex}].push_back(row);
    }

    int64_t seedsWithDifferentNodes = 0;
    int64_t missingRows = 0;
    int64_t hopMismatches = 0;
    double maxRelativeScoreDiff = 0.0;
    for (const auto& [key, referenceRows] : reference) {
        std::unordered_map<int64_t, const OutputRow*> rowsByNode;
        auto found = rows.find(key);
        if (found != rows.end()) {
            for (const auto& candidate : found->second) {
                rowsByNode.emplace(candidate.nodeId, &candidate);
            }
        }
        bool seedDiffers = found == rows.end() || found->second.size() != referenceRows.size();
        for (const auto& referenceRow : referenceRows) {
            auto match = rowsByNode.find(referenceRow.nodeId);
            if (match == rowsByNode.end()) {
                ++missingRows;
                seedDiffers = true;
                continue;
            }
            double scale = std::max(std::abs(referenceRow.score), 1e-300);
            maxRelativeScoreDiff =
                std::max(maxRelativeScoreDiff, std::abs(match->second->score - referenceRow.score) / scale);
            if (match->second->hopProximity != referenceRow.hopProximity) {
                ++hopMismatches;
            }
        }
        if (seedDiffers) {
            ++seedsWithDifferentNodes;
        }
    }
    std::printf("compare vs %s: %zu seeds, %ld seeds with a different node set, %ld missing rows, "
                "%ld hop mismatches, max relative score diff %.3e\n",
                path.c_str(),
                reference.size(),
                static_cast<long>(seedsWithDifferentNodes),
                static_cast<long>(missingRows),
                static_cast<long>(hopMismatches),
                maxRelativeScoreDiff);
    // A handful of seeds can swap a near-tied last row; flag anything more.
    return reference.size() == rows.size() && hopMismatches == 0 && maxRelativeScoreDiff < 1e-9 &&
           seedsWithDifferentNodes * 1000 <= static_cast<int64_t>(reference.size());
}

double median(std::vector<double> values) {
    std::sort(values.begin(), values.end());
    return values[values.size() / 2];
}

} // namespace

int main(int argc, char** argv) {
    BenchmarkOptions options = parseOptions(argc, argv);
    // The sampler calls the kernel from Python executor threads; intra-op
    // parallelism is not used by the kernel, so pin torch to one thread.
    torch::set_num_threads(1);

    auto graphStart = Clock::now();
    SyntheticGraph graph = buildGraph(options);
    std::printf("graph: %ld nodes, %ld edges, built in %.0f ms\n",
                static_cast<long>(options.numNodes),
                static_cast<long>(graph.rowOffsets.back()),
                elapsedMs(graphStart));

    std::mt19937_64 seedGenerator(options.randomSeed + 1);
    std::uniform_int_distribution<int64_t> seedDistribution(0, options.numNodes - 1);
    auto makeSeeds = [&]() {
        std::vector<int64_t> seeds(static_cast<size_t>(options.numSeeds));
        for (auto& seed : seeds) {
            seed = seedDistribution(seedGenerator);
        }
        return torch::tensor(seeds, torch::kLong);
    };

    for (int32_t warmupIndex = 0; warmupIndex < options.warmupBatches; ++warmupIndex) {
        runBatch(graph, makeSeeds(), options);
    }

    std::vector<BatchTiming> timings;
    OutputRows rows;
    bool keepRows = !options.dumpOutputPath.empty() || !options.compareOutputPath.empty();
    for (int32_t batchIndex = 0; batchIndex < options.numBatches; ++batchIndex) {
        BatchResult result = runBatch(graph, makeSeeds(), options);
        timings.push_back(result.timing);
        if (keepRows) {
            collectRows(batchIndex, result.extracted, rows);
        }
    }

    auto medianOf = [&](auto field) {
        std::vector<double> values;
        values.reserve(timings.size());
        for (const auto& timing : timings) {
            values.push_back(static_cast<double>(field(timing)));
        }
        return median(values);
    };
    std::printf("per-batch medians over %d batches of %d seeds:\n", options.numBatches, options.numSeeds);
    std::printf("  kernel_ms     %9.2f  (construct + drain + push + extract)\n",
                medianOf([](const BatchTiming& t) { return t.kernelMs(); }));
    std::printf("  construct_ms  %9.2f\n", medianOf([](const BatchTiming& t) { return t.constructMs; }));
    std::printf("  drain_ms      %9.2f\n", medianOf([](const BatchTiming& t) { return t.drainMs; }));
    std::printf("  push_ms       %9.2f\n", medianOf([](const BatchTiming& t) { return t.pushMs; }));
    std::printf("  extract_ms    %9.2f\n", medianOf([](const BatchTiming& t) { return t.extractMs; }));
    std::printf("  fetch_ms      %9.2f  (synthetic neighbor fetch, not kernel time)\n",
                medianOf([](const BatchTiming& t) { return t.fetchMs; }));
    std::printf("  iterations    %9.1f\n", medianOf([](const BatchTiming& t) { return t.iterations; }));
    std::printf("  fetched_nodes %9.0f\n", medianOf([](const BatchTiming& t) { return t.fetchedNodes; }));
    std::printf("  output_rows   %9.0f\n", medianOf([](const BatchTiming& t) { return t.extractedRows; }));

    if (!options.dumpOutputPath.empty()) {
        dumpRows(rows, options.dumpOutputPath);
    }
    if (!options.compareOutputPath.empty() && !compareRows(rows, options.compareOutputPath)) {
        std::cerr << "output differs from " << options.compareOutputPath << "\n";
        return 1;
    }
    return 0;
}
