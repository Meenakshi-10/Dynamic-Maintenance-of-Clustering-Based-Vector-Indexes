/*
 * Staleness simulator for IVF index — comparison of frozen vs. mutable.
 *
 * Builds three indexes on the same initial snapshot:
 *   1. A brute-force ground truth index (IndexFlatL2), which tracks every mutation.
 *   2. A frozen IVF1024,Flat index that never sees mutations (baseline for staleness).
 *   3. A mutable IVF1024,Flat (IndexIVFMutable) that receives every mutation and
 *      updates centroids incrementally via streaming-mean perturbation.
 *
 * On each measurement step we run the same query batch against all three,
 * compute recall of (2) vs. (1) and recall of (3) vs. (1), and log both.
 *
 * R@X = average fraction of true top-X neighbors recovered across all queries.
 *
 * The whole experiment is repeated N_RUNS times, each with a different random
 * permutation of the base set (so both which vectors land in the initial
 * snapshot and the order they mutate in afterward vary run to run). Each run
 * writes its own CSV (recall_metrics_run{i}.csv); a final recall_metrics_avg.csv
 * holds the mean and stddev across runs at each matching checkpoint.
 */

#include <algorithm>
#include <cassert>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <tuple>
#include <fstream>
#include <limits>
#include <numeric>
#include <random>
#include <unordered_set>
#include <utility>
#include <vector>

#include <sys/resource.h>
#include <sys/stat.h>
#include <sys/time.h>

#include <faiss/AutoTune.h>
#include <faiss/IndexFlat.h>
#include <faiss/IndexIVF.h>
#include <faiss/IndexIVFFlat.h>
#include <faiss/IndexIVFMutable.h>
#include <faiss/index_factory.h>

/*****************************************************
 * I/O functions for fvecs and ivecs
 *****************************************************/

float* fvecs_read(const char* fname, size_t* d_out, size_t* n_out) {
    FILE* f = fopen(fname, "r");
    if (!f) {
        fprintf(stderr, "could not open %s\n", fname);
        perror("");
        abort();
    }
    int d;
    fread(&d, 1, sizeof(int), f);
    assert((d > 0 && d < 1000000) && "unreasonable dimension");
    fseek(f, 0, SEEK_SET);
    struct stat st;
    fstat(fileno(f), &st);
    size_t sz = st.st_size;
    assert(sz % ((d + 1) * 4) == 0 && "weird file size");
    size_t n = sz / ((d + 1) * 4);

    *d_out = d;
    *n_out = n;
    float* x = new float[n * (d + 1)];
    size_t nr __attribute__((unused)) = fread(x, sizeof(float), n * (d + 1), f);
    assert(nr == n * (d + 1) && "could not read whole file");

    for (size_t i = 0; i < n; i++)
        memmove(x + i * d, x + 1 + i * (d + 1), d * sizeof(*x));

    fclose(f);
    return x;
}

int* ivecs_read(const char* fname, size_t* d_out, size_t* n_out) {
    return (int*)fvecs_read(fname, d_out, n_out);
}

double elapsed() {
    struct timeval tv;
    gettimeofday(&tv, nullptr);
    return tv.tv_sec + tv.tv_usec * 1e-6;
}

// Cumulative process CPU time (user + sys) in seconds, via getrusage.
double cpu_time() {
    struct rusage ru;
    getrusage(RUSAGE_SELF, &ru);
    return (ru.ru_utime.tv_sec + ru.ru_utime.tv_usec * 1e-6) +
           (ru.ru_stime.tv_sec + ru.ru_stime.tv_usec * 1e-6);
}

// Distribution of inverted-list (cluster) sizes across all nlist buckets.
struct ClusterStats {
    double mean, stddev, min, max;
};

static ClusterStats compute_cluster_stats(const faiss::IndexIVF* index) {
    size_t nl = index->nlist;
    double sum = 0, sum_sq = 0;
    double mn = std::numeric_limits<double>::max(), mx = 0;
    for (size_t i = 0; i < nl; i++) {
        double sz = (double)index->get_list_size(i);
        sum += sz;
        sum_sq += sz * sz;
        mn = std::min(mn, sz);
        mx = std::max(mx, sz);
    }
    double mean = sum / nl;
    double var = sum_sq / nl - mean * mean;
    return {mean, std::sqrt(std::max(var, 0.0)), mn, mx};
}

// ---------------------------------------------------------------- config
static const size_t nlist       = 1024;
static const size_t n_initial   = 100000; // snapshot size — frozen index stops here
static const size_t batch_size  = 100;    // vectors added per mutation step
static const size_t measure_every = 50;   // measure recall every N batches
static const int    nprobe      = 64;     // fixed for all IVF searches
static const size_t k           = 100;

static const size_t n_runs    = 3;   // number of insertion-order trials
static const unsigned base_seed = 42; // run i uses seed base_seed + i

// One row of the recall_metrics.csv output.
struct Metrics {
    long long gt_ntotal, frozen_ntotal, mutable_ntotal;
    float fr1, fr10, fr100;
    float mr1, mr10, mr100;
    double frozen_latency_ms, mutable_latency_ms; // wall-clock search time / query
    double cpu_time_s;                            // cumulative process CPU time so far
    double cpu_delta_s;                            // CPU time since previous checkpoint
    double frozen_cluster_mean, frozen_cluster_std, frozen_cluster_min, frozen_cluster_max;
    double mutable_cluster_mean, mutable_cluster_std, mutable_cluster_min, mutable_cluster_max;
};

// Gathers `count` d-dimensional vectors from `src`, selected via `perm`
// starting at perm[start], into the contiguous buffer `dst`.
static void gather(float* dst, const float* src, const std::vector<size_t>& perm,
                    size_t start, size_t count, size_t d) {
    for (size_t i = 0; i < count; i++)
        memcpy(dst + i * d, src + perm[start + i] * d, d * sizeof(float));
}

// Runs one full trial (train -> snapshot -> mutation loop) using a random
// permutation of the base set derived from `seed`. Writes
// recall_metrics_run{run_idx}.csv and returns the per-checkpoint metrics
// (in checkpoint order) so the caller can average across runs.
static std::vector<Metrics> run_experiment(size_t run_idx, unsigned seed,
                                            double t0,
                                            const float* xt, size_t nt,
                                            const float* xb, size_t nb,
                                            const float* xq, size_t nq,
                                            size_t d) {
    printf("[%.3f s] ===== run %ld (seed=%u) =====\n", elapsed() - t0, run_idx, seed);
    double cpu_t0 = cpu_time(); // per-run baseline, so cpu_time_s is comparable across runs

    // ------------------------------------------------------------ permute
    std::vector<size_t> perm(nb);
    std::iota(perm.begin(), perm.end(), 0);
    std::mt19937 rng(seed);
    std::shuffle(perm.begin(), perm.end(), rng);

    // ------------------------------------------------------------ train
    // Both IVF indexes are trained on the same learn set (order-independent),
    // so their initial centroids are identical across runs. The subsequent
    // divergence is due to insertion order plus the mutable index's
    // perturbation updates.
    faiss::IndexFlatL2* frozen_quantizer = new faiss::IndexFlatL2(d);
    faiss::IndexIVFFlat* frozen_index = new faiss::IndexIVFFlat(frozen_quantizer, d, nlist);
    frozen_index->own_fields = true;

    faiss::IndexFlatL2* mutable_quantizer = new faiss::IndexFlatL2(d);
    faiss::IndexIVFMutable* mutable_index = new faiss::IndexIVFMutable(mutable_quantizer, d, nlist);
    mutable_index->own_fields = true;

    frozen_index->train(nt, xt);
    mutable_index->train(nt, xt);

    // ------------------------------------------- populate initial snapshot
    // All three indexes start with the same n_initial vectors (chosen by the
    // permutation for this run). After this point, the frozen index stops
    // receiving vectors; the ground-truth and mutable indexes continue.
    assert(n_initial <= nb && "n_initial exceeds base set size");
    std::vector<float> snapshot_buf(n_initial * d);
    gather(snapshot_buf.data(), xb, perm, 0, n_initial, d);

    frozen_index->add(n_initial, snapshot_buf.data());
    mutable_index->add(n_initial, snapshot_buf.data());

    frozen_index->nprobe = nprobe;
    mutable_index->nprobe = nprobe;

    faiss::IndexFlatL2* gt_index = new faiss::IndexFlatL2(d);
    gt_index->add(n_initial, snapshot_buf.data());
    snapshot_buf.clear();
    snapshot_buf.shrink_to_fit();

    // ---------------------------------------------------------- search buffers
    faiss::Index::idx_t* I_frozen  = new faiss::Index::idx_t[nq * k];
    float*               D_frozen  = new float[nq * k];
    faiss::Index::idx_t* I_mutable = new faiss::Index::idx_t[nq * k];
    float*               D_mutable = new float[nq * k];
    faiss::Index::idx_t* I_gt      = new faiss::Index::idx_t[nq * k];
    float*               D_gt      = new float[nq * k];

    // -------------------------------------------------- recall helper lambda
    // Set-based recall: for each query, count how many IDs from the true top-r
    // appear in the retrieved top-r, divide by r, average across queries.
    // Set membership avoids the double-counting issue of nested loops.
    double frozen_latency_ms = 0, mutable_latency_ms = 0;

    auto measure_recalls = [&]() -> std::tuple<std::tuple<float,float,float>,
                                                std::tuple<float,float,float>> {
        double ts0 = elapsed();
        frozen_index->search(nq, xq, k, D_frozen, I_frozen);
        double ts1 = elapsed();
        mutable_index->search(nq, xq, k, D_mutable, I_mutable);
        double ts2 = elapsed();
        gt_index->search(nq, xq, k, D_gt, I_gt);

        frozen_latency_ms = (ts1 - ts0) * 1000.0 / nq;
        mutable_latency_ms = (ts2 - ts1) * 1000.0 / nq;

        float fr1 = 0, fr10 = 0, fr100 = 0;
        float mr1 = 0, mr10 = 0, mr100 = 0;

        auto recall_at_r = [&](size_t i, size_t r,
                               const faiss::Index::idx_t* retrieved) -> float {
            std::unordered_set<faiss::Index::idx_t> truth;
            for (size_t g = 0; g < r; g++)
                truth.insert(I_gt[i * k + g]);
            int hits = 0;
            for (size_t j = 0; j < r; j++)
                if (truth.count(retrieved[i * k + j])) hits++;
            return float(hits) / float(r);
        };

        for (size_t i = 0; i < nq; i++) {
            fr1   += recall_at_r(i, 1,   I_frozen);
            fr10  += recall_at_r(i, 10,  I_frozen);
            fr100 += recall_at_r(i, 100, I_frozen);
            mr1   += recall_at_r(i, 1,   I_mutable);
            mr10  += recall_at_r(i, 10,  I_mutable);
            mr100 += recall_at_r(i, 100, I_mutable);
        }
        return {
            {fr1 / nq, fr10 / nq, fr100 / nq},
            {mr1 / nq, mr10 / nq, mr100 / nq}
        };
    };

    // -------------------------------------------------- output CSV
    std::string csv_name = "recall_metrics_run" + std::to_string(run_idx) + ".csv";
    std::ofstream metrics_file(csv_name);
    metrics_file << "gt_ntotal,frozen_ntotal,mutable_ntotal,"
                    "frozen_R@1,frozen_R@10,frozen_R@100,"
                    "mutable_R@1,mutable_R@10,mutable_R@100,"
                    "frozen_latency_ms,mutable_latency_ms,"
                    "cpu_time_s,cpu_delta_s,"
                    "frozen_cluster_mean,frozen_cluster_std,frozen_cluster_min,frozen_cluster_max,"
                    "mutable_cluster_mean,mutable_cluster_std,mutable_cluster_min,mutable_cluster_max\n";

    std::vector<Metrics> checkpoints;
    double prev_cpu = cpu_t0;

    auto log_and_print = [&](const char* tag) {
        auto [frozen_rs, mutable_rs] = measure_recalls();
        auto [fr1, fr10, fr100] = frozen_rs;
        auto [mr1, mr10, mr100] = mutable_rs;

        double cpu_now = cpu_time();
        double cpu_time_s = cpu_now - cpu_t0;
        double cpu_delta_s = cpu_now - prev_cpu;
        prev_cpu = cpu_now;

        ClusterStats fcs = compute_cluster_stats(frozen_index);
        ClusterStats mcs = compute_cluster_stats(mutable_index);

        printf("[%.3f s] run=%ld gt=%7lld  frozen=%7lld  mutable=%7lld  "
               "frozen(%.3f/%.3f/%.3f)  mutable(%.3f/%.3f/%.3f)  "
               "latency(f=%.3fms m=%.3fms)  cpu(total=%.2fs delta=%.2fs)  "
               "cluster_std(f=%.1f m=%.1f)  %s\n",
               elapsed() - t0, run_idx, gt_index->ntotal,
               frozen_index->ntotal, mutable_index->ntotal,
               fr1, fr10, fr100, mr1, mr10, mr100,
               frozen_latency_ms, mutable_latency_ms,
               cpu_time_s, cpu_delta_s,
               fcs.stddev, mcs.stddev, tag);
        metrics_file << gt_index->ntotal << ","
                     << frozen_index->ntotal << ","
                     << mutable_index->ntotal << ","
                     << fr1 << "," << fr10 << "," << fr100 << ","
                     << mr1 << "," << mr10 << "," << mr100 << ","
                     << frozen_latency_ms << "," << mutable_latency_ms << ","
                     << cpu_time_s << "," << cpu_delta_s << ","
                     << fcs.mean << "," << fcs.stddev << "," << fcs.min << "," << fcs.max << ","
                     << mcs.mean << "," << mcs.stddev << "," << mcs.min << "," << mcs.max << "\n";
        checkpoints.push_back({gt_index->ntotal, frozen_index->ntotal, mutable_index->ntotal,
                                fr1, fr10, fr100, mr1, mr10, mr100,
                                frozen_latency_ms, mutable_latency_ms,
                                cpu_time_s, cpu_delta_s,
                                fcs.mean, fcs.stddev, fcs.min, fcs.max,
                                mcs.mean, mcs.stddev, mcs.min, mcs.max});
    };

    // ------------------------------------------- recall at initial snapshot
    log_and_print("[snapshot]");

    // ---------------------------------------------- mutation loop
    // Each iteration adds a batch of vectors (in permuted order) to both the
    // ground-truth index AND the mutable index. The frozen index is
    // deliberately left alone.
    size_t n_remaining = nb - n_initial;
    size_t n_batches   = n_remaining / batch_size;
    std::vector<float> batch_buf(batch_size * d);

    printf("[%.3f s] run=%ld Starting mutation loop: %ld batches x %ld vectors "
           "(measuring every %ld batches)\n",
           elapsed() - t0, run_idx, n_batches, batch_size, measure_every);

    for (size_t b = 0; b < n_batches; b++) {
        size_t offset = n_initial + b * batch_size;
        gather(batch_buf.data(), xb, perm, offset, batch_size, d);

        gt_index->add(batch_size, batch_buf.data());
        mutable_index->add(batch_size, batch_buf.data());
        // frozen index deliberately NOT updated

        if ((b + 1) % measure_every == 0) {
            log_and_print("");
        }
    }

    // ---------------------------------------------------- flush remainder
    size_t added_so_far = n_initial + n_batches * batch_size;
    if (added_so_far < nb) {
        size_t leftover = nb - added_so_far;
        std::vector<float> tail_buf(leftover * d);
        gather(tail_buf.data(), xb, perm, added_so_far, leftover, d);
        gt_index->add(leftover, tail_buf.data());
        mutable_index->add(leftover, tail_buf.data());
        log_and_print("[final]");
    }

    // ---------------------------------------------------------------- cleanup
    metrics_file.close();
    delete[] I_frozen;
    delete[] D_frozen;
    delete[] I_mutable;
    delete[] D_mutable;
    delete[] I_gt;
    delete[] D_gt;
    delete gt_index;
    delete frozen_index;    // will delete frozen_quantizer via own_fields
    delete mutable_index;   // will delete mutable_quantizer via own_fields

    return checkpoints;
}

int main() {
    double t0 = elapsed();
    size_t d = 0;

    printf("[%.3f s] Loading train set\n", elapsed() - t0);
    size_t nt;
    float* xt = fvecs_read("sift1M/sift_learn.fvecs", &d, &nt);

    size_t nb;
    float* xb;
    {
        size_t d2;
        xb = fvecs_read("sift1M/sift_base.fvecs", &d2, &nb);
        assert(d == d2 && "base dimension mismatch");
        printf("[%.3f s] Loaded base set: %ld vectors\n", elapsed() - t0, nb);
    }

    size_t nq;
    float* xq;
    {
        size_t d2;
        xq = fvecs_read("sift1M/sift_query.fvecs", &d2, &nq);
        assert(d == d2 && "query dimension mismatch");
        printf("[%.3f s] Loaded %ld queries\n", elapsed() - t0, nq);
    }

    // --------------------------------------------------------- run trials
    std::vector<std::vector<Metrics>> all_runs;
    all_runs.reserve(n_runs);
    for (size_t r = 0; r < n_runs; r++) {
        all_runs.push_back(run_experiment(r, base_seed + (unsigned)r, t0,
                                           xt, nt, xb, nb, xq, nq, d));
    }

    // ----------------------------------------------- average across runs
    // All runs share the same n_initial/batch_size/measure_every/nb, so they
    // produce the same number of checkpoints at the same ntotal values —
    // only the recall values differ. Average (and stddev) those directly by
    // checkpoint index.
    size_t n_checkpoints = all_runs.empty() ? 0 : all_runs[0].size();
    bool aligned = true;
    for (auto& run : all_runs)
        if (run.size() != n_checkpoints) aligned = false;

    if (!aligned) {
        fprintf(stderr, "warning: runs produced different numbers of checkpoints; "
                         "skipping recall_metrics_avg.csv\n");
    } else if (n_checkpoints > 0) {
        std::ofstream avg_file("recall_metrics_avg.csv");
        avg_file << "gt_ntotal,frozen_ntotal,mutable_ntotal,"
                     "frozen_R@1_mean,frozen_R@1_std,"
                     "frozen_R@10_mean,frozen_R@10_std,"
                     "frozen_R@100_mean,frozen_R@100_std,"
                     "mutable_R@1_mean,mutable_R@1_std,"
                     "mutable_R@10_mean,mutable_R@10_std,"
                     "mutable_R@100_mean,mutable_R@100_std,"
                     "frozen_latency_ms_mean,frozen_latency_ms_std,"
                     "mutable_latency_ms_mean,mutable_latency_ms_std,"
                     "cpu_time_s_mean,cpu_time_s_std,"
                     "cpu_delta_s_mean,cpu_delta_s_std,"
                     "frozen_cluster_mean_mean,frozen_cluster_mean_std,"
                     "frozen_cluster_std_mean,frozen_cluster_std_std,"
                     "frozen_cluster_min_mean,frozen_cluster_min_std,"
                     "frozen_cluster_max_mean,frozen_cluster_max_std,"
                     "mutable_cluster_mean_mean,mutable_cluster_mean_std,"
                     "mutable_cluster_std_mean,mutable_cluster_std_std,"
                     "mutable_cluster_min_mean,mutable_cluster_min_std,"
                     "mutable_cluster_max_mean,mutable_cluster_max_std\n";

        auto mean_std = [&](size_t c, auto field) -> std::pair<double,double> {
            double sum = 0;
            for (auto& run : all_runs) sum += run[c].*field;
            double mean = sum / n_runs;
            double var = 0;
            for (auto& run : all_runs) {
                double diff = run[c].*field - mean;
                var += diff * diff;
            }
            var = n_runs > 1 ? var / (n_runs - 1) : 0.0;
            return {mean, std::sqrt(var)};
        };

        for (size_t c = 0; c < n_checkpoints; c++) {
            auto [fr1m, fr1s]     = mean_std(c, &Metrics::fr1);
            auto [fr10m, fr10s]   = mean_std(c, &Metrics::fr10);
            auto [fr100m, fr100s] = mean_std(c, &Metrics::fr100);
            auto [mr1m, mr1s]     = mean_std(c, &Metrics::mr1);
            auto [mr10m, mr10s]   = mean_std(c, &Metrics::mr10);
            auto [mr100m, mr100s] = mean_std(c, &Metrics::mr100);

            auto [flatm, flats] = mean_std(c, &Metrics::frozen_latency_ms);
            auto [mlatm, mlats] = mean_std(c, &Metrics::mutable_latency_ms);
            auto [cpum, cpus]   = mean_std(c, &Metrics::cpu_time_s);
            auto [cpudm, cpuds] = mean_std(c, &Metrics::cpu_delta_s);

            auto [fcmm, fcms] = mean_std(c, &Metrics::frozen_cluster_mean);
            auto [fcsm, fcss] = mean_std(c, &Metrics::frozen_cluster_std);
            auto [fcnm, fcns] = mean_std(c, &Metrics::frozen_cluster_min);
            auto [fcxm, fcxs] = mean_std(c, &Metrics::frozen_cluster_max);
            auto [mcmm, mcms] = mean_std(c, &Metrics::mutable_cluster_mean);
            auto [mcsm, mcss] = mean_std(c, &Metrics::mutable_cluster_std);
            auto [mcnm, mcns] = mean_std(c, &Metrics::mutable_cluster_min);
            auto [mcxm, mcxs] = mean_std(c, &Metrics::mutable_cluster_max);

            avg_file << all_runs[0][c].gt_ntotal << ","
                     << all_runs[0][c].frozen_ntotal << ","
                     << all_runs[0][c].mutable_ntotal << ","
                     << fr1m << "," << fr1s << ","
                     << fr10m << "," << fr10s << ","
                     << fr100m << "," << fr100s << ","
                     << mr1m << "," << mr1s << ","
                     << mr10m << "," << mr10s << ","
                     << mr100m << "," << mr100s << ","
                     << flatm << "," << flats << ","
                     << mlatm << "," << mlats << ","
                     << cpum << "," << cpus << ","
                     << cpudm << "," << cpuds << ","
                     << fcmm << "," << fcms << ","
                     << fcsm << "," << fcss << ","
                     << fcnm << "," << fcns << ","
                     << fcxm << "," << fcxs << ","
                     << mcmm << "," << mcms << ","
                     << mcsm << "," << mcss << ","
                     << mcnm << "," << mcns << ","
                     << mcxm << "," << mcxs << "\n";
        }
        avg_file.close();
        printf("[%.3f s] Wrote recall_metrics_avg.csv (%ld checkpoints across %ld runs)\n",
               elapsed() - t0, n_checkpoints, n_runs);
    }

    delete[] xt;
    delete[] xb;
    delete[] xq;
    return 0;
}
