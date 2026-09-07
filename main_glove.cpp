/*
 * Staleness simulator for IVF index on GloVe (word embeddings).
 *
 * Same three-index setup as the SIFT version, but adapted for GloVe:
 *   - Uses inner product (cosine on unit vectors) instead of L2.
 *   - Dimension is 100 (GloVe.6B) instead of 128 (SIFT).
 *   - Vectors are already L2-normalized in the .fvecs files, so IP
 *     search is equivalent to cosine similarity.
 *
 * The interesting property vs. SIFT: GloVe words in the base file are
 * ordered by decreasing frequency. Training uses the top 50k most-common
 * words, then the mutation loop adds progressively rarer words (300k of
 * them). Rare words tend to occupy different regions of the embedding
 * space (specialized terms cluster around each other), so frozen
 * centroids should under-cover those regions — and mutable centroids
 * should drift outward to fill them.
 *
 * DEV_MODE=1 → fast iteration (fewer queries, sparser measurements).
 * DEV_MODE=0 → full-quality run.
 */

#include <cassert>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <tuple>
#include <fstream>
#include <unordered_set>

#include <sys/resource.h>
#include <sys/stat.h>
#include <sys/time.h>

#include <faiss/AutoTune.h>
#include <faiss/IndexFlat.h>
#include <faiss/IndexIVF.h>
#include <faiss/IndexIVFFlat.h>
#include <faiss/IndexIVFMutable.h>
#include <faiss/index_factory.h>

// ============================== CONFIG ===============================
#define DEV_MODE 1
// =====================================================================

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

double cpu_seconds() {
    struct rusage ru;
    getrusage(RUSAGE_SELF, &ru);
    return (ru.ru_utime.tv_sec + ru.ru_utime.tv_usec * 1e-6) +
           (ru.ru_stime.tv_sec + ru.ru_stime.tv_usec * 1e-6);
}

int main() {
    double t0 = elapsed();

    // ------------------------------------------------------------------ config
    // GloVe.6B has 400k vocab in the file we prepared:
    //   glove_learn.fvecs = 50k common words
    //   glove_base.fvecs  = next 300k words (progressively rarer)
    //   glove_query.fvecs = 10k queries sampled uniformly across vocab
    //
    // sqrt(300k) ≈ 550, so nlist=512 is a reasonable choice.
    // (SIFT1M uses nlist=1024 or 4096; we scale down for the smaller base.)
    const size_t nlist       = 512;
    const size_t n_initial   = 30000;  // ~10% of base — the initial snapshot
    const size_t batch_size  = 100;

#if DEV_MODE
    const size_t measure_every = 50;
    const size_t max_queries   = 1000;
    printf("[MODE] DEV — measure_every=%zu, using %zu queries\n",
           measure_every, max_queries);
#else
    const size_t measure_every = 10;
    const size_t max_queries   = 0;
    printf("[MODE] FULL — measure_every=%zu, using all queries\n",
           measure_every);
#endif

    // GloVe = word embeddings → cosine similarity. Vectors are already
    // unit-normalized during prep, so inner product == cosine similarity,
    // and we search for the LARGEST inner products (nearest neighbors).
    const faiss::MetricType metric = faiss::METRIC_INNER_PRODUCT;
    const int nprobe = 32;

    size_t d = 0;
    double frozen_cpu_s = 0.0;
    double mutable_cpu_s = 0.0;

    // ------------------------------------------------------------------ train
    faiss::IndexFlat* frozen_quantizer = nullptr;
    faiss::IndexFlat* mutable_quantizer = nullptr;
    faiss::IndexIVFFlat* frozen_index = nullptr;
    faiss::IndexIVFMutable* mutable_index = nullptr;

    {
        printf("[%.3f s] Loading train set\n", elapsed() - t0);
        size_t nt;
        float* xt = fvecs_read("glove/glove_learn.fvecs", &d, &nt);

        printf("[%.3f s] Building frozen IVF%zu (IP)  d=%zu\n",
               elapsed() - t0, nlist, d);
        frozen_quantizer = new faiss::IndexFlat(d, metric);
        frozen_index = new faiss::IndexIVFFlat(frozen_quantizer, d, nlist,
                                                metric);
        frozen_index->own_fields = true;

        printf("[%.3f s] Building mutable IVF%zu (IP) d=%zu\n",
               elapsed() - t0, nlist, d);
        mutable_quantizer = new faiss::IndexFlat(d, metric);
        mutable_index = new faiss::IndexIVFMutable(mutable_quantizer, d, nlist,
                                                    metric);
        mutable_index->own_fields = true;

        printf("[%.3f s] Training both on %zu vectors\n", elapsed() - t0, nt);
        frozen_index->train(nt, xt);
        mutable_index->train(nt, xt);
        delete[] xt;
       
    }

    // --------------------------------------------------------------- load base
    size_t nb;
    float* xb;
    {
        size_t d2;
        xb = fvecs_read("glove/glove_base.fvecs", &d2, &nb);
        assert(d == d2 && "base dimension mismatch");
        printf("[%.3f s] Loaded base set: %zu vectors\n", elapsed() - t0, nb);
    }

    assert(n_initial <= nb && "n_initial exceeds base set size");
    printf("[%.3f s] Adding initial %zu vectors to all three indexes\n",
           elapsed() - t0, n_initial);
    {
        double t = cpu_seconds();
        frozen_index->add(n_initial, xb);
        frozen_cpu_s += cpu_seconds() - t;

        t = cpu_seconds();
        mutable_index->add(n_initial, xb);
        mutable_cpu_s += cpu_seconds() - t;
    }

    frozen_index->nprobe = nprobe;
    mutable_index->nprobe = nprobe;
    printf("[%.3f s] nprobe set to %d on both IVF indexes\n",
           elapsed() - t0, nprobe);

    printf("[%.3f s] Building ground truth index on initial %zu vectors\n",
           elapsed() - t0, n_initial);
    faiss::IndexFlat* gt_index = new faiss::IndexFlat(d, metric);
    gt_index->add(n_initial, xb);

    // ---------------------------------------------------------- load queries
    size_t nq_full;
    float* xq;
    {
        size_t d2;
        xq = fvecs_read("glove/glove_query.fvecs", &d2, &nq_full);
        assert(d == d2 && "query dimension mismatch");
        printf("[%.3f s] Loaded %zu queries\n", elapsed() - t0, nq_full);
    }

    const size_t nq = (max_queries > 0 && max_queries < nq_full)
                       ? max_queries : nq_full;
    if (nq < nq_full) {
        printf("[%.3f s] Using %zu / %zu queries for this run\n",
               elapsed() - t0, nq, nq_full);
    }

    // ---------------------------------------------------------- search buffers
    const size_t k = 100;
    faiss::Index::idx_t* I_frozen  = new faiss::Index::idx_t[nq * k];
    float*               D_frozen  = new float[nq * k];
    faiss::Index::idx_t* I_mutable = new faiss::Index::idx_t[nq * k];
    float*               D_mutable = new float[nq * k];
    faiss::Index::idx_t* I_gt      = new faiss::Index::idx_t[nq * k];
    float*               D_gt      = new float[nq * k];

    auto measure_recalls = [&]() -> std::tuple<std::tuple<float,float,float>,
                                                std::tuple<float,float,float>> {
        frozen_index->search(nq, xq, k, D_frozen, I_frozen);
        mutable_index->search(nq, xq, k, D_mutable, I_mutable);
        gt_index->search(nq, xq, k, D_gt, I_gt);

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

    std::ofstream metrics_file("recall_metrics.csv");
    metrics_file << "gt_ntotal,frozen_ntotal,mutable_ntotal,"
                    "frozen_R@1,frozen_R@10,frozen_R@100,"
                    "mutable_R@1,mutable_R@10,mutable_R@100,"
                    "frozen_cpu_s,mutable_cpu_s\n";

    auto log_and_print = [&](const char* tag) {
        auto [frozen_rs, mutable_rs] = measure_recalls();
        auto [fr1, fr10, fr100] = frozen_rs;
        auto [mr1, mr10, mr100] = mutable_rs;
        printf("[%.3f s] gt=%7lld  frozen=%7lld  mutable=%7lld  "
               "frozen(%.3f/%.3f/%.3f)  mutable(%.3f/%.3f/%.3f)  "
               "cpu(f=%.2fs m=%.2fs)  %s\n",
               elapsed() - t0, gt_index->ntotal,
               frozen_index->ntotal, mutable_index->ntotal,
               fr1, fr10, fr100, mr1, mr10, mr100,
               frozen_cpu_s, mutable_cpu_s, tag);
        metrics_file << gt_index->ntotal << ","
                     << frozen_index->ntotal << ","
                     << mutable_index->ntotal << ","
                     << fr1 << "," << fr10 << "," << fr100 << ","
                     << mr1 << "," << mr10 << "," << mr100 << ","
                     << frozen_cpu_s << "," << mutable_cpu_s << "\n";
        metrics_file.flush();
    };

    log_and_print("[snapshot]");

    size_t n_remaining = nb - n_initial;
    size_t n_batches   = n_remaining / batch_size;

    printf("[%.3f s] Starting mutation loop: %zu batches x %zu vectors "
           "(measuring every %zu batches)\n",
           elapsed() - t0, n_batches, batch_size, measure_every);

    for (size_t b = 0; b < n_batches; b++) {
        size_t offset = n_initial + b * batch_size;

        gt_index->add(batch_size, xb + offset * d);

        double t = cpu_seconds();
        mutable_index->add(batch_size, xb + offset * d);
        mutable_cpu_s += cpu_seconds() - t;

        t = cpu_seconds();
        frozen_index->add(batch_size, xb + offset * d);
        frozen_cpu_s += cpu_seconds() - t;

        if ((b + 1) % measure_every == 0) {
            log_and_print("");
        }
    }

    size_t added_so_far = n_initial + n_batches * batch_size;
    if (added_so_far < nb) {
        size_t leftover = nb - added_so_far;
        gt_index->add(leftover, xb + added_so_far * d);

        double t = cpu_seconds();
        mutable_index->add(leftover, xb + added_so_far * d);
        mutable_cpu_s += cpu_seconds() - t;

        t = cpu_seconds();
        frozen_index->add(leftover, xb + added_so_far * d);
        frozen_cpu_s += cpu_seconds() - t;

        log_and_print("[final]");
    }

    metrics_file.close();
    delete[] I_frozen;
    delete[] D_frozen;
    delete[] I_mutable;
    delete[] D_mutable;
    delete[] I_gt;
    delete[] D_gt;
    delete[] xb;
    delete[] xq;
    delete gt_index;
    delete frozen_index;
    delete mutable_index;
    return 0;
}