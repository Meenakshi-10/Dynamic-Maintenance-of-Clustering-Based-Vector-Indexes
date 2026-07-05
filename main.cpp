/*
 * Staleness simulator for IVF index — comparison of frozen vs. mutable.
 *
 * Builds three indexes on the same initial snapshot:
 *   1. A brute-force ground truth index (IndexFlatL2), which tracks every mutation.
 *   2. A frozen IVF4096,Flat index that never sees mutations (baseline for staleness).
 *   3. A mutable IVF4096,Flat (IndexIVFMutable) that receives every mutation and
 *      updates centroids incrementally via streaming-mean perturbation.
 *
 * On each measurement step we run the same query batch against all three,
 * compute recall of (2) vs. (1) and recall of (3) vs. (1), and log both.
 *
 * R@X = average fraction of true top-X neighbors recovered across all queries.
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

int main() {
    double t0 = elapsed();

    // ------------------------------------------------------------------ config
    const size_t nlist       = 4096;
    const size_t n_initial   = 100000; // snapshot size — frozen index stops here
    const size_t batch_size  = 100;    // vectors added per mutation step
    const size_t measure_every = 10;   // measure recall every N batches
    const int    nprobe      = 64;     // fixed for all IVF searches

    size_t d = 0;

    // ------------------------------------------------------------------ train
    // Both IVF indexes are trained on the same learn set, so their initial
    // centroids are identical. The subsequent divergence is purely due to
    // the mutable index's perturbation updates.
    faiss::IndexFlatL2* frozen_quantizer = nullptr;
    faiss::IndexFlatL2* mutable_quantizer = nullptr;
    faiss::IndexIVFFlat* frozen_index = nullptr;
    faiss::IndexIVFMutable* mutable_index = nullptr;

    {
        printf("[%.3f s] Loading train set\n", elapsed() - t0);
        size_t nt;
        float* xt = fvecs_read("sift1M/sift_learn.fvecs", &d, &nt);

        printf("[%.3f s] Building frozen IVF4096,Flat  d=%ld\n",
               elapsed() - t0, d);
        frozen_quantizer = new faiss::IndexFlatL2(d);
        frozen_index = new faiss::IndexIVFFlat(frozen_quantizer, d, nlist);
        frozen_index->own_fields = true;   // frozen_index will delete its quantizer

        printf("[%.3f s] Building mutable IVF4096,Flat d=%ld\n",
               elapsed() - t0, d);
        mutable_quantizer = new faiss::IndexFlatL2(d);
        mutable_index = new faiss::IndexIVFMutable(mutable_quantizer, d, nlist);
        mutable_index->own_fields = true;

        printf("[%.3f s] Training both on %ld vectors\n", elapsed() - t0, nt);
        frozen_index->train(nt, xt);
        mutable_index->train(nt, xt);
        delete[] xt;
    }

    // --------------------------------------------------------------- load base
    size_t nb;
    float* xb;
    {
        size_t d2;
        xb = fvecs_read("sift1M/sift_base.fvecs", &d2, &nb);
        assert(d == d2 && "base dimension mismatch");
        printf("[%.3f s] Loaded base set: %ld vectors\n", elapsed() - t0, nb);
    }

    // ------------------------------------------- populate initial snapshot
    // All three indexes start with the same 100k vectors. After this point,
    // the frozen index stops receiving vectors; the ground-truth and mutable
    // indexes continue to receive every mutation.
    assert(n_initial <= nb && "n_initial exceeds base set size");
    printf("[%.3f s] Adding initial %ld vectors to all three indexes\n",
           elapsed() - t0, n_initial);
    frozen_index->add(n_initial, xb);
    mutable_index->add(n_initial, xb);

    frozen_index->nprobe = nprobe;
    mutable_index->nprobe = nprobe;
    printf("[%.3f s] nprobe set to %d on both IVF indexes\n",
           elapsed() - t0, nprobe);

    printf("[%.3f s] Building ground truth index on initial %ld vectors\n",
           elapsed() - t0, n_initial);
    faiss::IndexFlatL2* gt_index = new faiss::IndexFlatL2(d);
    gt_index->add(n_initial, xb);

    // ---------------------------------------------------------- load queries
    size_t nq;
    float* xq;
    {
        size_t d2;
        xq = fvecs_read("sift1M/sift_query.fvecs", &d2, &nq);
        assert(d == d2 && "query dimension mismatch");
        printf("[%.3f s] Loaded %ld queries\n", elapsed() - t0, nq);
    }

    // ---------------------------------------------------------- search buffers
    const size_t k = 100;
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

    // -------------------------------------------------- output CSV
    std::ofstream metrics_file("recall_metrics.csv");
    metrics_file << "gt_ntotal,frozen_ntotal,mutable_ntotal,"
                    "frozen_R@1,frozen_R@10,frozen_R@100,"
                    "mutable_R@1,mutable_R@10,mutable_R@100\n";

    auto log_and_print = [&](const char* tag) {
        auto [frozen_rs, mutable_rs] = measure_recalls();
        auto [fr1, fr10, fr100] = frozen_rs;
        auto [mr1, mr10, mr100] = mutable_rs;
        printf("[%.3f s] gt=%7lld  frozen=%7lld  mutable=%7lld  "
               "frozen(%.3f/%.3f/%.3f)  mutable(%.3f/%.3f/%.3f)  %s\n",
               elapsed() - t0, gt_index->ntotal,
               frozen_index->ntotal, mutable_index->ntotal,
               fr1, fr10, fr100, mr1, mr10, mr100, tag);
        metrics_file << gt_index->ntotal << ","
                     << frozen_index->ntotal << ","
                     << mutable_index->ntotal << ","
                     << fr1 << "," << fr10 << "," << fr100 << ","
                     << mr1 << "," << mr10 << "," << mr100 << "\n";
    };

    // ------------------------------------------- recall at initial snapshot
    log_and_print("[snapshot]");

    // ---------------------------------------------- mutation loop
    // Each iteration adds a batch of vectors to both the ground-truth index
    // AND the mutable index. The frozen index is deliberately left alone.
    size_t n_remaining = nb - n_initial;
    size_t n_batches   = n_remaining / batch_size;

    printf("[%.3f s] Starting mutation loop: %ld batches x %ld vectors "
           "(measuring every %ld batches)\n",
           elapsed() - t0, n_batches, batch_size, measure_every);

    for (size_t b = 0; b < n_batches; b++) {
        size_t offset = n_initial + b * batch_size;

        // mutate ground truth (source of truth)
        gt_index->add(batch_size, xb + offset * d);

        // mutate mutable IVF index — this is what your research is testing
        mutable_index->add(batch_size, xb + offset * d);

        // frozen index deliberately NOT updated

        if ((b + 1) % measure_every == 0) {
            log_and_print("");
        }
    }

    // ---------------------------------------------------- flush remainder
    size_t added_so_far = n_initial + n_batches * batch_size;
    if (added_so_far < nb) {
        size_t leftover = nb - added_so_far;
        gt_index->add(leftover, xb + added_so_far * d);
        mutable_index->add(leftover, xb + added_so_far * d);
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
    delete[] xb;
    delete[] xq;
    delete gt_index;
    delete frozen_index;    // will delete frozen_quantizer via own_fields
    delete mutable_index;   // will delete mutable_quantizer via own_fields
    return 0;
}