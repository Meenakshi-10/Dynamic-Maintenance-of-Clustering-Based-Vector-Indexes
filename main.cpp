/*
 * Staleness simulator for IVF index.
 * 
 * Builds a frozen IVF index on an initial snapshot, then simulates
 * dataset mutations (additions only for now) by updating a brute-force
 * ground truth index while leaving the IVF index frozen.
 * 
 * Recall is measured against dynamic ground truth, so it degrades
 * over time as the frozen index falls behind the live dataset.
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

#include <sys/stat.h>
#include <sys/time.h>

#include <faiss/AutoTune.h>
#include <faiss/IndexFlat.h>
#include <faiss/IndexIVF.h>
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
    const char* index_key    = "IVF4096,Flat";
    const size_t n_initial   = 100000; // snapshot size — index is frozen after this
    const size_t batch_size  = 100;    // vectors added to dataset per mutation step
    const size_t measure_every = 10;   // measure recall every N batches
    const int    nprobe      = 64;     // fixed for all IVF searches

    faiss::Index* index = nullptr;
    size_t d = 0;

    // ------------------------------------------------------------------ train
    {
        printf("[%.3f s] Loading train set\n", elapsed() - t0);
        size_t nt;
        float* xt = fvecs_read("sift1M/sift_learn.fvecs", &d, &nt);

        printf("[%.3f s] Preparing index \"%s\" d=%ld\n",
               elapsed() - t0, index_key, d);
        index = faiss::index_factory(d, index_key);

        printf("[%.3f s] Training on %ld vectors\n", elapsed() - t0, nt);
        index->train(nt, xt);
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

    // ------------------------------------------------- build frozen IVF index
    // The IVF index is populated with the initial snapshot and never updated
    // again. This is the "stale" index we are measuring recall degradation for.
    assert(n_initial <= nb && "n_initial exceeds base set size");
    printf("[%.3f s] Building frozen index on initial %ld vectors\n",
           elapsed() - t0, n_initial);
    index->add(n_initial, xb);

    {
        faiss::IndexIVF* ivf = dynamic_cast<faiss::IndexIVF*>(index);
        assert(ivf && "expected an IVF index");
        ivf->nprobe = nprobe;
        printf("[%.3f s] nprobe set to %d\n", elapsed() - t0, nprobe);
    }

    // ------------------------------------------------- build ground truth index
    // This is a brute-force flat index that receives every mutation.
    // It always reflects the current true state of the dataset, so searching
    // it gives us the exact nearest neighbors to compare against.
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
    // k is the number of neighbors we retrieve and evaluate recall at.
    // We use 100 to support R@1, R@10, R@100.
    const size_t k = 100;
    faiss::Index::idx_t* I    = new faiss::Index::idx_t[nq * k]; // IVF results
    float*               D    = new float[nq * k];
    faiss::Index::idx_t* I_gt = new faiss::Index::idx_t[nq * k]; // GT results
    float*               D_gt = new float[nq * k];

    // -------------------------------------------------- recall helper lambda
    // For each query, counts how many of the true top-X neighbors appear in
    // the retrieved top-X results, divides by X, then averages across queries.
    // This matches R@X = average fraction of true top-X neighbors recovered.
    auto measure_recall = [&]() -> std::tuple<float, float, float> {
        // search the frozen stale IVF index
        index->search(nq, xq, k, D, I);

        // compute current ground truth from live brute force index
        gt_index->search(nq, xq, k, D_gt, I_gt);

        float r1 = 0, r10 = 0, r100 = 0;
        for (size_t i = 0; i < nq; i++) {
            // count how many of the true top-r neighbors appear in retrieved top-r
            auto count_hits = [&](size_t r) -> int {
                int hits = 0;
                for (size_t j = 0; j < r; j++)
                    for (size_t g = 0; g < r; g++)
                        if (I[i * k + j] == I_gt[i * k + g]) hits++;
                return hits;
            };
            r1   += count_hits(1)   / float(1);
            r10  += count_hits(10)  / float(10);
            r100 += count_hits(100) / float(100);
        }
        // average across all queries
        return {r1 / nq, r10 / nq, r100 / nq};
    };

    // -------------------------------------------------- output CSV
    std::ofstream metrics_file("recall_metrics.csv");
    metrics_file << "index_ntotal,gt_ntotal,R@1,R@10,R@100\n";

    // ----------------------------------------- recall at the initial snapshot
    // At this point index and gt_index are identical, so recall should be
    // close to 1.0 (not exactly 1.0 because IVF with nprobe=64 is approximate).
    {
        auto [r1, r10, r100] = measure_recall();
        printf("[%.3f s] index=%7lld  gt=%7lld  R@1=%.4f  R@10=%.4f  R@100=%.4f  [snapshot]\n",
               elapsed() - t0, index->ntotal, gt_index->ntotal, r1, r10, r100);
        metrics_file << index->ntotal << "," << gt_index->ntotal << ","
                     << r1 << "," << r10 << "," << r100 << "\n";
    }

    // ---------------------------------------------- mutation + staleness loop
    // Each iteration adds a batch of vectors to the ground truth index only.
    // The frozen IVF index never sees these vectors, so recall degrades
    // as the dataset drifts further from the snapshot.
    size_t n_remaining = nb - n_initial;
    size_t n_batches   = n_remaining / batch_size;

    printf("[%.3f s] Starting mutation loop: %ld batches x %ld vectors "
           "(measuring every %ld batches)\n",
           elapsed() - t0, n_batches, batch_size, measure_every);

    for (size_t b = 0; b < n_batches; b++) {
        size_t offset = n_initial + b * batch_size;

        // mutate the live dataset — ground truth index tracks this
        gt_index->add(batch_size, xb + offset * d);

        // frozen index deliberately NOT updated:
        // index->add(batch_size, xb + offset * d);

        if ((b + 1) % measure_every == 0) {
            auto [r1, r10, r100] = measure_recall();
            printf("[%.3f s] index=%7lld  gt=%7lld  R@1=%.4f  R@10=%.4f  R@100=%.4f\n",
                   elapsed() - t0, index->ntotal, gt_index->ntotal, r1, r10, r100);
            metrics_file << index->ntotal << "," << gt_index->ntotal << ","
                         << r1 << "," << r10 << "," << r100 << "\n";
        }
    }

    // ---------------------------------------------------- flush remainder
    size_t added_to_gt = n_initial + n_batches * batch_size;
    if (added_to_gt < nb) {
        gt_index->add(nb - added_to_gt, xb + added_to_gt * d);
        auto [r1, r10, r100] = measure_recall();
        printf("[%.3f s] index=%7lld  gt=%7lld  R@1=%.4f  R@10=%.4f  R@100=%.4f  [final]\n",
               elapsed() - t0, index->ntotal, gt_index->ntotal, r1, r10, r100);
        metrics_file << index->ntotal << "," << gt_index->ntotal << ","
                     << r1 << "," << r10 << "," << r100 << "\n";
    }

    // ---------------------------------------------------------------- cleanup
    metrics_file.close();
    delete[] I;
    delete[] D;
    delete[] I_gt;
    delete[] D_gt;
    delete[] xb;
    delete[] xq;
    delete gt_index;
    delete index;
    return 0;
}