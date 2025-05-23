// QannealerFullConnectivity_USM.cpp
// Fully-connected Suzuki–Trotter Quantum Annealer using SYCL Unified Shared Memory
// Now accepts raw linear and quadratic terms (QUBO) and handles mapping internally

#include <sycl/sycl.hpp>
#include <vector>
#include <random>
#include <iostream>
#include <cmath>

using namespace sycl;

class QuantumAnnealer {
public:
    // Constructor now takes separate linear (a_i) and quadratic (b_{ij}) QUBO terms
    QuantumAnnealer(int numVars, int numTrotters, int numIters,
                    float Jt_start, float Jt_end,
                    float T_start, float T_end,
                    const std::vector<float>& linear,
                    const std::vector<float>& quadratic)
        : N(numVars), M(numTrotters), iterations(numIters),
          Jt_start(Jt_start), Jt_end(Jt_end),
          T_start(T_start), T_end(T_end),
          totalSpins(N * M),
          queue(default_selector_v)
    {
        // Allocate USM shared memory
        spins      = malloc_shared<int>(totalSpins, queue);
        randomVals = malloc_shared<float>(totalSpins, queue);
        Jmat       = malloc_shared<float>(N * N, queue);
        Hb         = malloc_shared<float>(N, queue);

        // Map QUBO: x^T Q x = sum_i linear[i]*x_i + sum_{i<j} quadratic[i,j] * x_i x_j
        // Convert to Ising couplings Jmat and biases Hb
        // Using mapping: J_{ij} = -0.25 * b_{ij},
        //                h_i    = -0.5 * a_i - 0.25 * sum_{j != i} b_{ij}
        for(int i = 0; i < N; ++i) {
            float hi = -0.5f * linear[i];
            for(int j = 0; j < N; ++j) {
                if(i == j) {
                    Jmat[i * N + j] = 0.0f;
                } else {
                    float b = quadratic[i * N + j];
                    Jmat[i * N + j] = -0.25f * b;
                    hi -= 0.25f * b;
                }
            }
            Hb[i] = hi;
        }

        initSpins();
        std::cout << "Running on device: "
              << queue.get_device().get_info<info::device::name>() << "\n";
    }

    ~QuantumAnnealer() {
        free(spins, queue);
        free(randomVals, queue);
        free(Jmat, queue);
        free(Hb, queue);
    }

    // Compute total energy (classical + quantum)
    double computeEnergy(float Jt_val) {
        double E = 0.0;
        // Classical pairwise + bias
        for(int t = 0; t < M; ++t) {
            for(int i = 0; i < N; ++i) {
                int si = spins[i * M + t];
                E -= Hb[i] * si;
                for(int j = i + 1; j < N; ++j) {
                    E -= Jmat[i * N + j] * si * spins[j * M + t];
                }
            }
        }
        // Quantum coupling between Trotter slices
        for(int t = 0; t < M; ++t) {
            int t_next = (t + 1) % M;
            for(int i = 0; i < N; ++i) {
                E -= Jt_val * spins[i * M + t] * spins[i * M + t_next];
            }
        }
        return E;
    }

    void printSpins() {
        for(int i = 0; i < N; ++i) {
            for(int t = 0; t < M; ++t)
                std::cout << (spins[i * M + t] > 0 ? "+1 " : "-1 ");
            std::cout << "\n";
        }
    }

    void printSolution() {
        auto sol = getSolution();
        for(int v : sol)
            std::cout << (v > 0 ? "+1 " : "-1 ");
        std::cout << "\n";
    }

    void solve(int numSweeps) {
        std::mt19937 rng(std::random_device{}());
        std::uniform_real_distribution<float> dist01(0.0f, 1.0f);

        // Copy locals to avoid capturing 'this'
        int n = N, m = M, total = totalSpins;
        int* spins_ptr   = spins;
        float* rand_ptr  = randomVals;
        float* Jmat_ptr  = Jmat;
        float* Hb_ptr    = Hb;
        float Jt0 = Jt_start, Jt1 = Jt_end;
        float T0  = T_start,  T1  = T_end;
        int   L   = iterations;

        // Log initial energy
        double E0 = computeEnergy(Jt0);
        std::cout << "Sweep 0, Energy = " << E0 << "\n";

        for(int s = 1; s <= numSweeps; ++s) {
            float λ   = float(s) / float(L);
            float JtL = Jt0 + λ * (Jt1 - Jt0);
            float TL  = T0  + λ * (T1  - T0);

            // Refill random values
            for(int i = 0; i < total; ++i)
                rand_ptr[i] = dist01(rng);

            queue.submit([&](handler& cgh) {
                cgh.parallel_for(range<2>(n, m), [=](item<2> id) {
                    int i = id[0], t = id[1];
                    int idx = i * m + t;
                    float lf = 0.0f;
                    for(int j = 0; j < n; ++j)
                        lf += Jmat_ptr[i * n + j] * spins_ptr[j * m + t];
                    int t_next = (t + 1) % m;
                    int t_prev = (t + m - 1) % m;
                    lf += JtL * (spins_ptr[i * m + t_next] + spins_ptr[i * m + t_prev]);
                    lf += (Hb_ptr ? Hb_ptr[i] : 0.0f);
                    int sgn = spins_ptr[idx];
                    float dE = 2.0f * sgn * lf;
                    if(dE < 0.0f || expf(-dE / TL) > rand_ptr[idx])
                        spins_ptr[idx] = -sgn;
                });
            }).wait();

            double E = computeEnergy(JtL);
            std::cout << "Sweep " << s << ", Energy = " << E << "\n";
        }
    }

    std::vector<int> getSolution() {
        std::vector<int> res(N);
        for(int i = 0; i < N; ++i) {
            int sum = 0;
            for(int t = 0; t < M; ++t)
                sum += spins[i * M + t];
            res[i] = (sum >= 0 ? 1 : -1);
        }
        return res;
    }

private:
    int N, M, iterations;
    int totalSpins;
    float Jt_start, Jt_end, T_start, T_end;
    queue queue;
    int* spins;
    float* randomVals;
    float* Jmat;
    float* Hb;

    void initSpins() {
        std::mt19937 rng(std::random_device{}());
        std::uniform_int_distribution<int> bit(0,1);
        for(int i = 0; i < totalSpins; ++i)
            spins[i] = bit(rng) ? +1 : -1;
    }
};

int main() {
    // Max-Cut on a 5-node "house" graph
    // Nodes: 0-1-2-3 form a square, with node 4 connected to 1 and 2 as the roof apex
    int N = 5, M = 8;
    // Build QUBO for Minimize H = x^T Q x <-> Maximize Cut
    // Q[i,i] = -deg(i), Q[i,j] = 2 for each edge
    std::vector<float> a(N);
    std::vector<float> b(N*N, 0.0f);
    // Define edges of the house graph
    std::vector<std::pair<int,int>> edges = {
        {0,1}, {1,2}, {2,3}, {3,0}, // square
        {1,4}, {2,4}                // roof
    };
    // Set quadratic terms b_{ij}
    for(auto &e : edges) {
        int i = e.first, j = e.second;
        b[i*N + j] = b[j*N + i] = 2.0f;
    }
    // Compute degrees and set linear terms a_i = Q[i,i] = -deg(i)
    std::vector<int> deg(N,0);
    for(auto &e : edges) {
        deg[e.first]++;
        deg[e.second]++;
    }
    for(int i = 0; i < N; ++i) {
        a[i] = -static_cast<float>(deg[i]);
    }

    // Instantiate the annealer
    QuantumAnnealer qa(
        N, M, /*schedule*/100,
        0.01f, 0.2f, 1.0f, 1e-4f,
        a, b
    );

    std::cout << "Initial spin board:" << std::endl;
    qa.printSpins();
    std::cout << "Initial solution (spins):" << std::endl;
    qa.printSolution();
    qa.solve(100);
    std::cout << "Final spin board:" << std::endl;
    qa.printSpins();
    std::cout << "Final solution (spins):" << std::endl;
    qa.printSolution();
    return 0;
}
