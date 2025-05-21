// QannealerFullConnectivity_USM.cpp
// Fully-connected Suzuki–Trotter Quantum Annealer using SYCL Unified Shared Memory
// Includes energy logging, spin-board prints, and solution outputs

#include <sycl/sycl.hpp>
#include <vector>
#include <random>
#include <iostream>
#include <cmath>

using namespace sycl;

class QuantumAnnealer {
public:
    QuantumAnnealer(int numVars, int numTrotters, int numIters,
                    float Jt_start, float Jt_end,
                    float T_start, float T_end,
                    const std::vector<float>& J_coupling,
                    const std::vector<float>& h_bias = {})
        : N(numVars), M(numTrotters), iterations(numIters),
          Jt_start(Jt_start), Jt_end(Jt_end),
          T_start(T_start), T_end(T_end),
          totalSpins(N * M),
          queue(default_selector_v)
    {
        spins      = malloc_shared<int>(totalSpins, queue);
        randomVals = malloc_shared<float>(totalSpins, queue);
        Jmat       = malloc_shared<float>(N * N, queue);
        Hb         = malloc_shared<float>(N, queue);

        // Initialize coupling matrix and biases
        for(int i = 0; i < N * N; ++i)
            Jmat[i] = J_coupling[i];
        for(int i = 0; i < N; ++i)
            Hb[i] = (h_bias.empty() ? 0.0f : h_bias[i]);

        initSpins();
    }

    ~QuantumAnnealer() {
        free(spins, queue);
        free(randomVals, queue);
        free(Jmat, queue);
        free(Hb, queue);
    }

    // Compute total energy: classical + quantum couplings
    double computeEnergy(float Jt_current) {
        double E = 0.0;
        // Classical Ising energy over all trotter slices
        for(int t = 0; t < M; ++t) {
            for(int i = 0; i < N; ++i) {
                int si = spins[i*M + t];
                // bias term
                E -= Hb[i] * si;
                // pair interactions (i<j)
                for(int j = i+1; j < N; ++j) {
                    E -= Jmat[i*N + j] * si * spins[j*M + t];
                }
            }
        }
        // Quantum coupling energy between adjacent trotter slices
        for(int t = 0; t < M; ++t) {
            int t_next = (t + 1) % M;
            for(int i = 0; i < N; ++i) {
                E -= Jt_current * spins[i*M + t] * spins[i*M + t_next];
            }
        }
        return E;
    }

    // Print the entire N x M spin matrix
    void printSpins() {
        for(int i = 0; i < N; ++i) {
            for(int t = 0; t < M; ++t)
                std::cout << (spins[i*M + t] > 0 ? "+1 " : "-1 ");
            std::cout << "\n";
        }
    }

    // Print the effective solution (majority vote over Trotter slices)
    void printSolution() {
        auto sol = getSolution();
        for(int i = 0; i < N; ++i)
            std::cout << (sol[i] > 0 ? "+1 " : "-1 ");
        std::cout << "\n";
    }

    void solve(int numSweeps) {
        std::mt19937 rng(std::random_device{}());
        std::uniform_real_distribution<float> dist01(0.0f, 1.0f);

        int n = N, m = M, total = totalSpins;
        int* spins_ptr = spins;
        float* rand_ptr = randomVals;
        float* Jmat_ptr = Jmat;
        float* Hb_ptr   = Hb;
        float Jt_s = Jt_start, Jt_e = Jt_end;
        float T_s  = T_start,  T_e  = T_end;
        int iters = iterations;

        // Log initial energy
        double E0 = computeEnergy(Jt_s);
        std::cout << "Sweep 0, Energy = " << E0 << "\n";

        for(int sweep = 1; sweep <= numSweeps; ++sweep) {
            float lambda = float(sweep) / float(iters);
            float Jt_loc = Jt_s + lambda * (Jt_e - Jt_s);
            float T_loc  = T_s  + lambda * (T_e  - T_s);
            bool hasH    = (Hb_ptr != nullptr);

            // Refill random numbers
            for(int i = 0; i < total; ++i)
                rand_ptr[i] = dist01(rng);

            queue.submit([&](handler& cgh) {
                cgh.parallel_for(range<2>(n, m), [=](item<2> id) {
                    int i = id[0], t = id[1];
                    int idx = i*m + t;
                    float local_field = 0.0f;
                    for(int j = 0; j < n; ++j)
                        local_field += Jmat_ptr[i*n + j] * spins_ptr[j*m + t];
                    int t_next = (t + 1) % m;
                    int t_prev = (t + m - 1) % m;
                    local_field += Jt_loc * (spins_ptr[i*m + t_next] + spins_ptr[i*m + t_prev]);
                    if(hasH) local_field += Hb_ptr[i];
                    int s = spins_ptr[idx];
                    float deltaE = 2.0f * s * local_field;
                    float r = rand_ptr[idx];
                    if(deltaE < 0.0f || expf(-deltaE / T_loc) > r)
                        spins_ptr[idx] = -s;
                });
            }).wait();

            // Log energy each sweep
            double E = computeEnergy(Jt_loc);
            std::cout << "Sweep " << sweep << ", Energy = " << E << "\n";
        }
    }

    std::vector<int> getSolution() {
        std::vector<int> result(N);
        for(int i = 0; i < N; ++i) {
            int sum = 0;
            for(int t = 0; t < M; ++t)
                sum += spins[i*M + t];
            result[i] = (sum >= 0 ? 1 : -1);
        }
        return result;
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
        std::uniform_int_distribution<int> distBool(0, 1);
        for(int i = 0; i < totalSpins; ++i)
            spins[i] = distBool(rng) ? 1 : -1;
    }
};

int main() {
    // Demo: fully-connected ferromagnet on 5 spins
    int N = 5, M = 8;
    std::vector<float> J(N*N);
    for(int i=0;i<N;i++) for(int j=0;j<N;j++)
        J[i*N+j] = (i==j?0.0f:1.0f);
    std::vector<float> h(N, 0.0f);

    QuantumAnnealer qa(N, M, 100,
                       0.01f, 0.2f, 1.0f, 1e-4f, J, h);

    std::cout << "Initial spin board:\n";
    qa.printSpins();
    std::cout << "Initial solution:\n";
    qa.printSolution();
    qa.solve(100);
    std::cout << "Final spin board:\n";
    qa.printSpins();
    std::cout << "Final solution:\n";
    qa.printSolution();

    return 0;
}
