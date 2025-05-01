#include <sycl/sycl.hpp>
#include <iostream>
#include <vector>
#include <random>
#include <cmath>    // exp, log

using namespace sycl;

// Problem sizes
constexpr int N_SPINS      = 1024;
constexpr int N_TROTTERS   = 8;
constexpr int N_ITERATIONS = 40;

// Hamiltonian couplings
constexpr float J_trot_start = 0.01f;
constexpr float J_trot_end   = 0.2f;
constexpr float J_spin       = 0.3f;

// Cooling schedule (exponential T_k = T0 * alpha^k)
constexpr float T_start = 1.0f;
constexpr float T_end   = 1e-4f;
const    float log_alpha = std::log(T_end / T_start) / N_ITERATIONS;

// Global bias field (decays linearly to 0)
constexpr float h_global_start = 0.05f;

void print_spins(const std::vector<int>& spins, int n, int m, const char *label) {
  std::cout << label << ":\n";
  for(int t = 0; t < m; ++t) {
    std::cout << "Trotter " << t << ": ";
    for(int i = 0; i < n; ++i) {
      std::cout << (spins[i * m + t] > 0 ? '+' : '-') << ' ';
    }
    std::cout << "\n";
  }
  std::cout << "--------------------------------\n";
}

// Full energy (for monitoring)
float calc_energy(const std::vector<int>& S, int n, int m, float Jt) {
  float E = 0;
  for(int i = 0; i < n; ++i) {
    for(int t = 0; t < m; ++t) {
      int si = S[i*m + t];
      int t2 = (t + 1) % m;
      int i2 = (i + 1) % n;
      E += -Jt     * si * S[i*m + t2]
           -J_spin * si * S[i2*m + t];
    }
  }
  return E;
}

int main() {
  queue q{ default_selector_v };

  // Host RNG
  std::mt19937_64 gen(12345);
  std::uniform_real_distribution<float> dist01(0.f, 1.f);

  // Initialize spins randomly ±1
  std::vector<int> spins(N_SPINS * N_TROTTERS);
  for(auto &s : spins) 
    s = (dist01(gen) < 0.5f) ? +1 : -1;

  // Print initial
  print_spins(spins, N_SPINS, N_TROTTERS, "BEFORE");
  std::cout << "E₀ = " << calc_energy(spins, N_SPINS, N_TROTTERS, J_trot_start) << "\n";

  // SYCL buffers
  buffer<int,2>   bufS(spins.data(), range<2>(N_SPINS, N_TROTTERS));
  buffer<float,2> bufR(range<2>(N_SPINS, N_TROTTERS)); // for per-update randoms

  for(int iter = 0; iter < N_ITERATIONS; ++iter) {
    // dynamic couplings & temp & global bias
    float λ   = float(iter) / float(N_ITERATIONS);
    float Jt  = J_trot_start + λ * (J_trot_end - J_trot_start);
    float T   = T_start * std::exp(log_alpha * iter);
    float hg  = h_global_start * (1.f - λ);

    // refill random buffer
    {
      auto r = bufR.get_host_access();
      for(int i = 0; i < N_SPINS; ++i)
        for(int t = 0; t < N_TROTTERS; ++t)
          r[i][t] = dist01(gen);
    }

    // two passes (checkerboard)
    for(int parity = 0; parity < 2; ++parity) {
      q.submit([&](handler &h) {
        auto S = bufS.get_access<access::mode::read_write>(h);
        auto R = bufR.get_access<access::mode::read>(h);

        h.parallel_for(range<2>(N_SPINS, N_TROTTERS), [=](id<2> idx) {
          int i = idx[0], t = idx[1];
          if(((i + t) & 1) != parity) return;
          int si = S[i][t];

          // neighbors
          int ip = (i + 1) % N_SPINS, im = (i == 0 ? N_SPINS - 1 : i - 1);
          int tp = (t + 1) % N_TROTTERS, tm = (t == 0 ? N_TROTTERS - 1 : t - 1);

          // local field
          float hloc =
              Jt     * (S[i][tp] + S[i][tm])
            + J_spin * (S[ip][t] + S[im][t])
            + hg;

          // Glauber probability
          float p = 1.f / (1.f + sycl::exp(-2.f * hloc / T));
          S[i][t] = (R[i][t] < p ? +1 : -1);
        });
      }).wait();
    }

    // monitor every 500 iters
    if(((iter + 1) % 1) == 0) {
      std::vector<int> tmp(N_SPINS * N_TROTTERS);
      {
        auto rS = bufS.get_access<access::mode::read>();
        for(int i = 0; i < N_SPINS; ++i)
          for(int t = 0; t < N_TROTTERS; ++t)
            tmp[i*N_TROTTERS + t] = rS[i][t];
      }
      float E = calc_energy(tmp, N_SPINS, N_TROTTERS,
                    J_trot_start + float(iter+1)/N_ITERATIONS*(J_trot_end-J_trot_start));
      std::cout << "E@" << (iter+1) << " = " << E << "\n";
    }
  }

  // final readback
  std::vector<int> out(N_SPINS * N_TROTTERS);
  {
    auto rS = bufS.get_access<access::mode::read>();
    for(int i = 0; i < N_SPINS; ++i)
      for(int t = 0; t < N_TROTTERS; ++t)
        out[i*N_TROTTERS + t] = rS[i][t];
  }

  print_spins(out, N_SPINS, N_TROTTERS, "AFTER");
  std::cout << "E_final = " 
            << calc_energy(out, N_SPINS, N_TROTTERS, J_trot_end) << "\n";

  return 0;
}
