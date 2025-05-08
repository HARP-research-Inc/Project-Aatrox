// QuantumAnnealer2D.cpp
// Modularized 2D Suzuki–Trotter annealer in C++/SYCL with optimized kernel synchronization

#include <sycl/sycl.hpp>
#include <vector>
#include <random>
#include <iostream>
#include <cmath>

using namespace sycl;

class QuantumAnnealer2D {
public:
    QuantumAnnealer2D(int dimX, int dimY, int numTrotters, int numIters,
                      float trotStart, float trotEnd,
                      float spinCoupling,
                      float tempStart, float tempEnd,
                      float globalBiasStart = 0.0f)
        : nx(dimX), ny(dimY), M(numTrotters), iterations(numIters),
          Jt_start(trotStart), Jt_end(trotEnd),
          J_spin(spinCoupling), T_start(tempStart), T_end(tempEnd),
          h_global0(globalBiasStart),
          totalSpins(nx * ny),
          spins(totalSpins * M),
          randomVals(totalSpins * M),
          bufSpins(range<2>(totalSpins, M)),
          bufRandom(range<2>(totalSpins, M)),
          queue(default_selector{})
    {
        initSpins();
        // Copy initial spins into buffer
        {
            auto acc = bufSpins.get_host_access();
            for(int i=0;i<totalSpins;i++)
                for(int t=0;t<M;t++)
                    acc[i][t] = spins[i*M + t];
        }
    }

    void solve(int monitorInterval = 100) {
        // Print initial energy
        float E0 = calcEnergyHost(Jt_start);
        std::cout << "Initial Energy = " << E0 << "\n";

        for(int iter = 1; iter <= iterations; ++iter) {
            float lambda = float(iter) / float(iterations);
            float Jt     = Jt_start + lambda * (Jt_end - Jt_start);
            float T      = T_start  + lambda * (T_end  - T_start);
            float hg     = h_global0 + lambda * (0.0f  - h_global0);
            float Jspin  = J_spin;

            // Refill random numbers on host
            {
                auto randHost = bufRandom.get_host_access();
                std::uniform_real_distribution<float> dist(0.f,1.f);
                for(int i = 0; i < totalSpins; ++i)
                    for(int t = 0; t < M; ++t)
                        randHost[i][t] = dist(rng);
            }

            // Copy locals to capture
            int nx_ = nx, ny_ = ny, tot_ = totalSpins, M_ = M;
            float Jt_ = Jt, T_ = T, hg_ = hg, Jspin_ = Jspin;

            // Submit both parity kernels without intermediate waits
            for(int parity = 0; parity < 2; ++parity) {
                queue.submit([&](handler &h) {
                    auto S = bufSpins.get_access<access::mode::read_write>(h);
                    auto R = bufRandom.get_access<access::mode::read>(h);
                    h.parallel_for(range<2>(tot_, M_), [=](id<2> idx) {
                        int flat = idx[0], t = idx[1];
                        if(((flat + t) & 1) != parity) return;
                        int ix = flat / ny_, iy = flat % ny_;
                        int ixm = (ix==0?nx_-1:ix-1), ixp=(ix+1)%nx_;
                        int iym = (iy==0?ny_-1:iy-1), iyp=(iy+1)%ny_;
                        int s_u = S[ix*ny_ + iyp][t], s_d = S[ix*ny_ + iym][t];
                        int s_l = S[ixm*ny_ + iy][t], s_r = S[ixp*ny_ + iy][t];
                        int t_prev = (t==0?M_-1:t-1), t_next=(t+1)%M_;
                        int s_tp = S[flat][t_prev], s_tn=S[flat][t_next];
                        float hloc = Jt_*(s_tp+s_tn) + Jspin_*(s_u+s_d+s_l+s_r) + hg_;
                        float prob = 1.f / (1.f + sycl::exp(-2.f*hloc/T_));
                        S[flat][t] = (R[flat][t] < prob ? +1 : -1);
                    });
                });
            }
            // Wait once after both parity passes
            queue.wait();

            if(iter % monitorInterval == 0 || iter == iterations) {
                float Ecur = calcEnergyHost(Jt);
                std::cout << "Energy at iteration " << iter << " = " << Ecur << "\n";
            }
        }
    }

    std::vector<int> getSolution() {
        // Copy final spins to host vector
        std::vector<int> sol(totalSpins);
        auto S = bufSpins.get_host_access();
        for(int i=0;i<totalSpins;i++){
            int sum=0;
            for(int t=0;t<M;t++) sum += S[i][t];
            sol[i] = (sum>=0?1:0);
        }
        return sol;
    }

    void printMatrix(const std::vector<int>& mat) const {
        for(int x=0;x<nx;x++){
            for(int y=0;y<ny;y++) std::cout<<mat[x*ny+y]<<' ';
            std::cout<<"\n";
        }
    }

private:
    int nx, ny, M, iterations, totalSpins;
    float Jt_start, Jt_end, J_spin, T_start, T_end, h_global0;

    std::vector<int>   spins;
    std::vector<float> randomVals;
    buffer<int,2>      bufSpins;
    buffer<float,2>    bufRandom;
    queue              queue;
    std::mt19937_64    rng{42};

    void initSpins() {
        std::uniform_int_distribution<int> dist(0,1);
        for(int i=0;i<totalSpins*M;i++)
            spins[i] = (dist(rng)? +1 : -1);
    }

    float calcEnergyHost(float Jt) {
        float E=0;
        auto S = bufSpins.get_host_access();
        for(int flat=0;flat<totalSpins;flat++){
            int ix=flat/ny, iy=flat%ny;
            int ixp=(ix+1)%nx, ixm=(ix==0?nx-1:ix-1);
            int iyp=(iy+1)%ny, iym=(iy==0?ny-1:iy-1);
            for(int t=0;t<M;t++){
                int si=S[flat][t];
                int sip=S[ixp*ny+iy][t], sim=S[ixm*ny+iy][t];
                int sjp=S[ix*ny+iyp][t], sjm=S[ix*ny+iym][t];
                int tnp=(t+1)%M;
                int stp=S[flat][tnp];
                E += -Jt*si*stp;
                E += -J_spin*si*(sip+sim+sjp+sjm);
            }
        }
        return E;
    }
};

int main(){
    int nx=32, ny=32;
    QuantumAnnealer2D annealer(nx,ny,8,200,
                                0.01f,0.2f,
                                0.3f,
                                1.0f,1e-4f,
                                0.0f);
    std::cout<<"Before:\n";
    auto initSol=annealer.getSolution();
    annealer.printMatrix(initSol);
    annealer.solve(20);
    std::cout<<"After:\n";
    auto finalSol=annealer.getSolution();
    annealer.printMatrix(finalSol);
    return 0;
}
