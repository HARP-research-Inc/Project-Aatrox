# **FPGA-Backed SBM API for Portfolio Optimization**

## **Project Overview**
This document outlines the project plan for developing an API that leverages an FPGA-accelerated **Simulated Bifurcation Machine (SBM)** to solve portfolio optimization problems efficiently.

---

## **1. Introduction**
### **1.1 Purpose**
The FPGA-backed SBM API is designed to solve large-scale **combinatorial optimization problems**, specifically **portfolio optimization**, by simulating bifurcation dynamics on FPGA hardware to achieve near-optimal solutions in real time.

### **1.2 Scope**
- **Inputs:** Financial asset data, risk constraints, optimization parameters.
- **Processing:** FPGA-accelerated SBM execution.
- **Outputs:** Optimized portfolio allocations with minimized risk and maximized returns.

---

## **2. System Architecture**
### **2.1 High-Level Design**
The API is structured as a microservices-based architecture with the following components:

1. **Frontend API**: Simple user interface for client interaction.
2. **Backend Service**: Handles request validation, pre-processing, and post-processing.
3. **FPGA Accelerator**: Executes the **Simulated Bifurcation Machine (SBM)** algorithm for solving the Quadratic Unconstrained Binary Optimization (QUBO) problem.

### **2.2 Technology Stack**
| Component          | Technology Choices  |
|--------------------|--------------------|
| API Framework     | FastAPI |
| Processing Language | Python |
| Optimization Algorithm | Simulated Bifurcation Machine (SBM) |
| Deployment | Docker |

---

## **3. Future Developments and Impacts**
### **3.1 High-Frequency Trading**
This project is built as a potential solution to optimizing portoflios and is part of a greater effort to improve the field of high frequency trading. Specifically, this project could be used to better handle the volatility and unpredictable nature of the markets. 

---

## **4. Project Timeline**
- [x] Project planning (Creating Markdown file for overall project plan)
- [ ] Portfolio Optimization Model implementation
- [ ] API development (backend: **Simulated Bifurcation Machine (SBM)** solver | frontend: Simple user interface)
- [ ] Testing and comparison of results with Stock Simulator
- [ ] Deployment with Docker

---
