# Quantum Annealer Simulation in SYCL

A flexible C++/SYCL annealer implemented with Unified Shared Memory (USM) that directly ingests explicit QUBO definitions—separate linear (a) and quadratic (b) terms—automatically maps them to an Ising model, and runs a Suzuki–Trotter Metropolis Monte Carlo annealing schedule on any SYCL device. Detailed runtime reporting (spin boards, energy profiles, and extracted solutions) is provided for transparent debugging and performance analysis.

## Prerequisites

- **Operating System**: Linux, Windows 10/11 (WSL2 recommended), or macOS
- **Compiler/SDK**: Intel oneAPI DPC++ Compiler (oneAPI Base & HPC Toolkits) or any SYCL 2020‑compatible compiler
  - Installation guide: https://software.intel.com/content/www/us/en/develop/tools/oneapi.html
- **CMake** (≥ 3.16) or a C++17‑capable build system
- **Hardware**: CPU or GPU with SYCL support (Intel® GPUs, AMD GPUs via open‑source SYCL, NVIDIA with triSYCL)

> **Note**: Even though the Intel DPC++/C++ OneAPI compiler is enough to compile for CPU, GPU, FPGA emulation, generating FPGA reports and generating RTL for FPGAs, there are extra software requirements for the FPGA simulation flow and FPGA compiles.
>
> For using the simulator flow, Intel® Quartus® Prime Pro Edition and one of the following simulators must be installed and accessible through your PATH:
> - Questa*-Intel® FPGA Edition
> - Questa*-Intel® FPGA Starter Edition
> - ModelSim® SE
>
> When using the hardware compile flow, Intel® Quartus® Prime Pro Edition must be installed and accessible through your PATH.
> **Warning** Make sure you add the device files associated with the FPGA that you are targeting to your Intel® Quartus® Prime installation.

## Set Environment Variables

When working with the command-line interface (CLI), you should configure the oneAPI toolkits using environment variables. Set up your CLI environment by sourcing the `setvars` script every time you open a new terminal window. This practice ensures that your compiler, libraries, and tools are ready for development.

> **Note**: If you have not already done so, set up your CLI
> environment by sourcing  the `setvars` script in the root of your oneAPI installation.
>
> Linux*:
> - For system wide installations: `. /opt/intel/oneapi/setvars.sh`
> - For private installations: ` . ~/intel/oneapi/setvars.sh`
> - For non-POSIX shells, like csh, use the following command: `bash -c 'source <install-dir>/setvars.sh ; exec csh'`
>
> Windows*:
> - `C:\"Program Files (x86)"\Intel\oneAPI\setvars.bat`
> - Windows PowerShell*, use the following command: `cmd.exe "/K" '"C:\Program Files (x86)\Intel\oneAPI\setvars.bat" && powershell'`
>
> For more information on configuring environment variables, see [Use the setvars Script with Linux* or macOS*](https://www.intel.com/content/www/us/en/develop/documentation/oneapi-programming-guide/top/oneapi-development-environment-setup/use-the-setvars-script-with-linux-or-macos.html) or [Use the setvars Script with Windows*](https://www.intel.com/content/www/us/en/develop/documentation/oneapi-programming-guide/top/oneapi-development-environment-setup/use-the-setvars-script-with-windows.html).

### Using Visual Studio Code* (VS Code) (Optional)

You can use Visual Studio Code* (VS Code) extensions to set your environment,
create launch configurations, and browse and download samples.

The basic steps to build and run a sample using VS Code include:
1. Configure the oneAPI environment with the extension **Environment Configurator for Intel Software Developer Tools**.
2. Download a sample using the extension **Code Sample Browser for Intel Software Developer Tools**.
3. Open a terminal in VS Code (**Terminal > New Terminal**).
4. Run the sample in the VS Code terminal using the instructions below.

To learn more about the extensions and how to configure the oneAPI environment, see the
[Using Visual Studio Code with Intel® oneAPI Toolkits User Guide](https://www.intel.com/content/www/us/en/develop/documentation/using-vs-code-with-intel-oneapi/top.html).

## Configuring and Building the System

### For Linux

1. Create a build directory:
    ```
    mkdir build
    cd build
    cmake ..
    ```
2. Building for devices:
    - CPU and GPU
    ```
    make cpu-gpu
    ```
    Optional Step:
    ```
    make clean
    ```
	- FPGA
		- Emulation
		```
		make fpga_emu
		```
		- Simulation
		```
		make fpga_sim
		```
		- Generating HTML performance reports
		```
		make report
		```
			The reports reside at `simple-add_report.prj/reports/report.html`.
		- Compiling for FPGA hardware. (may take a long time)
		```
		make fpga
		```
3. Clean the program (Optional)
	```
	make clean
	```
> **Note** When building for FPGAs, the default FPGA family will be used (Intel® Agilex® 7).
> You can change the default target by using the command :
```
cmake .. -DFPGA_DEVICE=<FPGA device family or FPGA part number>
```
> 
>Alternatively, you can target an explicit FPGA board variant and BSP by using the following command:
```
cmake .. -DFPGA_DEVICE=<board-support-package>:<board-variant>
```
> Here are a few examples of FPGA board variant BSP (this list is not exhaustive):
> 
> For Intel® PAC with Intel Arria® 10 GX FPGA, the USM is not supported, you can use below BSP:
> 
> 	intel_a10gx_pac:pac_a10
>
>For Intel® FPGA PAC D5005, use one of the following BSP based on the USM support:
>
> 	intel_s10sx_pac:pac_s10
> 	intel_s10sx_pac:pac_s10_usm
>
>You will only be able to run and executable on the FPGA if you specified a BSP. 

### For Windows

1. Create a build directory:
```
    mkdir build
    cd build
    cmake -G "NMake Makefiles" ..
```
2. Building for devices:
	- CPU and GPU
	```
	nmake cpu-gpu
	```
	- FPGA
		- Emulation
		```
		namke fpga_emu
		```
		- Simulation
		```
		nmake fpga_sim
		```
		- Generate HTML performance reports.
		```
		nmake report
		```
			The reports reside at `simple-add_report.prj/reports/report.html`.
		- Compile for FPGA hardware. (may take a long time)
		```
		nmake fpga
		```
3. Clean the program. (Optional)
```
nmake clean
```

## Running the Simulation
Running the simulation is simply finding the Aatrox file in the build folder. Usually it's `Qannealer.exe`.

### Optional CLI Flags

Currently the code is configured via compile‑time constants. To customize:
- Modify `N_SPINS`, `N_TROTTERS`, `N_ITERATIONS` in `Qannealer.cpp`
- Adjust coupling and temperature schedules in the same file
- Rebuild after changes

## Interpreting Output

- Spin Board: Displays the ±1 spin configurations for each logical variable (rows) across all Trotter slices (columns).
- Solution: Extracts the final spin s_i for each variable by majority vote over Trotter slices, shown as ±1.
- Energy Logs: Reports total energy (classical and quantum) at sweep 0 and after each sweep to illustrate convergence.
- Mapping Back to Bits: Convert spins to binary values via x_i = (s_i + 1) / 2 for cost evaluation and downstream use

These outputs enable tracing the annealing trajectory, verifying solution quality, and diagnosing performance for your QUBO model.

## Troubleshooting

- **Compiler errors**: Ensure the DPC++ compiler’s include and lib paths are set (source the oneAPI environment script)
- **Runtime errors**: Verify your hardware supports SYCL and that oneAPI environment variables are configured

## License

MIT License. See [LICENSE](LICENSE).

