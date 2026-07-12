# BRBtcHuntAMD-OpenCL

Porte do [BRBtcHuntAMD](https://github.com/jmr2704/BRBtcHuntAMD) (HIP/ROCm) para **OpenCL C 2.0**, otimizado para **GPUs AMD RDNA1** (RX 5700 XT, RX 5600 XT, RX 5500 XT) rodando em **Windows** com driver AMD Adrenalin.
## Pré-requisitos

### Windows
- **Visual Studio 2022** com workload "Desktop development with C++"
- **CMake 3.16+** — `choco install cmake`
- **AMD APP SDK 3.0** (para headers OpenCL) — baixar de https://developer.amd.com/amd-accelerated-parallel-processing-app-sdk/
  - Ou: `choco install opencl-headers` (alternativa)
- **OpenSSL** (para validação de endereços P2PKH)
  - Via vcpkg: `vcpkg install openssl:x64-windows`
  - Ou: `choco install openssl`
- **Driver AMD Adrenalin 23+** (já vem com runtime OpenCL)

### Linux (testes/CI)
- `apt install opencl-headers ocl-icd-opencl-dev libssl-dev cmake g++`

## Build

### Windows (PowerShell)
```powershell
cd BRBtcHuntAMD-OpenCL
powershell -ExecutionPolicy Bypass -File scripts\build-windows.ps1
```

Saída: `build\BRBtcHuntAMD-OpenCL.exe` + `build\kernels\*.cl`

### Linux / Manual
```bash
mkdir build && cd build
cmake .. -DCMAKE_BUILD_TYPE=Release
make -j$(nproc)
```

## Uso

```bash
# Busca por endereço (P2PKH)
BRBtcHuntAMD-OpenCL.exe --range 200000000:3FFFFFFFF \
                         --address 1HBtApAwR7JgqgzqERiA6R5T5o5mYh1k3j \
                         --grid 128,256 --slices 64

# Busca por hash160 raw
BRBtcHuntAMD-OpenCL.exe --range 200000000:3FFFFFFFF \
                         --target-hash160 79fbfc3e62c4d3e9b6d1c1c5d5d8c5e9a6b3c2d1 \
                         --gpus 0,1 --random --slices 16

# Modo vanity (salva chaves com N hex chars iniciais do hash160)
BRBtcHuntAMD-OpenCL.exe --range 1:FFFFFFFF \
                         --target-hash160 0000000000000000000000000000000000000000 \
                         --vanity 4 --random

# Listar GPUs detectadas
BRBtcHuntAMD-OpenCL.exe --help
```
