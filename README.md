<p align="center">
  <img src = "https://github.com/AD2605/cuPycl/blob/main/cupyclLogo.png">
</p>


# cuPycl
cuPycl (Pronounced cu-pickle) is a drop in replacement for the PIL Library but with SIMD and GPU Support. <br>
The Support extends for Nvidia using **Cu**da, AMD and other accelerators using S**YCL** and SIMD and parallel execution for x86 SIMD instructions and even for ARM Neon Architecture. 


## Usage
Exactly how one would use PIL, but with an additional parameter called the Device Parameter. This parameter can be as - `Device.Vanilla`, `Device.Accelerated`, `Device.Nvidia` and `Device.SYCL`

### Development Stage
Currently, this library is in it's very nascent stage and I have just started to write the kernels and add SIMD support. A lot of work has to go in it, namely writing a sophesticated dispatcher, CI/CD pipelines(Towards the end) and Cmake and the installation script


