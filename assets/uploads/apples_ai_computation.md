# Apple's AI Computation approach 



There are 2 major sub-systems that perform AI computations on Apple’s platform (specifically Apple Silicon):

- The GPU
- The NPU (Apple Neural Engine, ANE)

// two columns with A19Pro for mobile chip and M5 for desktop chip
A19Pro dieshot: ( single SoC )
![image-20260902165330563](/Users/bytedance/Library/Application Support/typora-user-images/image-20260902165330563.png)
M5Pro dieshot ( disaggregated CPU tile + GPU tile )

![M5 Pro CPU die – Six Super Cores, 12 Performance Cores (in two clusters), 16-core Neural Engine, and four dedicated Thunderbolt 5 controllers](/Users/bytedance/Documents/personal-blog/assets/uploads/684c52d0-5e90-11f1-8c52-79b37c860eb4.jpg)

![M5 Pro GPU die – 20 GPU cores with per-core Neural Accelerators, 4-channel LPDDR5X-9600 PHY, Display Engines, and Media Engine, including the ProRes Engine](/Users/bytedance/Documents/personal-blog/assets/uploads/6ed28cf0-5e90-11f1-8c52-79b37c860eb4.jpg)

// AGENT: a table to compare Apple‘s GPU and NPU
GPU vs NPU overview
usage scenario: desktop vs mobile
supported model type: general model vs dedicated model
programming framework: mlx/metal vs CoreAI (formerly CoreML)
programming granularity:  instructions vs NN model/layer
custom kernels: yes vs no
power consumption: medium vs low
supported NN ops: any vs limited (// needs confirm)
supported datatypes: any(including micro-scaling) vs fp16 (int4/int8 emulated // needs confirm on newest ANE)


// AGENT: make it a table
// AGENT: fill the actual number by solid evidence and concrete reasoning
A19Pro vs M5Max (configurations vary on A-series and M-series)
GPU TFlops(fp16)
GPU cores
NPU TFlops(fp16)
NPU cores
GPU cores



## software stacks

![ML software stack](/Users/bytedance/Library/Application Support/typora-user-images/image-20260902165345614.png)

// AGENT: make this a caption of image above
CoreAI as QNN
MPS as cuda device
metal as cuda
tensorops as cuteDSL/triton

### NPU(blackbox)

NN model --> CoreAI --> compiled model with compatible layers on NPU, others on GPU/CPU

// AGENT: Add python example code for running simple PyTorch model with only Linear Module on NPU(ANE) + CoreAI

### GPU(whitebox): 
As a PyTorch backend: NN model --> PyTorch operations --> MPS device --> metal (+tensorops) --> GPU

// AGENT: add python examples on running a simple PyTorch model with only Linear Module on GPU+mlx

// AGENT: Dive deeply into how mlx implements an FP16 MatMul, using source code from  ~/workspace/mlx-src.



## references
// AGENT: Polish the reference and format
- https://arxiv.org/pdf/2606.22283v1
- mlx Github source
- metal 4.1 specification