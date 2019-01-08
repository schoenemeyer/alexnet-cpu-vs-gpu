# alexnet-cpu-vs-gpu
Performance of Alexnet Benchmark on CPU vs GPU


# Purpose of this lab

Evaluate the deep learning capabilities of desktop computers using NVIDIA Geforce GPU Engine. 


# Tested machines

- CPU-only: Lenovo Ideapad 1 x AMD A6-7310 APU with AMD Radeon R4 Graphics and 16GB RAM running under CENTOS7.4   
- CPU-only: Lenovo Notebook Yoga 500 - 15ISK with 1x i5-6200U CPU Linux running with Ubuntu Subsystem ( 4.4.0-17134-Microsoft #471-Microsoft Fri Dec 07 20:04:00 PST 2018 x86_64 x86_64 x86_64 GNU/Linux ); Host OS Windows 10 HOME, Build 1803.   
- CPU-only: Microsoft Surface 2 Notebook with 1x i7-8650 CPU CPU Linux running with Ubuntu Subsystem
- GPU equipped Workstation 1 x AMD FX-6300 6c with 1 x NVIDIA GTX 1050Ti running under CentOS 7.4, 16GB RAM, CUDA 9.1 and NVIDIA Driver 390.87. The GPU has 768 cores running with 1.3 GHz and comes with 4 GB GDDR5. The underlying architecture is Pascal.    

For this lab you have installed python and tensorflow correctly on the platform. For this benchmark I used these versions     
- Python 2.7.12   
- Keras 2.2.4  
- Tensorflow 1.12.0 (Intel)  and 1.10.0 (for AMD)  

If you are missing these basics, I recommend to read this very useful guide for beginners    
https://machinelearningmastery.com/setup-python-environment-machine-learning-deep-learning-anaconda/

# CPU vs GPU-Performance-Test

Assumpation: 

You followed the instructions in 

In this lab we run the Alexnet benchmark

```
cd models/tutorials/image/alexnet

python alexnet_benchmark.py
WARNING:tensorflow:From /home/thomas/anaconda2/lib/python2.7/site-packages/tensorflow/python/framework/op_def_library.py:263: colocate_with (from tensorflow.python.framework.ops) is deprecated and will be removed in a future version.
Instructions for updating:
Colocations handled automatically by placer.
conv1   [128, 56, 56, 64]
pool1   [128, 27, 27, 64]
conv2   [128, 27, 27, 192]
pool2   [128, 13, 13, 192]
conv3   [128, 13, 13, 384]
conv4   [128, 13, 13, 256]
conv5   [128, 13, 13, 256]
pool5   [128, 6, 6, 256]
2019-01-08 10:01:11.607183: I tensorflow/core/platform/cpu_feature_guard.cc:141] Your CPU supports instructions that this TensorFlow binary was not compiled to use: SSE4.1 SSE4.2 AVX FMA
2019-01-08 10:01:11.800530: I tensorflow/core/common_runtime/gpu/gpu_device.cc:1434] Found device 0 with properties: 
name: GeForce GTX 1050 Ti major: 6 minor: 1 memoryClockRate(GHz): 1.392
pciBusID: 0000:01:00.0
totalMemory: 3.94GiB freeMemory: 3.63GiB
2019-01-08 10:01:11.800621: I tensorflow/core/common_runtime/gpu/gpu_device.cc:1513] Adding visible gpu devices: 0
2019-01-08 10:01:11.801508: I tensorflow/core/common_runtime/gpu/gpu_device.cc:985] Device interconnect StreamExecutor with strength 1 edge matrix:
2019-01-08 10:01:11.801532: I tensorflow/core/common_runtime/gpu/gpu_device.cc:991]      0 
2019-01-08 10:01:11.801564: I tensorflow/core/common_runtime/gpu/gpu_device.cc:1004] 0:   N 
2019-01-08 10:01:11.801747: I tensorflow/core/common_runtime/gpu/gpu_device.cc:1116] Created TensorFlow device (/job:localhost/replica:0/task:0/device:GPU:0 with 3422 MB memory) -> physical GPU (device: 0, name: GeForce GTX 1050 Ti, pci bus id: 0000:01:00.0, compute capability: 6.1)
2019-01-08 10:01:12.587119: I tensorflow/stream_executor/platform/default/dso_loader.cc:154] successfully opened CUDA library libcudnn.so.7 locally
2019-01-08 10:01:15.496202: step 0, duration = 0.131
2019-01-08 10:01:16.811481: step 10, duration = 0.133
2019-01-08 10:01:18.122998: step 20, duration = 0.131
2019-01-08 10:01:19.435479: step 30, duration = 0.131
2019-01-08 10:01:20.746376: step 40, duration = 0.132
2019-01-08 10:01:22.055767: step 50, duration = 0.130
2019-01-08 10:01:23.373633: step 60, duration = 0.132
2019-01-08 10:01:24.684571: step 70, duration = 0.131
2019-01-08 10:01:25.996646: step 80, duration = 0.131
2019-01-08 10:01:27.310601: step 90, duration = 0.131
2019-01-08 10:01:28.494875: Forward across 100 steps, 0.131 +/- 0.001 sec / batch
2019-01-08 10:01:33.045798: step 0, duration = 0.316
2019-01-08 10:01:36.194006: step 10, duration = 0.314
2019-01-08 10:01:39.339279: step 20, duration = 0.314
2019-01-08 10:01:42.485914: step 30, duration = 0.315
2019-01-08 10:01:45.638622: step 40, duration = 0.315
2019-01-08 10:01:48.793497: step 50, duration = 0.315
2019-01-08 10:01:51.939761: step 60, duration = 0.315
2019-01-08 10:01:55.085868: step 70, duration = 0.315
2019-01-08 10:01:58.233939: step 80, duration = 0.315
2019-01-08 10:02:01.388645: step 90, duration = 0.317
2019-01-08 10:02:04.218553: Forward-backward across 100 steps, 0.315 +/- 0.001 sec / batch

```



f
