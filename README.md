# Running Benchmarks with Tensorflow backend on NVIDIA GeForce GPUs

In this experiment I tested the performance of Convolution Networks like Resnet and Alexnet on CPUs vs GPUs. This benchmark is also a useful test, whether the GPU on your system is running correctly.

# Tested machines

- CPU-only:   
Microsoft Surface 2 Notebook with 1x i7-8650 CPU CPU Linux running with Ubuntu Subsystem
- GPU equipped Workstation:   
1 x AMD FX-6300 6c with 1 x NVIDIA GTX 1050Ti running under CentOS 7.4, 
16GB RAM, 
CUDA Toolkit 9.1
NVIDIA Driver 390.87. 

The NVIDIA GTX 1050Ti comes with 768 cores running with 1.3 GHz and has 4 GB GDDR5. The underlying architecture is Pascal.    
For this lab you have installed python and tensorflow correctly on the platform. For this benchmark I used these versions     
- Python 2.7.12   
- Keras 2.2.4  
- Tensorflow 1.12.0 (Intel)  and 1.10.0 (for AMD)  

If you are missing these basics, I recommend to read this very useful guide for beginners    
https://machinelearningmastery.com/setup-python-environment-machine-learning-deep-learning-anaconda/

# CPU vs GPU-Performance-Test

Assumption: 

You followed the instructions in https://github.com/schoenemeyer/CPUvsGPU-Performance-Test
In short, you need your specific tensorflow version as well as the models repository from https://github.com/tensorflow/models.git

In this lab we focus on the Alexnet benchmark
https://www.tensorflow.org/guide/performance/benchmarks 

Running the Alexnet Benchmark on the GPU equipped Workstation 1 x AMD FX-6300 6c with 1 x NVIDIA GTX 1050Ti yields a much higer performance compared the i7 platform.

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
On average we see 135ms per batch, and the benchmark needs 56 sec to finish. 
The optimized i7 tensorflow version takes 2965ms per batch, the whole benchmark needs 30minutes to run!

If you want to compare differnet CNNs with tensorflow backend such as Densenet or Resnet, I recommend to clone the scripts from https://github.com/tensorflow/benchmarks

In order to see the full options, you can run:
```
python tf_cnn_benchmarks.py --help
```
An example how to run:
```
python tf_cnn_benchmarks.py --num_gpus=1 --batch_size=32 --model=resnet50 --variable_update=parameter_server
Done warm up
Step    Img/sec total_loss
1       images/sec: 52.3 +/- 0.0 (jitter = 0.0) 8.169
10      images/sec: 51.8 +/- 0.2 (jitter = 0.1) 7.593
20      images/sec: 51.8 +/- 0.1 (jitter = 0.1) 7.696
30      images/sec: 51.7 +/- 0.1 (jitter = 0.2) 7.753
40      images/sec: 51.7 +/- 0.1 (jitter = 0.2) 8.007
50      images/sec: 51.7 +/- 0.1 (jitter = 0.3) 7.520
60      images/sec: 51.7 +/- 0.1 (jitter = 0.3) 7.989
70      images/sec: 51.7 +/- 0.1 (jitter = 0.4) 8.028
80      images/sec: 51.7 +/- 0.1 (jitter = 0.4) 7.930
90      images/sec: 51.6 +/- 0.1 (jitter = 0.4) 7.853
100     images/sec: 51.7 +/- 0.1 (jitter = 0.3) 7.797
----------------------------------------------------------------
total images/sec: 51.66
----------------------------------------------------------------

```
The scripts allow to run benchmarks contain benchmarks for several convolutional neural networks.   

Precision: fp16     
Dataset: Synthetic   

### 1050 Ti  (images/sec)
| Batchsize   | InceptionV3   | ResNet-50   | VGG16   | Alexnet   | Nasnet   | 
|:--------------|:-------------|:--------------|:-----------------|:------------------|:-------------------|
| 16  |  34      |   54    | 29          | 213          | 38            | 
| 32 |  37      | 62      | 33          | 386          | 41            | 
| 64 |  39      | 65       | 30          | 466          | 44            | 
| Optimizer |  sgd      | sgd        | sgd            | sgd           | sgd            | 

<img src="https://github.com/schoenemeyer/convolution-cpu-vs-gpu/blob/master/tensorflow1-bs32.png" width="580"> <img> 


If you compare with some figures in https://www.tensorflow.org/guide/performance/benchmarks you will notice that the performance is very close to a NVIDIA Tesla K80 card. 
<img src="https://github.com/schoenemeyer/convolution-cpu-vs-gpu/blob/master/tensorflow1-bs32.png" width="580"> <img> 

