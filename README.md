# PSCSE22-0071 (PARAMETERIZED DNN DESIGN FOR IDENTIFYING THE RESOURCE LIMITATIONS OF EDGE DEEP LEARNING HARDWARE)

## Description
This project is designed to detect resource limitation of Edge Deep Learning Devices to deploy Deep Neural Networks. Furthermore, complexity oriented prunning method using DepGraph algorithm is implemented to minimize model's resource requirement.
Measurement metrics considered for resource requirments are power usage, memory consumption (reserved and allocated memory) and inference time (latency and throughput).
Model's accuracy is evaluated on the ImageNet classfication benchmark dataset.

## Dependencies
- Python 3.x
- Jupyter Notebook
- Pytorch
- CUDA Environment
- torch-pruning

## Executing Programs
- Make sure "./model/" folder is created and save models in *.pt format.
- Make sure "./data/" folder is also created with ImageNet ILSVRC 2017 classification dataset. Please follow steps to download dataset from https://image-net.org/download.

## Profiling
- To start profiling, make sure CUDA 11.8 or above is already set up in your device. Then, ensure to install all dependencies contained in "e_p_requirments.txt".
- Command to install   `pip install requirments.txt`.
- Exectute command `python ./src/inference_profiler.py --model_path=<your model path> --batch=<your batch size> --width=<image width> --height=<image height> --log=<your log file in csv>`.

## FLOPs Oriented Pruning
- To start with complexity oriented prunning, please follow through `FLOPs Oriented Pruning.ipynb` notebook.

## Author
- Shin Thant Aung