# Job Defense Shield

The software in this repo creates a report of problem users and problem jobs on the large Research Computing clusters. The software identifies the following:

+ the heaviest users with low CPU or GPU utilization  
+ multinode CPU jobs where one or more nodes have zero utilization  
+ multi-GPU jobs where one or more GPUs have zero utilization  
+ users that have been over-allocating CPU or GPU time  
+ jobs with CPU or GPU fragmentation (e.g., 1 GPU per node over 4 nodes)  
+ jobs with the most CPU-cores and jobs with the most GPUs  
+ jobs with the longest queue times  
+ jobs that request more than the default memory but do not use it  
+ users (G1 and G2) with all failed jobs on the last day they submitted  

The script does not protect against:
+ abuses of file storage or I/O  

### Notes for developers

- As partitions are added and removed the script should be updated.  
