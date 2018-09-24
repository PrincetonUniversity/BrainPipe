from tensorflow.python.client import device_lib
print(device_lib.list_local_devices())


import sys

if __name__ == '__main__':

    print sys.argv
    stepid = int(sys.argv[1])    
    if dd() != '/home/wanglab/': 
        print os.environ["SLURM_ARRAY_TASK_ID"]    
        jobid = int(os.environ["SLURM_ARRAY_TASK_ID"]) 
    else:
        jobid = int(sys.argv[2]) 

    print sys.argv

    print sys.argv[3]
