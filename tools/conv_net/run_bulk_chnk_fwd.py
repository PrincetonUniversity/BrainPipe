#transfer from PNI scratch to tigress using globus https://app.globus.org/file-manager
import os
from utils.io import make_inference_output_folder
inn = "/jukebox/scratch/zmd/"
out = "/scratch/gpfs/zmd/"
logs = os.path.join(out, "logs")
if not os.path.exists(logs): os.mkdir(logs)
repo = "/tigress/zmd/3dunet"
slurm_script = "/tigress/zmd/3dunet/pytorchutils/slurm_scripts/run_chnk_fwd.sh"
fld = os.path.join(out, "slurm_scripts")
if not os.path.exists(fld): os.mkdir(fld)
print(fld)

#function to run
from subprocess import check_output
def sp_call(call):
    print(check_output(call, shell=True))
    return


#loop
paths = [xx for xx in os.listdir(out) if "chck_errs.sh" not in xx and "logs" not in xx and "slurm_scripts" not in xx]
paths = [xx for xx in paths if not "output_chnks" in os.listdir(os.path.join(out, xx))]

paths_to_call = []
for pth in paths:
    
    #read file
    with open(os.path.join(repo, slurm_script), "r") as fl:
        lines = fl.readlines()
        fl.close()
    
    #modify
    new_lines = lines[:]
    for i,line in enumerate(lines):
        
        #new dir
        if "cd pytorchutils/" in line:
            new_lines[i] = line.replace("pytorchutils/", "/tigress/zmd/3dunet/pytorchutils/")
        
        #path to folder
        if "python run_chnk_fwd.py" in line:
            new_lines[i] = line.replace("20170115_tp_bl6_lob6a_rpv_03", pth)
            
        #logs
        if "logs/chnk_" in line:
            new_lines[i] = line.replace("logs", logs)
            
    #save out
    with open(os.path.join(fld, "run_"+pth+".sh"), "w+") as fl:
        [fl.write(new_line) for new_line in new_lines]
        fl.close()
    
    #make output folder first - need to this since array jobs can run simultaneously
    make_inference_output_folder(os.path.join(out, pth))
        
    #collect
    paths_to_call.append(os.path.join(fld, "run_"+pth+".sh"))
    
#call
for pth in paths_to_call:
    call = "sbatch --array=0-100 {}".format(pth)
    print(call)
    sp_call(call)
