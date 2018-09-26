import u
import imp 

from sys import argv

#Quick script for testing that your samples look
# the way you expect them to

SAMPLER_FNAME = argv[1]
NUM_SAMPLES   = int(argv[2]) if len(argv) > 2 else 3


sampler = imp.load_source("S",SAMPLER_FNAME)

s = sampler.Sampler("~/research/datasets/CSHL_GAD/", dsets=["train"],
                    mode="train", patchsz=(18,160,160))

for i in range(NUM_SAMPLES):

    samp = s(imgs=["input"])


    for (k,v) in samp.items():
        for j in range(v.shape[0]):
          u.write_file(v[j,:,:,:], "tests/sample{}_{}{}.h5".format(i,k,j))

