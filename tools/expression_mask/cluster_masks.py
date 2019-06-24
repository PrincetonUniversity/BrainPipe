import numpy as np, tifffile, os, sys, re, csv
from scipy.stats import linregress
import multiprocessing as mp, pylab as pl, pandas as pd
from scipy.cluster.vq import kmeans2
from shutil import copyfile

def visualize():
    with open('/jukebox/wang/deverett/sandbox/clust.csv','r') as c:
        dw = csv.DictReader(c)
        cl = pd.DataFrame(list(dw))
    
    src = '/jukebox/wang/pisano/DREADDs/newmasking'
    cl.ix[:,'path'] = [os.path.join(src,a+'_registration_results','summary.png') for a in cl.brain]

    for cli in cl.cluster.unique():
        clu = cl[cl.cluster==cli]
        for pat,nam in zip(clu.path,clu.brain):
            clustfolder = os.path.join('/jukebox/wang/deverett/sandbox','clust_'+str(cli))
            if not os.path.exists(clustfolder):
                os.mkdir(clustfolder)
            copyfile(pat, os.path.join(clustfolder,nam+'.png'))

def cluster(n=2):
    src = '/jukebox/wang/pisano/DREADDs/newmasking'
    animal_dirs = sorted([f for f in os.listdir(src) if 'registration_results' in f])
    paths = [os.path.join(src,ad,'zyx_voxels_C1_mask.npy') for ad in animal_dirs if os.path.exists(os.path.join(src,ad,'zyx_voxels_C1_mask.npy'))]
    reg = re.compile(r'(.*)/(.*)_registration_results')
    names = [reg.match(ad).groups()[1] for ad in paths]

    print 'loading data..';sys.stdout.flush()
    data = np.array([np.array([np.ravel_multi_index(i, (110, 320, 456)) for i in np.load(f)]) for f in paths])
    dflat = np.hstack(data.flat)
    data2 = np.zeros([len(data),dflat.max()-dflat.min()+1])
    for i,d in enumerate(data):
        d -= dflat.min()
        data2[i,d] = 1

    np.savez_compressed('/jukebox/deverett/sandbox/clusterdata.npz', data=data2)

    print 'kmeans..';sys.stdout.flush()
    centroids,label = kmeans2(data2.astype(float), n)

    with open('/jukebox/wang/deverett/sandbox/clust.csv','w') as c:
        dw = csv.DictWriter(c, fieldnames=['brain','cluster'])
        dw.writeheader()
        for l,n in zip(label,names):
            dw.writerow(dict(brain=n, cluster=l))

    print centroids
    print label

if __name__ == '__main__':
    cluster(n=3)
    visualize()
