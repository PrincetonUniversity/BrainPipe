#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Tue Apr 18 13:13:33 2017

@author: tpisano
"""
import os, sys, copy, shutil, numpy as np, scipy, cv2, time
from skimage.external import tifffile
from tools.utils.io import makedir, save_kwargs, listall, load_kwargs, load_dictionary
from tools.utils.directorydeterminer import pth_update
import multiprocessing as mp

def terastitcher_from_params(**params):
    """
    """
    assert params["stitchingmethod"] in ["terastitcher", "Terastitcher", "TeraStitcher"]
    kwargs = pth_update(load_kwargs(**params))
    kwargs["cores"] = params["cores"] if "cores" in params else 12
    terastitcher_wrapper(**kwargs)
    return


def terastitcher_wrapper(**kwargs):
    """Functions to handle folder consisting of files, stitch, resample, and combine using the complete pipeline. For single folders see stitch_single.py
    
    Inputs
    --------------
    src: folder of files
    dst: location to save - "fullsizedatafolder
    raw: if looking for raw data (LVBT - "raw_Datastack" in filename)
    regex: regular expression used. Must contain z,y,x,ch information (e.g. "(.*)(?P<y>\d{2})(.*)(?P<x>\d{2})(.*C+)(?P<ls>[0-9]{1,2})(.*Z+)(?P<z>[0-9]{1,4})(.*r)(?P<ch>[0-9]{1,4})(.ome.tif)")
    voxel_size (tuple) of X,Y,Z dimensions. (e.g. (1.25, 1.25, 3.0))
    tiling_overlap (flaot): percentage of overalp taken. (e.g. 0.1)
    jobid(optional)

    NOT YET IMPLEMENTED
    #multipage: True: output single multipage tiff
               False: output each XY plane as a tiff
               

    #test 3 planes
    #image_dictionary["zchanneldct"] = {xx:image_dictionary["zchanneldct"][xx] for xx in ["0450", "0451", "0452"]}
    
    
    src = "/home/wanglab/wang/zahra/troubleshooting/terastitcher/data"
    dst = "/home/wanglab/wang/zahra/troubleshooting/terastitcher/tpout"
    """
    #handle inputs:
    dst = kwargs["dst"] if "dst" in kwargs else False
    voxel_size = kwargs["xyz_scale"] if "xyz_scale" in kwargs else (1.63, 1.63, 7.5)
    percent_overlap = kwargs["tiling_overlap"] if "tiling_overlap" in kwargs else 0.1
    threshold = kwargs["threshold"] if "threshold" in kwargs else 0.7
    algorithm = kwargs["algorithm"] if "algorithm" in kwargs else "MIPNCC"
    transfertype = kwargs["transfertype"] if "transfertype" in kwargs else "copy"#"move"#"copy" #"move"
    outbitdepth = kwargs["outbitdepth"] if "outbitdepth" in kwargs else 16
    cores = kwargs["cores"] if "cores" in kwargs else 12
    cleanup = kwargs["cleanup"] if "cleanup" in kwargs else True
    resizefactor = kwargs["resizefactor"] if "resizefactor" in kwargs else 6
    jobid = kwargs["jobid"] if "jobid" in kwargs else False
    print ("Jobid={}, transfertype={}".format(jobid, transfertype))
    print("\nalgorithm = {}".format(algorithm))
    print("\nthreshold = {}".format(threshold))
    print("\nvoxel size = {}\n".format(voxel_size))
    #     
    image_dictionary=copy.deepcopy(kwargs)
    dst = os.path.join(kwargs["outputdirectory"], "full_sizedatafld")
    makedir(dst)

    #determine jobs:
    jobdct=make_jobs(image_dictionary, jobid = jobid)
    dct = {"transfertype": transfertype, "scalefactor":voxel_size, "percent_overlap":percent_overlap, 
           "threshold":threshold, "dst":dst, "algorithm":algorithm, "outbitdepth": outbitdepth, "transfertype":transfertype, 
           "cleanup":cleanup}
    [inndct.update(dct) for k,inndct in jobdct.items()]
    
    #Terastitcher
    if cores>=2:
        #parallezation
        iterlst = [copy.deepcopy(inndct) for inndct in list(jobdct.values())]
        p = mp.Pool(cores)
        outlst = p.map(terastitcher_par, iterlst)
        p.terminate()

    else:
        outlst = [terastitcher_par(copy.deepcopy(inndct)) for inndct in list(jobdct.values())]
    
    #collapse        
    outdct = {xx[0]:[] for xx in outlst}; [outdct[xx[0]].append(xx[1]) for xx in outlst] #{final_dst, [ts_out(lls), ts_out(rls)]}
    
    #blend lighsheets
    sys.stdout.write("\n\nBlending lightsheets and moving to final dst..."); sys.stdout.flush()
    for dst, flds in outdct.items():
        if len(flds) ==2: blend_lightsheets(flds, dst, cores)
        if len(flds) ==1: blend_lightsheets([flds[0], flds[0]], dst, cores) ##simulating two
    sys.stdout.write("...completed"); sys.stdout.flush()
    
    #downsize
    sys.stdout.write("\n\nDownsizing images..."); sys.stdout.flush()
    resize(os.path.dirname(list(outdct.keys())[0]), resizefactor, cores)
    sys.stdout.write("completed :]"); sys.stdout.flush()    
    return 

def make_jobs(image_dictionary, jobid=False):
    """
    Simple function to create job dct for parallelization
    """
    jobdct={}
    lslst = ["left_lightsheet", "right_lightsheet"]
    if type(jobid)==int: volumes = [image_dictionary["volumes"][jobid]]
    elif not jobid: volumes = image_dictionary["volumes"]
    
    #loop
    for volume in volumes:
        for lightsheet in range(volume.lightsheets):
            
            zdct = copy.deepcopy(volume.zdct)
            
            if volume.lightsheets == 2:
                side=["_C00_", "_C01_"][lightsheet] 
                ls = lslst[lightsheet]
                zdct={k:{kk:[xx for xx in vv if side in xx]} for k,v in zdct.items() for kk,vv in v.items()}
                
            if volume.lightsheets==1:
                side="_C00_"
                ls = lslst[0]
                zdct={k:{kk:[xx for xx in vv if side in xx]} for k,v in zdct.items() for kk,vv in v.items()}
                
            dct={"lightsheets":volume.lightsheets, "xtile": volume.xtile, "ytile": volume.ytile, "horizontalfoci":volume.horizontalfoci,
                 "fullsizedimensions":volume.fullsizedimensions, "channels":[xx.channel for xx in image_dictionary["volumes"]],
                 "zchanneldct":zdct}
            
            name = os.path.basename(volume.full_sizedatafld_vol)
            final_dst = volume.full_sizedatafld_vol; makedir(final_dst)
            tmp_dst = os.path.join(os.path.dirname(list(list(volume.zdct.values())[0].values())[0][0]), 
                                   os.path.basename(volume.full_sizedatafld_vol)+"_"+ls); makedir(os.path.dirname(tmp_dst)) 
            makedir(tmp_dst)
            ts_out = os.path.join(os.path.dirname(list(list(volume.zdct.values())[0].values())[0][0]), 
                                  os.path.basename(volume.full_sizedatafld_vol)+"_"+ls+"_ts_out"); makedir(ts_out)
            
            jobdct["{}_lightsheet{}".format(os.path.basename(volume.full_sizedatafld_vol), 
                   lightsheet)] = copy.deepcopy({"job":"{}_lightsheet{}".format(os.path.basename(volume.full_sizedatafld_vol), lightsheet),
            "name": name, "ts_out": ts_out, "lightsheet": ls, "channel": volume.channel, "dst":volume.full_sizedatafld, 
            "dct": copy.deepcopy(dct), "final_dst":final_dst, "tmp_dst":tmp_dst})
            
    return jobdct

def terastitcher_par(inndct):
    """Parallelize terastitcher using dct made by make_jobs function
    """

    image_dictionary = inndct["dct"]; tmp_dst = inndct["tmp_dst"]; job=inndct["job"]; channel = inndct["channel"]; 
    lightsheet = inndct["lightsheet"]; name = inndct["name"]; transfertype = inndct["transfertype"]
    voxel_size = inndct["scalefactor"]; percent_overlap = inndct["percent_overlap"]; dst=inndct["dst"]; 
    algorithm = inndct["algorithm"]; outbitdepth=inndct["outbitdepth"]; threshold=inndct["threshold"]
    cores = 1
    
    #format data
    make_folder_heirarchy(image_dictionary, dst=tmp_dst, channel=channel, lightsheet=lightsheet, final_dst=inndct["final_dst"], 
                          transfertype=transfertype, cores=cores, scalefactor=voxel_size, percent_overlap=percent_overlap)    
        
    #stitch
    call_terastitcher(src=tmp_dst, dst=inndct["ts_out"], voxel_size=voxel_size, threshold=threshold, 
                      algorithm = algorithm, outbitdepth = outbitdepth, resolutions="0") #
    
    return [inndct["final_dst"], inndct["ts_out"]] #final dst, ts_out


def call_terastitcher(src, dst, voxel_size=(1,1,1), threshold=0.7, algorithm = "MIPNCC", outbitdepth = "16", resolutions="0"):
    """
    Wrapper to use Terastitcher: https://doi.org/10.1186/1471-2105-13-316
    NOTE: terastitcher needs to be compiled using cmake and its path must be made global
        (usually done in bashrc): export PATH="/home/wanglab/TeraStitcher/src/bin:$PATH"

    
    Inputs
    -------------
    src = location of folder heirarchically formatted. See: make_folder_heirarchy
    dst = location to output to
    voxel_size = XYZ um/pixel
    threshold = https://github.com/abria/TeraStitcher/wiki/User-Interface#--thresholdreal
    algorithm = https://github.com/abria/TeraStitcher/wiki/User-Interface#--algorithmstring-advanced
    outbitdepth = 8, 16, .... https://github.com/abria/TeraStitcher/wiki/User-Interface#--imout_depthstring
    resolutions = 0, 01, 012, ... https://github.com/abria/TeraStitcher/wiki/User-Interface#--resolutionsstring
    
    Returns:
    folder location
    
    command line example (inpsired by: https://github.com/abria/TeraStitcher/wiki/Demo-and-batch-scripts)
    terastitcher --import --volin=/home/wanglab/LightSheetTransfer/test_stitch/00 --volin_plugin="TiledXY|3Dseries" --imin_plugin="tiff3D" --imout_plugin="tiff3D" --ref1=1 --ref2=2 --ref3=3 --vxl1=1 --vxl2=1 --vxl3=1 --projout=xml_import
    terastitcher --displcompute --projin="/home/wanglab/LightSheetTransfer/test_stitch/00/xml_import.xml"
    terastitcher --displproj --projin="/home/wanglab/LightSheetTransfer/test_stitch/00/xml_import.xml"
    terastitcher --displthres --projin="/home/wanglab/LightSheetTransfer/test_stitch/00/xml_displproj.xml" --projout="/home/wanglab/LightSheetTransfer/test_stitch/00/xml_displthres.xml" --threshold=0.7
    terastitcher --placetiles --projin="/home/wanglab/LightSheetTransfer/test_stitch/00/xml_displthres.xml" --projout="/home/wanglab/LightSheetTransfer/test_stitch/00/xml_placetiles.xml" --algorithm=MIPNCC
    terastitcher --merge --projin="/home/wanglab/LightSheetTransfer/test_stitch/00/xml_placetiles.xml" --volout="/home/wanglab/LightSheetTransfer/test_stitch/00" --imout_depth=16 --resolutions=012345
    
    """
    st = time.time()
    #import
    voxel_size = tuple(map(lambda x: isinstance(x, float) and round(x, 2) or x, voxel_size)) #make sure its rounded (fix for 1.1x images)
    
    sys.stdout.write("\n\nRunning Terastitcher import on {}....".format(" ".join(src.split("/")[-2:]))); sys.stdout.flush()
    xml_import = os.path.join(src, "xml_import.xml")
    call0 = "terastitcher -1 --volin={} --volin_plugin='TiledXY|3Dseries' --imin_plugin='tiff3D' --imout_plugin='tiff3D' --ref1=1 --ref2=2 --ref3=3 --vxl1={} --vxl2={} --vxl3={} --projout={}".format(src, voxel_size[0],voxel_size[1], voxel_size[2], xml_import)
    sp_call(call0)
    sys.stdout.write("\n...completed!"); sys.stdout.flush()
    
    #align
    sys.stdout.write("\n\nRunning Terastitcher alignment on {}, this can take some time....".format(" ".join(src.split("/")[-2:]))); sys.stdout.flush()
    xml_displcomp = os.path.join(src, "xml_displcomp.xml")
    call1 = "terastitcher --displcompute --projin={} --projout={}".format(xml_import, xml_displcomp)
    sp_call(call1)
    sys.stdout.write("\n...completed!"); sys.stdout.flush()
    
    #projection
    xml_displproj = os.path.join(src, "xml_displproj.xml")
    call2 = "terastitcher --displproj --projin={}".format(xml_import)
    sp_call(call2)
    
    #threshold
    xml_displthresh = os.path.join(src, "xml_displthresh.xml")
    call3 = "terastitcher --displthres --projin={} --projout={} --threshold={}".format(xml_displproj, xml_displthresh, threshold)
    sp_call(call3)
    
    #placetiles
    xml_placetiles = os.path.join(src, "xml_placetiles.xml")
    call4 = "terastitcher --placetiles --projin={} --projout={} --algorithm={}".format(xml_displthresh, xml_placetiles, algorithm)
    sp_call(call4)
    
    #merge
    sys.stdout.write("\nOutputting images, this can also take some time...."); sys.stdout.flush()
    makedir(dst)
    call5 = "terastitcher --merge --projin={} --volout={} --imout_depth={} --resolutions={}".format(xml_placetiles, dst, outbitdepth, resolutions)
    sp_call(call5)
    sys.stdout.write("\n...completed! :] in {} minutes.\n".format(np.round((time.time() - st) / 60), decimals=2)); sys.stdout.flush()   
    
    return 

def sp_call(call):

    from subprocess import check_output
    print(check_output(call, shell=True))
    
    return 

    
     
def make_folder_heirarchy(image_dictionary, dst, channel, lightsheet=False, 
                          final_dst=False, transfertype="move", scalefactor=(1.63, 1.63, 7.5), percent_overlap=0.1, cores=False, **kwargs):
    """Function to make folders for compatibility with Terastitcher
    
    Inputs:
    --------------
    image_dictionary: dctionary generated from make_image_dictionary
    dst: place to make folder structure somewhere else
    transfertype (optional): "move": move files from current location to dst
                             "copy": copy files from current location to dst
                                     
    Returns:
    -----------------
    paths to each channel folder
    """    
    #inputs
    st = time.time()
    makedir(dst)
    lightsheet = lightsheet if lightsheet else str(np.random.randint(1000,9999))
    
    #image dims
    im = tifffile.imread(pth_update(list(list(image_dictionary["zchanneldct"].values())[0].values())[0][0]), multifile=False)
    if im.ndim == 3:
        sys.stdout.write("flattening (maxip) multiplane tiffs"); sys.stdout.flush()
        hf, ypx, xpx = im.shape
        iterlst = [zz for xx in list(image_dictionary["zchanneldct"].values()) for yy in list(xx.values()) for zz in yy]
        if cores>=2:
            p = mp.Pool(cores)
            p.map(max_ip, iterlst)    
            p.terminate()
        else:
            [max_ip(fl) for fl in iterlst]
        sys.stdout.write("completed flattening (maxip) multiplane tiffs"); sys.stdout.flush()
        
    elif im.ndim == 2: ypx,xpx = im.shape
    
    #factor in percent overlap
    ypx = ypx * (1-percent_overlap)
    xpx = xpx * (1-percent_overlap)
    
    #tiles
    xtile = image_dictionary["xtile"]
    ytile = image_dictionary["ytile"]
    
    sys.stdout.write("\nMaking Folders,"); sys.stdout.flush()    

    iterlst = []

    index = -1
    for y in range(ytile):
        ynm = str(int(ypx*y*scalefactor[1])*10).zfill(6)
        ydst = dst+"/"+ynm; makedir(ydst)
        for x in range(xtile):
            xnm = str(int(xpx*x*scalefactor[0])*10).zfill(6)
            xdst = ydst+"/"+ynm+"_"+xnm; makedir(xdst)
            index+=1
            for z in image_dictionary["zchanneldct"]:
                znm = str(int(int(z)*scalefactor[2])*10).zfill(6)
                #lst is all XY tiles of the appropriate z plane and channel
                lst = image_dictionary["zchanneldct"][str(z).zfill(4)][channel]; lst.sort()
                iterlst.append((lst[index], xdst+"/"+ynm+"_"+xnm+"_"+znm+".tif", transfertype))
                
    #generate backup just in case
    if final_dst:
        kwargs["terastitcher_dct"] = {xx[0]:xx[1] for xx in iterlst}
        save_kwargs(os.path.join(final_dst, "terastitcher_dct_{}.p").format(lightsheet), **kwargs)
                    
    #move/copy files
    if cores>=2:
        sys.stdout.write(" populating folders: {} files using {} cores...\n".format(len(iterlst), cores)); sys.stdout.flush()
        p=mp.Pool(cores)
        p.starmap(make_folder_heirarchy_helper, iterlst)
        p.terminate()
        
    else:
        sys.stdout.write(" populating folders..."); sys.stdout.flush()
        [make_folder_heirarchy_helper(i[0], i[1], i[2]) for i in iterlst]          
        
    sys.stdout.write("\n...finished in {} minutes.\n".format(np.round((time.time() - st) / 60), decimals=2)); sys.stdout.flush()        
    return
    
def make_folder_heirarchy_helper(src, dst, transfertype):
    """
    """
    import shutil
    if transfertype == "move" and not os.path.exists(dst): shutil.move(src, dst)
    elif transfertype == "copy" and not os.path.exists(dst): shutil.copy(src, dst)
    
    return


def max_ip(src):
    im = tifffile.imread(src)
    dtype = im.dtype
    if im.ndim == 3: im = np.max(im, 0)
    tifffile.imsave(src, im.astype(dtype))
    return


def blend_lightsheets(flds, dst, cores, cleanup=False):
    """0=L, 1=R
    """
    #make sure l and r are in appropriate positions
    if np.any(["right_lightsheet" in xx for xx in flds]) and np.any(["left_lightsheet" in xx for xx in flds]):
        l = [xx for xx in flds if "left_lightsheet" in xx][0]
        r = [xx for xx in flds if "right_lightsheet" in xx][0]
        flds = [l,r]
    #
    st = time.time()
    name = os.path.basename(dst); ch = dst[-2:]
    sys.stdout.write("\nStarting blending of {}...".format(dst)); sys.stdout.flush()
    ydim, xdim =tifffile.imread(listall(flds[0], keyword=".tif")[0]).shape
    #alpha=np.tile(scipy.stats.logistic.cdf(np.linspace(-250, 250, num=xdim)), (ydim, 1))
    fls0 = listall(flds[0], keyword=".tif"); fls0.sort()
    fls1 = listall(flds[1], keyword=".tif"); fls1.sort()
    assert set([os.path.basename(xx) for xx in fls0]) == set([os.path.basename(xx) for xx in fls1]), "uneven number of z planes between L and R lightsheets"
    makedir(dst);#makedir(os.path.join(dst, name))
    iterlst=[{"xdim":xdim, "ydim":ydim, "fl0":fls0[i], "channel": ch, "fl1":fls1[i], "dst":dst, "name":name, "zplane":i} for i,fl0 in enumerate(fls0)]
    if cores>=2:
        p=mp.Pool(cores)
        p.map(blend, iterlst)
        p.terminate()
    else:
        [blend(dct) for dct in iterlst]
    
    #if cleanup: [shutil.rmtree(xx) for xx in flds]
    sys.stdout.write("\n...finished in {} minutes.\n".format(np.round((time.time() - st) / 60), decimals=2)); sys.stdout.flush()        
    return

def blend(dct):
    """0=L, 1=R"""
    fl0 = dct["fl0"]; fl1 = dct["fl1"]
    alpha=np.tile(scipy.stats.logistic.cdf(np.linspace(-20, 20, num=dct["xdim"])), (dct["ydim"], 1)) #linspace was -275,275
    im0 = tifffile.imread(fl0); dtype = im0.dtype; im1 = tifffile.imread(fl1)
    ch = "_C"+dct["channel"]
    im0, im1 = pixel_shift(im0, im1)
    #tifffile.imsave(os.path.join(dct["dst"], dct["name"], dct["name"]+ch+"_Z"+str(dct["zplane"]).zfill(4)+".tif"), (im0*(1-alpha) + im1* (alpha)).astype(dtype), compress=1)
    tifffile.imsave(os.path.join(dct["dst"], dct["name"]+ch+"_Z"+str(dct["zplane"]).zfill(4)+".tif"), (im0*(1-alpha) + im1* (alpha)).astype(dtype), compress=1)
    return

def pixel_shift(a,b):
    """Function to shift the higher of two images intensity towards the lower ones pixel intensity based on median values (more robust to extremes)
    """
    a = tifffile.imread(a) if type(a) == str else a
    b = tifffile.imread(b) if type(b) == str else b
    dtype = a.dtype
    
    am = np.median(a); bm = np.median(b)
    delta = float(np.abs(am - bm))
    
    if am>bm: a = (a - delta).clip(min=0).astype(dtype)
    if bm>am: b = (b - delta).clip(min=0).astype(dtype)
    return a,b

def resize(src, resizefactor, cores):
    """src = fullsizedata_fld
    """
    #find files
    fls = listall(src, keyword=".tif")
    
    #calc resize
    y,x = tifffile.imread(fls[0], multifile=False).shape
    yr = int(y/resizefactor); xr = int(x/resizefactor)
    
    #set up dsts
    [makedir(os.path.join(os.path.dirname(src), xx[:-4]+"resized_"+xx[-4:])) for xx in os.listdir(src) if ".txt" not in xx]
    
    #parallelize
    iterlst = [copy.deepcopy({"fl":fl, "xr":xr, "yr":yr}) for fl in fls]
    p = mp.Pool(cores)
    p.map(resize_helper, iterlst)
    p.terminate()
    
    return
    
    
def resize_helper(dct):
    """
    """
    src =dct["fl"]; xr = dct["xr"]; yr=dct["yr"]
    xx = os.path.basename(src)
    nm = os.path.join("/".join(src.split("/")[:-3]), xx[:-18]+"resized_ch"+xx[-12:-10])
    dst = os.path.join(nm, xx)
    im = tifffile.imread(src)
    im = cv2.resize(im, (xr, yr), interpolation=cv2.INTER_LINEAR)
    tifffile.imsave(dst, im.astype("uint16"), compress=1)
    return

def multiple_original_directory_structure(src, transfertype="move"):
    """
    src = "/home/wanglab/wang/pisano/tracing_output/retro_4x/20180312_jg_bl6f_prv_17/full_sizedatafld"
    """
    fls = listall(src, keyword = "terastitcher_dct")
    for fl in fls:
        print(fl)
        original_directory_structure(fl, transfertype=transfertype)
    return

def original_directory_structure(src, transfertype="move"):
    """
    Move files back
    src = "/home/wanglab/wang/pisano/ymaze/cfos_4x/20171129_ymaze_23/full_sizedatafld/20171129_ymaze_23_488_050na_z7d5um_50msec_10povlap_ch00/terastitcher_dct.p"
    
    """
    for k,v in pth_update(load_dictionary(src)["terastitcher_dct"]).items():
        try:
            if transfertype == "copy": shutil.copy(pth_update(v),pth_update(k))
            if transfertype == "move": shutil.move(pth_update(v),pth_update(k))
        except:
            print("?")
        
    return


