# -*- coding: utf-8 -*-
"""
Created on Wed Feb 24 22:07:30 2016

@author: tpisano

"""

import os, sys, cv2, time, re, warnings, shutil, mmap, collections, random, numpy as np, SimpleITK as sitk, multiprocessing as mp
import scipy.stats, pickle, scipy.ndimage
from skimage.external import tifffile
from tools.imageprocessing.orientation import fix_orientation
from tools.utils.io import listdirfull, makedir, removedir, chunkit, writer, load_kwargs, load_tif_list, save_kwargs

class volume:
    """Class to represent a volume for a particular channel
    z,y,x is the location of the determined center ###FOLLOWING NP CONVENTION
    """
    kind="volume"
    def __init__(self, ch_type):
        self.ch_type=ch_type       ##type of volume: "regch", injch, cellch
        self.channel=()
        self.outdr=()          #location of output directory
        self.indr=()
        self.raw=False
        self.regex=()
        self.packagedirectory=()
        self.parameterfolder=()
        self.zdct={}                        #zdct containing the plains for
        self.plns=[]                        #PLANES POST STITCHING
        self.downsized_vol=()               #location(s)*** for downsized folder or file, downsized 2d cell file will store here
        self.resampled_for_elastix_vol=()    #location for resample_for_elastix_file
        self.brainname=()
        self.injdetect3dfld=()
        self.allen_id_table=()
        self.full_sizedatafld=()            #path to fullsizedatafld
        self.full_sizedatafld_vol=()        #path to volume"s folder in fullsizedatafld
        self.cell_folder=()
        self.injcoordinatesfld=()
        self.fullsizedimensions=()
        self.horizontalfoci=()
        self.lightsheets=()
        self.ytile=()
        self.xtile=()
        self.tiling_overlap=()
        self.xyz_scale=()
        self.registration_volume=()         #pth to result file from elastix
        self.atlasfile=()                   #atlasfile to use for registration
        self.elastixfld=()
        self.inverse_elastixfld=()
        self.reg_to_atlas_vol=()
        self.reg_to_atlas_transform=()
        self.ch_to_reg_to_atlas=()


    def add_channel(self, ch):
        self.channel=ch
    def add_raw(self, raw):
        self.raw=raw
    def add_outdr(self, dr):
        self.outdr=dr
    def add_indr(self, dr):
        self.indr=dr
    def add_regex(self, expr):
        self.regex=expr
    def add_packagedirectory(self, packagedirectory):
        self.packagedirectory=packagedirectory
    def add_parameterfolder(self, parfolder):
        self.parameterfolder=parfolder
    def add_zdct(self, zdct):
        self.zdct=zdct
    def update_zdct(self, new_zdct):
        zdct=self.zdct
        zdct.update(new_zdct)
    def add_brainname(self, nm):
        self.brainname=nm
    def add_injdetect3dfld(self, loc):
        self.injdetect3dfld=loc
    def add_allen_id_table(self, loc):
        self.allen_id_table=loc
    def add_full_sizedatafld(self, loc):            #path to fullsizedatafld
        self.full_sizedatafld=loc
    def add_full_sizedatafld_vol(self, loc):       #path to volume"s folder in fullsizedatafld
        self.full_sizedatafld_vol=loc
    def add_injcoordinatesfld(self, loc):
        self.injcoordinatesfld=loc
    def add_fullsizedimensions(self, dim):
        self.fullsizedimensions=dim
    def add_horizontalfoci(self, num):
        self.horizontalfoci=num
    def add_lightsheets(self, num):
        self.lightsheets=num
    def add_ytile(self, num):
        self.ytile=num
    def add_xtile(self, num):
        self.xtile=num
    def add_tiling_overlap(self, num):
        self.tiling_overlap=num
    def add_xyz_scale(self, num):
        self.xyz_scale=num
    def update_plns(self, loc):
        self.plns.append(loc)
    def add_downsized_vol(self, loc):      #location(s)*** for downsized folder or file, downsized 2d cell file will store here
        self.downsized_vol=loc
    def add_resampled_for_elastix_vol(self, loc): #location for resample_for_elastix_file
        self.resampled_for_elastix_vol=loc
    def add_registration_volume(self, loc):         #pth to result file from elastix
        self.registration_volume=loc
    def add_atlasfile(self, loc):               #atlasfile to use for registration
        self.atlasfile=loc
    def add_elastixfld(self, loc):
        self.elastixfld=loc
    def add_inverse_elastixfld(self, loc):
        self.inverse_elastixfld=loc
    def add_reg_to_atlas_vol(self, loc):
        self.reg_to_atlas_vol=loc
    def add_reg_to_atlas_transform(self, loc):
        self.reg_to_atlas_transform=loc
    def add_ch_to_reg_to_atlas(self, loc):
        self.ch_to_reg_to_atlas=loc

def regex_determiner(raw, dr):
    """helper function to determine appropriate regular expression

    """
    lst=[r"(.*)(?P<y>\d{2})(.*)(?P<x>\d{2})(.*C+)(?P<ch>[0-9]{1,2})(.*Z+)(?P<z>[0-9]{1,4})(.ome.tif)", ###lavision processed
    r"(.*)(?P<y>\d{2})(.*)(?P<x>\d{2})(.*C+)(?P<ls>[0-9]{1,2})(.*Z+)(?P<z>[0-9]{1,4})(.*r)(?P<ch>[0-9]{1,4})(.ome.tif)", #lavision rawdata
    r"(.*)(?P<y>\d{2})(.*)(?P<x>\d{2})(.*C+)(?P<ch>[0-9]{1,2})(.*Z+)(?P<z>[0-9]{1,4})(.ome.tif)",
    r"(.*)(.*C+)(?P<ch>[0-9]{1,2})(.*Z+)(?P<z>[0-9]{1,4})(.ome.tif)", # lavision NONTILING + nonraw # or r"(.*)(.*C+)(?P<ch>[0-9]{1,2})(.*)(.*Z+)(?P<z>[0-9]{1,4})(.*)(.ome.tif)"
    r"(.*)(.*C+)(?P<ch>[0-9]{1,2})(.*)(.*Z+)(?P<z>[0-9]{1,4})(.ome.tif)"
         ]
    ###########determine raw vs non
    if raw==True:
        fl=[f for f in os.listdir(dr) if "raw_DataStack" in f or "raw_RawDataStack" in f] #sorted for raw files
    elif raw==False:
        fl=[f for f in os.listdir(dr) if "raw_DataStack" not in f and ".tif" in f or "raw_RawDataStack" in f]
    #### determine regex
    for regexpression in lst:
        reg=re.compile(regexpression)
        match=reg.match(fl[0])
        try:
            match.groups()
            break
        except AttributeError:
            pass #print ("Not correct regex")
    ############# check to see if data
    return regexpression


def generateparamdict(cwd, dst=False, update=False, **kwargs):
    """Function to automatically determine structure of light sheet scan, including channels, light sheets, tiles etc. This feeds the rest of the pipeline.
    _______
    Inputs:
        dst = (optional), set to name of param dictionary: e.g. "param_dict_local.p"; useful to make several param dicts each with a different cwd allowing for work on local machine and clusters
        Update(optional) set to False to force complete regeneration of param dict

    """
    #inputs
    kwargscopy=kwargs.copy() #this is done to get rid of the systemdirectory
    outdr=kwargs["outputdirectory"]
    raw=kwargs["rawdata"]
    kwargs["packagedirectory"]=cwd
    makedir(os.path.join(cwd, "logs"))
    if not "parameterfolder" in kwargs: kwargs["parameterfolder"] = os.path.join(kwargs["packagedirectory"], "parameterfolder")
    ############# make output directory #############
    makedir(outdr[:outdr.rfind("/")]) #in case of nested folders
    makedir(outdr)
    #optional for updating params from same initial function
    if os.path.exists(kwargs["outputdirectory"]+"/param_dict.p") and update:
        updateparams(kwargs["outputdirectory"], **kwargs)
    else:
        ###unpack input directories
        vols=[]
        for pth, lst in kwargs["inputdictionary"].items():
            if "regexpression" in kwargs:
                regex = kwargs["regexpression"]
                sys.stdout.write("\nUsing given regular expression: {}\n".format(regex)); sys.stdout.flush()
            else:
                regex=regex_determiner(raw, pth)
                kwargs["regexpression"]=regex
                sys.stdout.write("\nDetermined regex as {}\n".format(regex)); sys.stdout.flush()
    
            writer(outdr, "\n*******************STEP 0**********************************\n")
            
            ##update params with channeldictionary of files
            tmpdct = flatten(pth, **kwargs) if raw else zchanneldct(pth, **kwargs)
            if raw: print ("USING RAW DATA")
            if not raw: print ("Using postprocessed data (nonraw data)")
            
            #make the beast
            for i in lst: ### i=["regch", "00"]
                vol=volume(i[0])
                vol.add_channel(i[1])
                vol.add_raw(kwargs["rawdata"])
                vol.add_outdr(kwargs["outputdirectory"])
                vol.add_indr(pth)
                vol.add_packagedirectory(kwargs["packagedirectory"])
                vol.add_brainname(pth[pth.rfind("/")+8:-9])
                #add downsized volume path. This is done here instead of during process to prevent pickling IO issues
                vol.add_downsized_vol(os.path.join(outdr, vol.brainname+"_resized_ch"+vol.channel))
                vol.add_resampled_for_elastix_vol(vol.downsized_vol + "_resampledforelastix.tif")
                vol.add_injcoordinatesfld(os.path.join(outdr, "injection", "injcoordinatesfld"))
                makedir(os.path.join(outdr, "injection")); makedir(vol.injcoordinatesfld)
                vol.add_injdetect3dfld(os.path.join(outdr, "injection", "injdetect3d"))
                makedir(vol.injdetect3dfld)
                vol.add_allen_id_table(os.path.join(kwargs["packagedirectory"], "supp_files/allen_id_table.xlsx"))
                vol.add_full_sizedatafld(os.path.join(outdr, "full_sizedatafld"))
                vol.add_full_sizedatafld_vol(os.path.join(vol.full_sizedatafld, "{}_ch{}".format(vol.brainname, vol.channel)))
                vol.add_regex(regex)
                vol.add_tiling_overlap(kwargs["tiling_overlap"])
                vol.add_xyz_scale(kwargs["xyz_scale"])
                vol.add_atlasfile(kwargs["AtlasFile"])
                vol.add_elastixfld(outdr+"/elastix")
                vol.add_reg_to_atlas_vol(outdr+"/elastix/result.{}.tif".format(len([xx for xx in os.listdir(kwargs["parameterfolder"]) if "Order" in xx and "~" not in xx])-1))
                vol.add_reg_to_atlas_transform(outdr+"/elastix/TransformParameters.{}.txt".format(len([xx for xx in os.listdir(kwargs["parameterfolder"]) if "Order" in xx and "~" not in xx])-1))
                secondary_registration = kwargs["secondary_registration"] if "secondary_registration" in kwargs else True
                if not secondary_registration: vol.add_ch_to_reg_to_atlas(outdr+"/elastix/{}_resized_ch{}/result.{}.tif".format(vol.brainname, vol.channel, len([xx for xx in os.listdir(kwargs["parameterfolder"]) if "Order" in xx and "~" not in xx])-1))
                if secondary_registration: vol.add_ch_to_reg_to_atlas(outdr+"/elastix/{}_resized_ch{}/result.tif".format(vol.brainname, vol.channel, len([xx for xx in os.listdir(kwargs["parameterfolder"]) if "Order" in xx and "~" not in xx])-1))
                vol.add_inverse_elastixfld(outdr+"/elastix_inverse_transform")
                try:
                    vol.add_parameterfolder(kwargs["parameterfolder"])
                except KeyError:
                    kwargs["parameterfolder"]=os.path.join(kwargs["packagedirectory"], "parameterfolder")
                    vol.add_parameterfolder(kwargs["parameterfolder"])
                makedir(vol.full_sizedatafld)
                    ############# check to see if data is raw or preprocessed from imspector #############
                vol.add_fullsizedimensions(tmpdct["fullsizedimensions"])
                vol.add_horizontalfoci(tmpdct["horizontalfoci"])
                vol.add_lightsheets(tmpdct["lightsheets"])
                vol.add_ytile(tmpdct["ytile"])
                vol.add_xtile(tmpdct["xtile"])
                ##parse out only the correct channel and add as planes
                #tmpzdct={}; [tmpzdct.update(dict([(keys, dict([(k,v)]))])) for keys, values in tmpdct["zchanneldct"].items() for k, v in values.items() if i[1] == k]
                tmpzdct={}; [tmpzdct.update(dict([(keys, dict([(k,v)]))])) for keys, values in tmpdct["zchanneldct"].items() for k, v in values.items() if int(i[1]) == int(k)]
                if len(tmpzdct)==0: print("zdct contains 0 planes - this will result in an error for stitching")
                vol.add_zdct(tmpzdct)
                writer(outdr, "\nGenerated volume class('{}') channel('{}') pth: {}\n".format(vol.ch_type, vol.channel, vol.indr))
                ###add the volume to the vols list
                vols.append(vol)
        #add volumes
        kwargs.update(dict([("volumes", vols)]))
        #copy cuz it works better for some reason
        kwargs.update(kwargscopy)
    
        #save it
        writer(outdr, "\nParameter dictionary saved in the outputdirectory: {}\n".format(kwargs["outputdirectory"]))
        if not dst: dst = kwargs["outputdirectory"]+"/param_dict.p"
        save_kwargs(pckloc=dst, **kwargs)

    return


def updateparams(cwd=False, **kwargs):
    """Function to update parameter dictionary that feeds the rest of the pipeline. Useful for developing but this shouldn"t need to be used often. Can be used to "reinitialize" volumes if the class has been changed.
    _______
    Inputs:
        cwd = current working directory
        dst = (optional), set to name of param dictionary: e.g. "param_dict_local.p"; useful to make several param dicts each with a different cwd allowing for work on local machine and clusters


        ##FIXME: fix this to a try statemnt for original raw data and then decouple the basedirectory
    """
    if cwd:
        try:
            oldkwargs = load_kwargs(cwd)#**kwargs)
        except IOError:
            oldkwargs = load_kwargs(os.path.dirname(cwd))
    else:
        cwd = os.getcwd()
        oldkwargs = load_kwargs(**kwargs)
    makedir(os.path.join(cwd, "logs"))
    oldkwargs.update(kwargs)
    kwargs = oldkwargs
    #inputs
    outdr=kwargs["outputdirectory"]
    raw=kwargs["rawdata"]
    if not "packagedirectory" in kwargs: kwargs["packagedirectory"] = os.getcwd()
    if not "parameterfolder" in kwargs: kwargs["parameterfolder"] = os.path.join(kwargs["packagedirectory"], "parameterfolder")
    ############# make output directory #############
    makedir(outdr[:outdr.rfind("/")]) #in case of nested folders
    makedir(outdr)
    ###unpack input directories
    vols=[]
    for pth, lst in kwargs["inputdictionary"].items():
        writer(outdr, "\n*******************STEP 0 UPDATE**********************************\n")
        if "regexpression" in kwargs:
            regex = kwargs["regexpression"]
            sys.stdout.write("\nUsing given regular expression: {}\n".format(regex)); sys.stdout.flush()
        else:
            regex=regex_determiner(raw, pth)
            kwargs["regexpression"]=regex
            sys.stdout.write("\nDetermined regex as {}\n".format(regex)); sys.stdout.flush()
                #attempt to redo input this could fail
        try:
            if raw:
                tmpdct=flatten(pth, **kwargs) ##update params with channeldictionary of files
                print ("USING RAW DATA")
            #### NON RAW NOT TESTED!!
            elif raw==False:
                print ("Using postprocessed data (nonraw data)")
                tmpdct = zchanneldct(pth, **kwargs) #update params with channeldictionary of files
        except:
            sys.stdout.write("\nUpdating original input data failed...continuing.\n"); sys.stdout.flush()
        #make the beast
    for old_vol in kwargs["volumes"]:
        vol=volume(old_vol.ch_type)
        vol.add_channel(old_vol.channel)
        vol.add_raw(kwargs["rawdata"])
        vol.add_outdr(kwargs["outputdirectory"])
        vol.add_indr(pth)
        vol.add_packagedirectory(kwargs["packagedirectory"])
        vol.add_brainname(pth[pth.rfind("/")+8:-9])
        #add downsized volume path. This is done here instead of during process to prevent pickling IO issues
        vol.add_downsized_vol(os.path.join(outdr, vol.brainname+"_resized_ch"+vol.channel))
        vol.add_resampled_for_elastix_vol(vol.downsized_vol + "_resampledforelastix.tif")
        vol.add_injcoordinatesfld(os.path.join(outdr, "injection", "injcoordinatesfld"))
        makedir(os.path.join(outdr, "injection")); makedir(vol.injcoordinatesfld)
        vol.add_injdetect3dfld(os.path.join(outdr, "injection", "injdetect3d"))
        makedir(vol.injdetect3dfld)
        vol.add_allen_id_table(os.path.join(kwargs["packagedirectory"], "supp_files/allen_id_table.xlsx"))
        vol.add_full_sizedatafld(os.path.join(outdr, "full_sizedatafld"))
        vol.add_full_sizedatafld_vol(os.path.join(vol.full_sizedatafld, "{}_ch{}".format(vol.brainname, vol.channel)))
        vol.add_regex(regex)
        vol.add_tiling_overlap(kwargs["tiling_overlap"])
        vol.add_xyz_scale(kwargs["xyz_scale"])
        vol.add_atlasfile(kwargs["AtlasFile"])
        vol.add_elastixfld(outdr+"/elastix")
        vol.add_reg_to_atlas_vol(outdr+"/elastix/result.{}.tif".format(len([xx for xx in os.listdir(kwargs["parameterfolder"]) if "Order" in xx and "~" not in xx])-1))
        vol.add_reg_to_atlas_transform(outdr+"/elastix/TransformParameters.{}.txt".format(len([xx for xx in os.listdir(kwargs["parameterfolder"]) if "Order" in xx and "~" not in xx])-1))
        secondary_registration = kwargs["secondary_registration"] if "secondary_registration" in kwargs else True
        if not secondary_registration: vol.add_ch_to_reg_to_atlas(outdr+"/elastix/{}_resized_ch{}/result.{}.tif".format(vol.brainname, vol.channel, len([xx for xx in os.listdir(kwargs["parameterfolder"]) if "Order" in xx and "~" not in xx])-1))
        if secondary_registration: vol.add_ch_to_reg_to_atlas(outdr+"/elastix/{}_resized_ch{}/result.tif".format(vol.brainname, vol.channel, len([xx for xx in os.listdir(kwargs["parameterfolder"]) if "Order" in xx and "~" not in xx])-1))
        vol.add_inverse_elastixfld(outdr+"/elastix_inverse_transform")
        try:
            vol.add_parameterfolder(kwargs["parameterfolder"])
        except KeyError:
            kwargs["parameterfolder"]=os.path.join(kwargs["packagedirectory"], "parameterfolder")
            vol.add_parameterfolder(kwargs["parameterfolder"])
        makedir(vol.full_sizedatafld)
        #copy over old information to new volume
        vol.add_fullsizedimensions(old_vol.fullsizedimensions)
        vol.add_horizontalfoci(old_vol.horizontalfoci)
        vol.add_lightsheets(old_vol.lightsheets)
        vol.add_ytile(old_vol.ytile)
        vol.add_xtile(old_vol.xtile)
        vol.add_zdct(old_vol.zdct)
        writer(outdr, "\nUpdated volume class('{}') channel('{}') pth: {}\n".format(vol.ch_type, vol.channel, vol.indr))
        ###add the volume to the vols list
        vols.append(vol)
    #add volumes
    del kwargs["volumes"]
    kwargs.update(dict([("volumes", vols)]))

    #save it
    writer(outdr, "\nParameter dictionary saved in the outputdirectory: {}\n".format(kwargs["outputdirectory"]))
    save_kwargs(**kwargs)

    return kwargs


def zchanneldct(dr=None, **kwargs):
    """Regex to create zplane dictionary of subdictionaries (keys=channels, values=single zpln list sorted by x and y)
    ****function filters out nonraw data****

    """
    ####INPUTS:
    if dr == None:
        vols = kwargs["volumes"]
        reg_vol = [xx for xx in vols if xx.ch_type == "regch"][0]
        dr = reg_vol.indr
    regexpression = kwargs["regexpression"]

    ###load files from directory
    fl = [os.path.join(dr, f) for f in os.listdir(dr) if ".tif" in f and "raw_DataStack" not in f]
    reg=re.compile(regexpression)
    matches = list(map(reg.match, fl))
    dct={}
    ##find index of z,y,x,ch in matches
    z_indx=matches[0].span("z")
    try:
        y_indx=matches[0].span("y")
        x_indx=matches[0].span("x")
        tiling = True
    except IndexError:
        y_indx=1
        x_indx=1
        tiling = False

    ###determine number of channels in folder
    chs=[]; [chs.append(matches[i].group("ch")) for i in range(len(matches)) if matches[i].group("ch") not in chs]
    ##make dct consisting of each channel sorted by z plane and then in xy order (topleft-->top right to bottom-right)
    if tiling:
        for ch in chs:
            lst=[]
            try:
                [lst.append("".join(match.groups())) for num,match in enumerate(matches) if ch in match.group("ch")]
            except:
                pass
            srtd=sorted(lst, key=lambda a: (a[z_indx[0]:z_indx[1]], a[y_indx[0]:y_indx[1]], a[x_indx[0]:x_indx[1]])) #sort by z, then x, then y
            ytile=int(max([yy for f in lst for yy in f[y_indx[0]:y_indx[1]]]))+1 #automatically find the number of tiles
            xtile=int(max([xx for f in lst for xx in f[x_indx[0]:x_indx[1]]]))+1 #automatically find the number of tiles
            intvl=xtile*ytile
            grpd=[srtd[itr*intvl:(intvl*(itr+1))] for itr in range(int(len(srtd)/intvl))] ##each zplane in sequence of x positions then y position Top Left to bottom right
            dct[ch]=grpd
    #FIXME: This should work but haven"t tested
    elif not tiling:
        for ch in chs:
            lst=[]
            try:
                [lst.append("".join(match.groups())) for num,match in enumerate(matches) if ch in match.group("ch")]
            except:
                pass
            srtd=sorted(lst, key=lambda a: (a[z_indx[0]:z_indx[1]])) #sort by z, then x, then y
            ytile= 1
            xtile= 1
            intvl=xtile*ytile
            grpd=[srtd[itr*intvl:(intvl*(itr+1))] for itr in range(int(len(srtd)/intvl))] ##each zplane in sequence of x positions then y position Top Left to bottom right
            dct[ch]=grpd
    ###dictionary: zplane=keys, values=channeldictionary(keys=channels, values=zplane sorted by xandy)
    zdct={}
    for zpln in range(len(dct[ch])):
        zplndct={}
        for ch in chs:
            zplndct[ch]=dct[ch][zpln]
        zdct.update(dict([(str(zpln).zfill(4), zplndct)]))
    #New: find light sheets
    try:
        ls_indx=matches[0].span("ls")
        lsheets=[]; [lsheets.append(matches[i].group("ls")[-2:]) for i in range(len(matches)) if matches[i].group("ls")[-2:] not in lsheets]
        lsheets=int(max([lsh for f in lst for lsh in f[ls_indx[0]:ls_indx[1]]]))+1 #automatically find the number of light sheets
    except IndexError:
        lsheets=1
    #New find HF:
    with tifffile.TiffFile(os.path.join(dr, "".join(matches[0].groups()))) as tif:
        hf=len(tif.pages) #number of horizontal foci
        y,x=tif.pages[0].shape
        tif.close()

    print ("{} Zplanes found".format(len(zdct.keys())))
    print ("{} Channels found".format(len(zdct["0000"].keys())))
    print ("{}x by {}y tile scan determined".format(xtile, ytile))
    #return dict([("zchanneldct", zdct), ("xtile", xtile), ("ytile", ytile), ("channels", chs)])
    return dict([("zchanneldct", zdct), ("xtile", xtile), ("ytile", ytile), ("channels", chs), ("lightsheets", lsheets), ("horizontalfoci", hf), ("fullsizedimensions", (len(zdct.keys()),(y*ytile),(x*xtile)))])



def arrayjob(jobid, cores=5, compression=0, verbose=True, **kwargs):
    """wrapper function to submit to process_planes
    """

    kwargs = load_kwargs(kwargs["outputdirectory"])
    sf = int(kwargs["slurmjobfactor"])
    [process_planes(job, cores, compression, verbose, **kwargs) for job in [(jobid*sf)+x for x in range(sf)]] #submits sequential jobs in range starting at arrayid for length slurmfactor

    return


def process_planes(job, cores, compression, verbose=True, **kwargs):
    """Function for Slurm Array job to process single zplane; this could be parallelized"""
    ############################inputs
    kwargs = load_kwargs(kwargs["outputdirectory"])
    zpln = str(job).zfill(4)
    resizefactor=kwargs["resizefactor"]
    bitdepth = kwargs["bitdepth"] if "bitdepth" in kwargs else "uint16" #default to uint16 unless specified
    ####################################
    
    for i, vol in enumerate(kwargs["volumes"]):
        if i==0: writer(vol.full_sizedatafld, "Processing zpln {}...".format(zpln), flnm="process.txt", verbose=verbose)
        try:
            dct=vol.zdct[zpln] #dictionary of files for single z plane
        except KeyError:
            print("ArrayJobID/SF exceeds number of planes. Skipping")
            continue
        #handle raw vs nonraw
        if vol.raw: stitchdct = flatten_stitcher(cores, vol.outdr, vol.tiling_overlap, vol.xtile, vol.ytile, zpln, dct, vol.lightsheets, **kwargs)
        if not vol.raw: stitchdct = stitcher(cores, vol.outdr, vol.tiling_overlap, vol.xtile, vol.ytile, zpln, dct, **kwargs)

        #save for eventual compression
        saver(cores, stitchdct, vol.full_sizedatafld, vol.brainname, zpln, compression, bitdepth)

        #####################Cell detect ###CAN ADD A INJ CHANNEL HERE
        #resize
        resize_save(cores, stitchdct, vol.outdr, resizefactor, vol.brainname, zpln, bitdepth) ##can return location of cell channel but not interested in this; if so remove [0]
        writer(vol.full_sizedatafld, "\n   completed zpln {}, channel {}, channel_type {}, nm {}".format(zpln, vol.channel, vol.ch_type, os.path.basename(vol.full_sizedatafld_vol)), flnm="process.txt", verbose=verbose)

    #log and return
    writer(vol.full_sizedatafld, "\n   ...completed all volumes of zpln {}\n".format(zpln), flnm="process.txt", verbose=verbose)   
    return

def process_planes_from_fullsizedatafolder(job, cores, compression, verbose=True, **kwargs):
    """Easy way to pull outfiles from fullsizedatafolder"""
    ############################inputs
    #zd additions
    bitdepth = kwargs["bitdepth"] if "bitdepth" in kwargs else "uint16" #default to uint16 unless specified
    if cores < 2:
        zpln = str(job).zfill(4)
        ####################################
        for vol in kwargs["volumes"]:
            fl = [xx for xx in listdirfull(vol.full_sizedatafld_vol) if "Z{}".format(zpln) in xx]
            if len(fl) == 1: resize_save_helper(vol.outdr, kwargs["resizefactor"], vol.brainname, zpln, vol.channel, 
                  tifffile.imread(fl[0]).astype(bitdepth), bitdepth)
    else:
        p = mp.Pool(cores)
        for vol in kwargs["volumes"]:
            fl = [xx for xx in listdirfull(vol.full_sizedatafld_vol, "tif")]; fl.sort()
            zplns = [str(job).zfill(4) for job in range(len(fl))]
            iterlst = []; [iterlst.append((vol.outdr, kwargs["resizefactor"], vol.brainname, zpln, 
                       vol.channel, tifffile.imread(fl[i]).astype(bitdepth), bitdepth)) for i, zpln in enumerate(zplns)]
            #parallelize    
            p.starmap(resize_save_helper, iterlst)
    return



def process_planes_completion_checker(**kwargs):
    """Function to check to ensure each of the process_planes array jobs saved the full size stitched planes for each volume.
    
    Checks for numbers and file size
    """
    kwargs = load_kwargs(**kwargs)
    outdr = kwargs["outputdirectory"]
    zmax = kwargs["volumes"][0].fullsizedimensions[0]
    writer(outdr, "\n***********************STEP 1***********************\n\n")
    #check lenght of folder vs zmax
    for vol in kwargs["volumes"]:
        try:
        #check if number or file sizes are bad
            if len(listdirfull(vol.full_sizedatafld_vol, keyword=".tif")) != zmax or len(find_outlier_files_from_size(listdirfull(vol.full_sizedatafld_vol, keyword=".tif")))>0:
                writer(outdr, "STEP 1: FAILED: {} files found in {}. Should have {}.\n".format(len(listdirfull(vol.full_sizedatafld_vol, keyword=".tif")), vol.full_sizedatafld_vol[vol.full_sizedatafld_vol.rfind("/")+1:], zmax))
                completed = [int(xx[-8:-4]) for xx in listdirfull(vol.full_sizedatafld_vol, keyword=".tif") if int(xx[-8:-4]) in range(zmax)]
                missing = list(set(range(zmax)).difference(set(completed)))+find_outlier_files_from_size(listdirfull(vol.full_sizedatafld_vol, keyword=".tif"))
                tick = 0
                while not np.any((len(missing) == 0, tick>=8)):
                    writer(outdr, "   Loop {}, Missing planes: \n    {}\n\n     Attempting to fix".format(tick+1, missing))
                    try:
                        [process_planes(job, 3, 1, verbose=True, **kwargs) for job in list(missing)]
                        if len(listdirfull(vol.full_sizedatafld_vol, keyword=".tif")) != zmax:
                            writer(outdr, "\n\n       **process_planes_completion_checker successfully fixed missing planes**\n".format(missing))
                    except Exception as e:
                        print ("        Error on loop {}:\n          {}\n".format(tick+1, str(e)))
                    tick+=1
                    completed = [int(xx[-8:-4]) for xx in listdirfull(vol.full_sizedatafld_vol, keyword=".tif") if int(xx[-8:-4]) in range(zmax)]
                    missing = list(set(range(zmax)).difference(set(completed)))+find_outlier_files_from_size(listdirfull(vol.full_sizedatafld_vol, keyword=".tif"))

            else:
                writer(outdr, "\n\n***STEP 1: Correct number({}) of files found for {}.**\n\n".format(len(listdirfull(vol.full_sizedatafld_vol, keyword=".tif")), vol.full_sizedatafld_vol[vol.full_sizedatafld_vol.rfind("/")+1:]))
        except Exception as e:
          print ("process_planes_completion_checker has failed :/ \n\nerror message: \n{}".format(str(e)))
    return


def find_outlier_files_from_size(src, deviation = 5):
    """Function to look for files that are larger or smaller(important one) based on other files" sizes
    
    src = list of files
    deviation = number of deviations in file size to consider
    """
    arr = np.asarray([os.stat(fl).st_size for fl in src])
    mean = np.mean(arr)
    std = np.std(arr)
    lst = [fl for fl in src if os.stat(fl).st_size > (mean+deviation*std) or os.stat(fl).st_size < (mean-deviation*std)]
    return lst


def combiner_completionchecker(**kwargs):
    time.sleep(10) #10 second wait period to ensure last files have been fully saved
    outdr=kwargs["outputdirectory"]
    with open(os.path.join(kwargs["outputdirectory"], "param_dict.p"), "rb") as pckl:
        kwargs.update(pickle.load(pckl))
        pckl.close()
    resizedregdct=kwargs["resizedregchtif"]
    resizedsigdct=kwargs["resizedsigchtif"]
    ############################
    allchns={}; allchns.update(resizedregdct); allchns.update(resizedsigdct)
    chnlst=[]; missing_chn=[]
    for ch in allchns.keys():
        try:
            with open(os.path.join(outdr, "LogFile.txt"), "r") as f:
                s = mmap.mmap(f.fileno(), 0, access=mmap.ACCESS_READ)
                s.seek(s.find(str(ch)+" in")); chn=s.readline(); chn=chn[:chn.rfind(" in")]
                s.close()
            chnlst.append(chn)
        except ValueError:
           missing_chn.append(ch)
    if len(chnlst) == len(allchns):
        writer(outdr, "******************STEP 2******************\n      All chns: {} combined!".format(chnlst))
    else:
        writer(outdr, "******************STEP 2******************\n      Chns not combined {}".format(missing_chn))
    return


def resize_save(cores, stitchdct, outdr, resizefactor, brainname, zpln, bitdepth):
    makedir(outdr)
    if cores>1:
        try:
            p
        except NameError:
            p = mp.Pool(cores)
        iterlst = []; [iterlst.append((outdr, resizefactor, brainname, zpln, ch, im, bitdepth)) for ch,im in stitchdct.items()]
        svloc = p.starmap(resize_save_helper, iterlst); svloc.sort()
        p.terminate()
    else:
        for ch,im in stitchdct.items():
            svloc = resize_save_helper(outdr, resizefactor, brainname, zpln, ch, im, bitdepth)
    return svloc

def resize_save_helper(outdr, resizefactor, brainname, zpln, ch, im, bitdepth):
    
    svloc = os.path.join(outdr, brainname+"_resized_ch"+ch)
    makedir(svloc)
    if len(im.shape) == 2:  #grayscale images
        y,x = im.shape
        xfct = int(x/resizefactor)
        yfct = int(y/resizefactor)
    elif len(im.shape) == 3: #color images from cell detect
        y,x,c = im.shape
        xfct = int(x/resizefactor)
        yfct = int(y/resizefactor)
    im1 = cv2.resize(im, (xfct, yfct), interpolation=cv2.INTER_LINEAR) #interpolation=cv2.INTER_AREA) #
    tifffile.imsave(os.path.join(svloc, brainname + "_C"+ch+"_Z"+zpln+".tif"), im1.astype(bitdepth))
    del im1, im
    return svloc

def saver(cores, stitchdct, outdr, brainname, zpln, compression, bitdepth):
    
    makedir(outdr)
    if cores > 1:
        try:
            p
        except NameError:
            p = mp.Pool(cores)
        iterlst = []; [iterlst.append((outdr, brainname, zpln, ch, im, compression, bitdepth)) for ch,im in stitchdct.items()]
        lst = p.starmap(saver_helper, iterlst); lst.sort()
        p.terminate(); del p
    else:
        for ch,im in stitchdct.items():
            lst = saver_helper(outdr, brainname, zpln, ch, im, compression, bitdepth)
    return lst

def saver_helper(outdr, brainname, zpln, ch, im, compression, bitdepth):
    svloc = os.path.join(outdr, brainname +"_ch"+ch)
    makedir(svloc)
    pth = os.path.join(svloc, brainname + "_C"+ch+"_Z"+zpln+".tif")
    tifffile.imsave(pth, im.astype(bitdepth), compress=compression); del im
    return pth



def stitcher(cores, outdr, ovlp, xtile, ytile, zpln, dct, blndtype = False, intensitycorrection = False, **kwargs):
    """return numpy arrays of     """
    #easy way to set ch and zplnlst
    ["stitching for ch_{}".format(ch) for ch, zplnlst in dct.items()] #cheating way to set ch and zplnlst
    
    ###dim setup
    zplnlst = dct[list(dct.keys())[0]] #changed for py37
    ydim, xdim =cv2.imread(zplnlst[0], -1).shape
    xpxovlp=int(ovlp*xdim)
    ypxovlp=int(ovlp*ydim)
    tiles=len(zplnlst) ##number of tiles

    #check for inputs:
    if not blndtype and "blendtype" in kwargs: blndtype = kwargs["blendtype"]
    if not blndtype and "blendtype" not in kwargs: blndtype = "sigmoidal"
    if not intensitycorrection: intensitycorrection = kwargs["intensitycorrection"]
    if "blendfactor" in kwargs:
        blendfactor = int(kwargs["blendfactor"])
    else:
        blendfactor = 4

### blending setup
    if blndtype == "linear":
        alpha=np.tile(np.linspace(0, 1,num=xpxovlp), (ydim, 1)) ###might need to change 0-1 to 0-255?
        yalpha=np.swapaxes(np.tile(np.linspace(0, 1, num=int(ypxovlp)), (int(xdim+((1-ovlp)*xdim*(xtile-1))),1)), 0, 1)

    elif blndtype == "sigmoidal":
        alpha=np.tile(scipy.stats.logistic.cdf(np.linspace(-blendfactor, blendfactor,num=xpxovlp)), (ydim, 1)) ###might need to change 0-1 to 0-255?
        yalpha=np.swapaxes(np.tile(scipy.stats.logistic.cdf(np.linspace(-blendfactor, blendfactor, num=int(ypxovlp))), (int(xdim+((1-ovlp)*xdim*(xtile-1))),1)), 0, 1)

    elif blndtype == False or blndtype == None: #No blending: generate np array with 0 for half of overlap and 1 for other.
        alpha=np.zeros((ydim, xpxovlp)); alpha[:, int(xpxovlp/2):] = 1
        yalpha=np.zeros((ypxovlp, (xdim+((1-ovlp)*xdim*(xtile-1))))); yalpha[int(ypxovlp/2):, :] = 1

    else: #default to sigmoidal
        alpha=np.tile(scipy.stats.logistic.cdf(np.linspace(-blendfactor, blendfactor,num=xpxovlp)), (ydim, 1)) ###might need to change 0-1 to 0-255?
        yalpha=np.swapaxes(np.tile(scipy.stats.logistic.cdf(np.linspace(-blendfactor, blendfactor, num=int(ypxovlp))), (int(xdim+((1-ovlp)*xdim*(xtile-1))),1)), 0, 1)


    iterlst = [(xdim, ydim, xtile, ytile, ovlp, xpxovlp, ypxovlp, tiles, alpha, yalpha, zpln, ch, zplnlst, intensitycorrection) for ch, zplnlst in dct.items()]
    ##parallel processing
    #OFF RIGHT NOW
    if False:#cores>1:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            #stitchlst=Parallel(n_jobs=cores)(delayed(xystitcher)(xdim, ydim, xtile, ytile, ovlp, xpxovlp, ypxovlp, tiles, alpha, yalpha, zpln, ch, zplnlst, intensitycorrection) for ch, zplnlst in dct.items())
            try:
                p
            except:
                p = mp.Pool(cores)
            stitchlst = p.starmap(xystitcher, iterlst)
    else:
        stitchlst = [xystitcher(i) for i in iterlst]

    stitchdct={}
    for i in stitchlst:
        stitchdct.update(i)
    return stitchdct
###

##need to take half from each frame rather than all overlap from one; and parallelize
def xystitcher(xdim, ydim, xtile, ytile, ovlp, xpxovlp, ypxovlp, tiles, alpha, yalpha, zpln, ch, zplnlst, intensitycorrection):
    """helper function for parallelization of stitcher; color is not functional yet"""
    warnings.warn("depreciated", DeprecationWarning)
    ytick=0
    xtick=0
    tick=0
    #cxfrm=np.zeros((ydim+((xtile-1)*ydim*(1-ovlp)), xdim+((xtile-1)*xdim*(1-ovlp)), 3)).astype("uint16")
    if intensitycorrection:
        while ytick < ytile:
            while xtick < xtile:
                #im=cv2.imread(os.path.join(dr,"".join(zplnlst[tick])), cv2.CV_LOAD_IMAGE_GRAYSCALE).astype("int16")
                #im=sitk.GetArrayFromImage(sitk.RescaleIntensity(sitk.GetImageFromArray(cv2.imread(os.path.join(dr,"".join(zplnlst[tick])), -1).astype("uint16")))) #TP adding intensity rescale to try and account for issues with different tiles
                im=cv2.imread("".join(zplnlst[tick]), -1).astype("uint16") #
                ###normalize
                #p5,p95 = np.percentile(im, [1,99])
                #im[im<p5] = p5
                #im[im>p95] = p95
                #im = (im-im.min())/(im.max()-im.min()) #normalize
                tick+=1
                if xtick != 0:
                    ###correct for overall image intensity difference by shifting higher mean overlapping pixels towards smaller
                    l_im_ovlp=xfrm[:,-(xpxovlp):]; r_im_ovlp=im[:,:(xpxovlp)]
                    #pl.figure(); pl.hist(l_im_ovlp.ravel(), bins=100, alpha=0.5); pl.hist(r_im_ovlp.ravel(), bins=100, alpha=0.5)
                    l_im_mean=np.mean(l_im_ovlp.ravel()); r_im_mean=np.mean(r_im_ovlp.ravel()); mean_dif=abs(r_im_mean - l_im_mean)
                    ###move the means two images closer to each other
                    if l_im_mean < r_im_mean:
                        im=im-mean_dif; im=im.clip(min=0)
                    elif r_im_mean < l_im_mean:
                        xfrm=xfrm - mean_dif; xfrm=xfrm.clip(min=0)
                    #stitch
                    xblend = (xfrm[:,-(xpxovlp):]*(1-alpha) + im[:,:(xpxovlp)] * (alpha)).astype("uint16")            #1-alhpa alpha
                    xfrm=np.concatenate((xfrm[:,:-(xpxovlp)], xblend, im[:,xpxovlp:]), axis=1)
                    #cxfrm[(ytick*ydim - (ydim*(ytick)*ovlp)):((ytick+1)*ydim - (ydim*(ytick)*ovlp)),(xtick*xdim - (xdim*(xtick)*ovlp)):((xtick+1)*xdim - (xdim*(xtick)*ovlp)),xtick]=np.concatenate((xblend, im[:,xpxovlp:]), axis=1)
                else:
                    xfrm=im
                    #cxfrm[(ytick*ydim - (ydim*(ytick)*ovlp)):((ytick+1)*ydim - (ydim*(ytick)*ovlp)),:xdim,0]=im
                xtick+=1
                #print ("xtick {}".format(xtick))
            xtick=0
            if ytick !=0:
                ###correct for overall image intensity difference by shifting higher mean overlapping pixels towards smaller
                t_im_ovlp=xyfrm[-(ypxovlp):,:]; b_im_ovlp=xfrm[:ypxovlp,:]
                #pl.hist(t_im_ovlp.ravel(), bins=100, alpha=0.5); pl.hist(b_im_ovlp.ravel(), bins=100, alpha=0.5)
                t_im_mean=np.mean(t_im_ovlp.ravel()); b_im_mean=np.mean(b_im_ovlp.ravel()); mean_dif=abs(t_im_mean - b_im_mean)
                ###move the means two images closer to each other
                if t_im_mean < b_im_mean:
                    xfrm=xfrm-mean_dif; xfrm=xfrm.clip(min=0)
                elif b_im_mean < t_im_mean:
                    xyfrm=xyfrm - mean_dif; xyfrm=xyfrm.clip(min=0)
                #stitch
                yblend=(xyfrm[-(ypxovlp):,:]*(1-yalpha) + xfrm[:(ypxovlp),:] * yalpha).astype("uint16")
                xyfrm=np.concatenate((xyfrm[:-(ypxovlp),:], yblend, xfrm[ypxovlp:,:]), axis=0)
                #cxfrm[(ytick*ydim - (ydim*(ytick)*ovlp)):((ytick+1)*ydim - (ydim*(ytick)*ovlp)),(xtick*xdim - (xdim*(xtick)*ovlp)):((xtick+1)*xdim - (xdim*(xtick)*ovlp)),xtick]=np.concatenate((xblend, im[:,xpxovlp:]), axis=1)
            else:
                xyfrm=xfrm
            ytick+=1
            #y,x,c=cxfrm.shape
            #cxfrm=cv2.resize(cxfrm, (x/4, y/4), cv2.INTER_AREA)
    elif not intensitycorrection:
        while ytick < ytile:
            while xtick < xtile:
                #im=cv2.imread(os.path.join(dr,"".join(zplnlst[tick])), cv2.CV_LOAD_IMAGE_GRAYSCALE).astype("int16")
                #im=sitk.GetArrayFromImage(sitk.RescaleIntensity(sitk.GetImageFromArray(cv2.imread(os.path.join(dr,"".join(zplnlst[tick])), -1).astype("uint16")))) #TP adding intensity rescale to try and account for issues with different tiles
                im=cv2.imread("".join(zplnlst[tick]), -1).astype("uint16") #
                ###normalize
                #p5,p95 = np.percentile(im, [1,99])
                #im[im<p5] = p5
                #im[im>p95] = p95
                #im = (im-im.min())/(im.max()-im.min()) #normalize
                tick+=1
                if xtick != 0:
                    xblend = (xfrm[:,-(xpxovlp):]*(1-alpha) + im[:,:(xpxovlp)] * (alpha)).astype("uint16")            #1-alhpa alpha
                    xfrm=np.concatenate((xfrm[:,:-(xpxovlp)], xblend, im[:,xpxovlp:]), axis=1)
                    #cxfrm[(ytick*ydim - (ydim*(ytick)*ovlp)):((ytick+1)*ydim - (ydim*(ytick)*ovlp)),(xtick*xdim - (xdim*(xtick)*ovlp)):((xtick+1)*xdim - (xdim*(xtick)*ovlp)),xtick]=np.concatenate((xblend, im[:,xpxovlp:]), axis=1)
                else:
                    xfrm=im
                    #cxfrm[(ytick*ydim - (ydim*(ytick)*ovlp)):((ytick+1)*ydim - (ydim*(ytick)*ovlp)),:xdim,0]=im
                xtick+=1
                #print ("xtick {}".format(xtick))
            xtick=0
            if ytick !=0:
                yblend=(xyfrm[-(ypxovlp):,:]*(1-yalpha) + xfrm[:(ypxovlp),:] * yalpha).astype("uint16")
                xyfrm=np.concatenate((xyfrm[:-(ypxovlp),:], yblend, xfrm[ypxovlp:,:]), axis=0)
                #cxfrm[(ytick*ydim - (ydim*(ytick)*ovlp)):((ytick+1)*ydim - (ydim*(ytick)*ovlp)),(xtick*xdim - (xdim*(xtick)*ovlp)):((xtick+1)*xdim - (xdim*(xtick)*ovlp)),xtick]=np.concatenate((xblend, im[:,xpxovlp:]), axis=1)
            else:
                xyfrm=xfrm
            ytick+=1
            #y,x,c=cxfrm.shape
            #cxfrm=cv2.resize(cxfrm, (x/4, y/4), cv2.INTER_AREA)
    #return dict([(ch, xyfrm), (ch+"_color", cxfrm)])
    return dict([(ch, xyfrm.astype("uint16"))])


def resample_par(cores, mv_pth, fx_pth=None, size=None, svlocname=None, singletifffile=None, resamplefactor=None):
    """resamples the folder mv_pth containing a series of tif files to the xyz size of fx_pth.
   need to specify size = (x, y, z) or fixed_image_path that is of desired resultant size of movingimage

   Inputs:
   -------------------------
       svlocname=None, returns array of resized image without saving. To save set svlocname to desired PATH, NAME, and optional .tif extension.
       singletifffile=None function takes location of folder contain individual tifffiles as mv_pth.
       singletifffile=True, mv_pth is the actual single tiffstack file; False/None mv_pth is folder
       resamplefactor: scale factor to shrink (<1) or expand (>1) to either size or fixed_image_path;
    """
    ###set up parallelization
    try:
        p
    except NameError:
        p=mp.Pool(cores)
    ###ensure proper input to function
    if fx_pth == None and size==None:
        raise Exception ("Need to specify fx_pth or size for resample function")
    if fx_pth !=None and size != None:
        raise Exception ("Cannot specify both fx_pth and size for resample function")
    total_time = time.time()
    ###determine output image stack size
    if isinstance(size, tuple):
        x, y, z = size
    elif isinstance(fx_pth, str):
        z,y,x=sitk.GetArrayFromImage(sitk.ReadImage(fx_pth)).shape
    if resamplefactor != None:
        x=int(float(resamplefactor)*float(x)); y=int(float(resamplefactor)*float(y)); z=int(float(resamplefactor)*float(z))
    tmpsvloc=os.path.join(os.getcwd(), "resampler_tmpfld_{}".format(random.randint(1,100)))
    tmpfl=os.path.join(tmpsvloc, "resampler_tmp_im.tif")
    makedir(tmpsvloc)
################# chunk then resize in xy ########################
    if not singletifffile:
        #find size of mv_pth
        zmov = len([xx for xx in listdirfull(mv_pth) if "~" not in xx])
        ymov, xmov = tifffile.imread(listdirfull(mv_pth)[0], multifile=False).shape
        #file list of moving image
        movingimage_list_full=[f for f in listdirfull(mv_pth) if ".tif" in f]; movingimage_list_full.sort();
        sys.stdout.write("Running resizing with {} cores\n  Dims {} -> {}\n".format(cores, (xmov, ymov, zmov), (x,y,z))); sys.stdout.flush()
        iterlst=[]; [iterlst.append((core, cores, x, y, movingimage_list_full, tmpsvloc)) for core in range(cores)]
        lst = p.starmap(xyresizer, iterlst); lst.sort()
        p.terminate()
    elif singletifffile:
        #find size of mv_pth
        try:
            with tifffile.TiffFile(mv_pth) as tif:
                zmov=len(tif.pages)
                #print zmov
                tif.close()
        except: #cuz sometimes the above fail
            print(mv_pth)
            zmov, ymov, xmov = tifffile.imread(mv_pth).shape
        sys.stdout.write("Running resizing with {} cores\n  Zplns {} -> {}\n".format(cores, (zmov), (z))); sys.stdout.flush()
        iterlst=[]; [iterlst.append((core, cores, x, y, mv_pth, zmov, tmpsvloc)) for core in range(cores)]
        lst = p.starmap(xyresizer2, iterlst); lst1=[]; [lst1.append(xx) for xx in lst if xx not in lst1]; del lst; lst=lst1; lst.sort()
    print("Completed XY resizing in {} seconds, starting to resize in Z...".format((time.time() - total_time)))
################### combine chunks ################
    z_time=time.time(); lst.sort()
    for i in range(len(lst)):
        im=np.load(lst[i]+".npy")
        try:
            stack = np.concatenate((stack, im), axis=0)
        except NameError:
            stack = im
    zstack, ystack, xstack = stack.shape
    ###clean up
    del lst
    shutil.rmtree(tmpsvloc)
    makedir(tmpsvloc)
##########swapaxes to yzx, save then resize in xz #####################
    stack=np.swapaxes(stack, 0, 1) #yzx
    tifffile.imsave(tmpfl, stack)
    del stack
    print("Running resizing with {} cores\n".format(cores))
    iterlst=[]; [iterlst.append((core, cores, xstack, z, tmpfl, ystack, tmpsvloc)) for core in range(cores)]
    lst = p.starmap(xyresizer2, iterlst); lst1=[]; [lst1.append(xx) for xx in lst if xx not in lst1]; del lst; lst=lst1; lst.sort()
    p.terminate()
    print("Completed XZ resizing in {} seconds\n".format((time.time() - z_time)))
################### re-combine chunks ################
    lst.sort();
    for i in range(len(lst)):
        im=np.load(lst[i]+".npy")
        try:
            stack = np.concatenate((stack, im), axis=0)
        except NameError:
            stack = im
    stack=np.swapaxes(stack, 0, 1)
    shutil.rmtree(tmpsvloc)
################### output or save ################
    if svlocname==None:
        print("Total time for XY and Z resizing: {} seconds.\nFile NOT SAVED. IF DESIRED make svlocname='desiredname'".format((time.time() - total_time), os.path.join(os.getcwd())))
        return stack
    elif svlocname != None:
        if svlocname[-4:] == ".tif":
            tifffile.imsave(os.path.join(svlocname), stack.astype("uint16"))
        else:
            tifffile.imsave(os.path.join(svlocname + ".tif"), stack.astype("uint16"))
        print("Total time for XY and Z resizing: {} seconds.\nFile saved as {}".format((time.time() - total_time), os.path.join(os.getcwd(), svlocname)))
        return
######################## helper functions for parallelization ############################
def xyresizer(core, cores, x, y, movingimage_list_full, tmpsvloc):
    chnkrng=chunkit(core, cores, movingimage_list_full)
    print("   chunk {} of {} - planes {}-{}\n".format((core+1), (cores+1), chnkrng[0], chnkrng[1]))
    image=tifffile.imread(movingimage_list_full[chnkrng[0]:chnkrng[1]])
    x=int(x); y=int(y)
    imstack=np.zeros((len(movingimage_list_full), y, x))
    for zpln in range(len(movingimage_list_full)):
        im=image[zpln,...]
        imstack[zpln,...]=cv2.resize(im, (x, y), interpolation=cv2.INTER_LINEAR) #cv2.INTER_AREA)
    del image
    np.save(tmpsvloc + "/_{}".format(str(core).zfill(3)), imstack.astype("uint16"))
    return tmpsvloc + "/_{}".format(str(core).zfill(3))

def xyresizer2(core, cores, x, y, mv_pth, zmov, tmpsvloc):
    chnkrng=chunkit(core, cores, zmov)
    sys.stdout.write("   chunk {} of {} - planes {}-{}\n".format((core+1), (cores+1), chnkrng[0], chnkrng[1])); sys.stdout.flush()
    image=tifffile.imread(mv_pth, key=range(chnkrng[0], chnkrng[1]))
    zim,yim,xim=image.shape
    x=int(x); y=int(y)
    imstack=np.zeros((zim, y, x))
    for zpln in range(zim):
        im=image[zpln,...]
        imstack[zpln,...]=cv2.resize(im, (x, y), interpolation=cv2.INTER_LINEAR) #cv2.INTER_AREA)
    del image
    np.save(tmpsvloc + "/_{}".format(str(core).zfill(3)), imstack.astype("uint16")); del imstack
    return tmpsvloc + "/_{}".format(str(core).zfill(3))


def color_movie_merger(movie2, movie4, savelocation=None, savefilename=None, movie5=None):
    """Function used to overlay movies to visually inspect registration quality. RED=Atlas, Green=Moving Image, Blue=OPTIONAL.
       movie2=atlas
       movie4=final elatixs output.
       savelocation=None: function returns np.array of movie (z,y,x,[rbg])
       savefilename=None: function saves as color_movie_merger_output
    Note: there is no movie 1 or 3. Naming convention is to prevent confusion with other functions."""
    ####################LOAD ROUTINE, Compatible with np arrays or saved files
    if movie2[-3:] == "tif" or movie2[-3:]=="mhd":
            m2=sitk.GetArrayFromImage(sitk.RescaleIntensity(sitk.ReadImage(movie2)))
    elif (type(movie2).__module__ == np.__name__) == True:
        m2 = movie2
    else:
        print ("ERROR: movie2 type not recognized")
    if movie4[-3:] == "tif" or movie4[-3:]=="mhd":
            m4=sitk.GetArrayFromImage(sitk.RescaleIntensity(sitk.ReadImage(movie4)))
    elif (type(movie4).__module__ == np.__name__) == True:
        m4 = movie4
    else:
        print ("ERROR: movie4 type not recognized")
    ###merging atlas and registered images into color image
    ########## GENERATION OF MERGED IMAGE
    #shape of images
    z,y,x = m2.shape
    #Make array with last dimenision being color (r,g,b)
    colorfile = np.zeros([z,y,x,3], "uint16") #z,y,x,color (0=r, 1=g, 2=b)
    colorfile[:,:,:, 1] =  m4[:,:,:]#green channel, use for registered image
    colorfile[:,:,:, 0] =  m2[:,:,:]#red channel use for atlas image
    ####optional movie5. Compatibility with numpy or saved images
    if movie5!= None:
        if movie5[-3:] == "tif" or movie5[-3:]=="mhd":
            m5=sitk.GetArrayFromImage(sitk.RescaleIntensity(sitk.ReadImage(movie5)))
        elif (type(movie5).__module__ == np.__name__) == True:
            m5 = movie5
        ###create blue channel
        colorfile[:,:,:, 2] =  m5[:,:,:]#blue channel, use for registered image
    ### saving as multipage tif using tifffile
    if savefilename == None:
        savefilename = "color_movie_merger_output"
    if savelocation != None:
    ##### Folder Creation
        makedir(savelocation)
        makedir(os.path.join(savelocation,"combinedmovies"))
        tifffile.imsave(os.path.join(savelocation,"combinedmovies", "Merged_{}.tif".format(savefilename)), colorfile)
        print ("Merged Color tif made for {}\nSaved in {}".format(savefilename, savelocation))
    elif savelocation == None:
        return colorfile


def resample(mv_pth, fx_pth=None, size=None, svlocname=None, singletifffile=None, resamplefactor=None, verbose=False):
    """resamples the folder mv_pth containing a series of tif files to the xyz size of fx_pth.
   need to specify size = (x, y, z) or fixed_image_path that is of desired resultant size of movingimage

   Inputs:
       svlocname=None, returns array of resized image without saving. To save set svlocname to desired name AND PATH, and extension.
       singletifffile=None function takes location of folder contain individual tifffiles as mv_pth.
       singletifffile=True, mv_pth is the actual single tiffstack file
       resamplefactor: scale factor to apply to either size or fixed_image_path. resamplefactor of 2 results in 2x(fixed_image)
       verbose
    Function has not been parallelized."""

###ensure proper input to function
    if fx_pth == None and size==None:
        raise Exception ("Need to specify fx_pth or size for resample function")
    if fx_pth !=None and size != None:
        raise Exception ("Cannot specify both fx_pth and size for resample function")
    ##determine output image stack size
    if isinstance(size, tuple):
        x, y, z = size
    elif isinstance(fx_pth, str):
        fixed=sitk.ReadImage(fx_pth) ##change this to cv2***
        x, y, z=fixed.GetSize() #
    total_time=time.time()
    if resamplefactor != None:
        x=int(float(resamplefactor)*float(x)); y=int(float(resamplefactor)*float(y)); z=int(float(resamplefactor)*float(z))
    #setup temp savefolder to save np arrays
    print("Resizing {}".format(mv_pth))
    ticker=0
    if singletifffile== None:
        ### make list of files for moving image
        movingimage_list = [f for f in os.listdir(mv_pth) if os.path.isfile (os.path.join(mv_pth, f))]
        movingimage_list_full=[]
        for items in movingimage_list:
            if items[-3:] == "tif":
                movingimage_list_full.append(os.path.join(mv_pth, items))
        movingimage_list_full.sort()
        length=len(movingimage_list_full)
        for files in movingimage_list_full:
            timer=time.time()
            image=cv2.imread(files, -1)     ###might need grayscale command: 0
            image=cv2.resize(image, (x,y), interpolation=cv2.INTER_LINEAR) #interpolation=cv2.INTER_AREA) #cv2.INTER_LINEAR, cv2.INTER_CUBIC, cv2.INTER_AREA
            try:
                filestack=np.dstack((filestack, image))
            except NameError:
                filestack=image
            ticker+=1
            if verbose: print("Resized slice {} of {} to {}x{} pixels in {} seconds".format(ticker, length, x, y, (time.time()-timer)))
        print("Completed XY resizing in {} seconds, starting to resize in Z...".format((time.time() - total_time)))
    elif singletifffile == True:
        imagestack=tifffile.imread(mv_pth)
        zmov, ymov, xmov = imagestack.shape
        for files in range(zmov):
            timer=time.time()
            image=imagestack[files,:,:]     ###might not need grayscale command
            image=cv2.resize(image, (x,y), interpolation=cv2.INTER_LINEAR) #interpolation=cv2.INTER_AREA) #cv2.INTER_LINEAR, cv2.INTER_CUBIC, cv2.INTER_AREA
            try:
                filestack=np.dstack((filestack, image))
            except NameError:
                filestack=image
            ticker+=1
            if verbose: print("Resized slice {} of {} to {}x{} pixels in {} seconds".format(ticker, zmov, x, y, (time.time()-timer)))
        print("Completed XY resizing in {} seconds, starting to resize in Z...".format((time.time() - total_time)))
#### need to figure out z now....
    ticker=0
    for xplane in filestack: ###YXZ
        timer=time.time()
        image=cv2.resize(xplane, (z,x), interpolation=cv2.INTER_LINEAR) #interpolation=cv2.INTER_AREA) #resize in xz
        try:
            finalfilestack=np.dstack((finalfilestack, image)) #####XZY
        except NameError:
            finalfilestack=image
        ticker+=1
        if verbose: print("Z Resizing {} of {} to {}x{} pixels in {} seconds".format(ticker, y, x, z, (time.time()-timer)))

    finalfilestack=np.swapaxes(finalfilestack, 0, 1) ####from XZY to ZXY
    finalfilestack=np.swapaxes(finalfilestack, 1, 2) ####from ZXY to ZYX
    finalfilestack=finalfilestack.astype("uint16") #convert from 64 bit to 16 bit
    if svlocname==None:
        print("Completed XY and Z resizing in {} seconds.\nFile NOT SAVED. IF DESIRED make svlocname='desiredname'".format((time.time() - total_time), os.path.join(os.getcwd())))
        return finalfilestack
    elif svlocname != None:
        tifffile.imsave(os.path.join(os.getcwd(), svlocname), finalfilestack.astype("uint16"))
        print("Completed XY and Z resizing in {} seconds.\nFile saved as {}".format((time.time() - total_time), os.path.join(os.getcwd(), svlocname)))


def getvoxels(filelocation, savefilename=None):
    """Function used to used to find coordinates of nonzero voxels in image. Used to quantify injection site of masked registered tiffstack
       Returns [z,y,x] coordinates
       use savefilename to instead save a numpy file of coordinates"""
    ##read image and convert to np array
    image=tifffile.imread(filelocation)
    ##find voxel coordinates of all points greater than zero (all should be 1)
    voxels=np.argwhere(image)#note this automatically looks for voxels above 0 intensity. But an arguement can be used if nonthresholded image
    if savefilename==None:
        return voxels
    else:
        np.save(savefilename, voxels)


def tiffcombiner(jobid, cores = None, remove = True, save_before_reorientation = False, **kwargs):
    """function to load/save all tifs within a folder, function deletes folder of individual tifs and saves the single multipage tif

    Inputs:
    ---------------------
        jobid = chindex: ch to be processed (eg. brainname+resized_ch00, brainname+resized_ch01_cells)
        cores (optional) = # of cores to use for parallelization, if None will use a single core
        remove (optional) = if True removes folder, if False keeps folder
        save_before_reorientation (optional) = if true, saves tifffile before orientation, useful when troubleshooting rotations

        kwargs"""
    ##########inputs
    kwargs = load_kwargs(kwargs["outputdirectory"])
    vols=kwargs["volumes"]
    start=time.time()

    ### set job
    if jobid >= len(vols):
        print ("jobid greater than number of channels\n") #protects against extra array jobs
        return
    lst = []; [lst.append(vol.downsized_vol) for vol in vols]
    vol_to_process = lst[jobid]

    ###load ims and return dct of keys=str(zpln), values=np.array
    sys.stdout.write("Jobid: {},\n   loading {}\n\n   if memory issues with the cluster, it will happen after this line...\n\n   ".format(jobid, vol_to_process))

    #note this is not the most efficient way to do things, but gives the highest chance of not resulting in memory issues on cluster
    if not cores:
        imstack=load_tif_list(vol_to_process)
    if cores:
        try:
            impth=load_tif_list(vol_to_process, cores=cores, save=vol_to_process)
            imstack = np.lib.format.open_memmap(impth, dtype = "uint16", mode = "r")
        except ValueError as e:
            print("Error {} \n loading it the slow way".format(e))
            imstack=load_tif_list(vol_to_process)

    #squeeze if necessary depending on np/tifffile versions
    if len(imstack.shape) >3:
        imstack=np.squeeze(imstack)

    #optional saveout:
    if save_before_reorientation:
        z,y,x = imstack.shape
        fp = np.lib.format.open_memmap(vol_to_process[:vol_to_process.rfind("/")]+"tiffcombiner_before_reorientation.npy", dtype = "uint16", mode = "w+", shape = (z,y,x))
        fp = imstack
        sys.stdout.write("\n     Saving image before reorientation as: {}".format(vol_to_process[:vol_to_process.rfind("/")]+"tiffcombiner_before_reorientation.npy"))

    #account for orientation differences between Atlas and volume
    imstack = fix_orientation(imstack, verbose = True, **kwargs)
    sys.stdout.write("\nimstack shape after reorientation: {}\n".format(imstack.shape))
    tifffile.imsave(vol_to_process + ".tif", imstack.astype("uint16"))

    #clean up
    if remove:
        shutil.rmtree(vol_to_process)
        try:
            if cores: removedir(impth)
        except:
            pass
    writer(kwargs["outputdirectory"], "\n\n**************STEP 2********************\n     tiffcombine step job({}):\n       {}\n       in {} seconds\n*********************************\n\n".format(jobid, vol_to_process, (time.time() - start)))
    return


def summarycomparision(dr1, dr2, svloc, dr1search=None, dr2search=None):
    """quick way to compare two summary outputs side by side, could be helpful for different masking parameters. dr2 should be the directory with more successfully completed files
    Inputs:
            dr1, dr2=outputdirectories of registration
            svloc=only path to save output
            dr#search=portion of file name unique to .pngs"""
    if dr1search != None:
        searchfor1=dr1search
    else:
        searchfor1="summary"
    if dr2search != None:
        searchfor2=dr2search
    else:
        searchfor2="summary"
    fl1=[]
    efl1=[]
    for (root, dir, files) in os.walk(dr1):
         for f in files:
             path = os.path.join(root, f)
             if os.path.exists(path) and searchfor1 in path:
                 fl1.append(path)
             elif os.path.exists(path) and "elastix.log" in path:
                 efl1.append(path)
    fl1.sort()


    fl2=[]
    efl2=[]
    for (root, dir, files) in os.walk(dr2):
         for f in files:
             path = os.path.join(root, f)
             if os.path.exists(path) and searchfor2 in path:
                 fl2.append(path)
             elif os.path.exists(path) and "elastix.log" in path:
                 efl2.append(path)
    fl2.sort()
    longdict={}
    shortdict={}
    if len(fl1) > len(fl2):
        fls=fl2; fll=fl1
        efls=efl2; efll=efl1
        shortdict[dr2]=fls; longdict[dr1]=fll
    elif len(fl2) >= len(fl1):
        fls=fl1; fll=fl2
        efls=efl1; efll=efl2
        shortdict[dr1]=fls; longdict[dr2]=fll

    badlst=[]
    font = cv2.FONT_HERSHEY_SIMPLEX
    color=3*(0,)
    tick=0
    for names in longdict.values()[0]:
        filename=names[names.rfind("GR"):names.rfind("_registration_results")]
        if [filename for subnames in fls if filename in subnames] == [filename for subnames in fll if filename in subnames]:
            print ("Found: {}".format(filename))
            im1=cv2.imread(names)
            cv2.putText(im1,longdict.keys()[0][longdict.keys()[0].rfind("/")+1:],(10,30), font, 1,color, 2)
            cv2.putText(im1,metricfinder([metric for metric in efll if filename in metric][0]), (470,30), font, 0.6, color, 2)
            im2=cv2.imread([sbnm for sbnm in fls if filename in sbnm][0])
            cv2.putText(im2,shortdict.keys()[0][shortdict.keys()[0].rfind("/")+1:],(10,30), font, 1,color, 2)#, cv2.LINE_AA)
            cv2.putText(im2,metricfinder([metric for metric in efls if filename in metric][0]), (470,30), font, 0.6, color, 2)
            im12=np.concatenate((im1, im2), axis=1)
            cv2.putText(im12,names[names.rfind("DREADDs/"):names.rfind("/"+searchfor1)],(100,535), font, 1.5,color, 2)#, cv2.LINE_AA)
            try:
                stack[tick,...]=im12
            except NameError:
                y,x,c=im12.shape
                stack=np.zeros((len(fll),y,x,c))
                stack[tick,...]=im12
            tick+=1
        else:
            badlst.append(names)
    if badlst != 0:
        for badim in badlst:
                filename=badim[badim.rfind("GR"):badim.rfind("_registration_results")]
                print("badlist: {}".format(badim))
                badim1=cv2.imread(badim)
                cv2.putText(badim1,longdict.keys()[0][longdict.keys()[0].rfind("/")+1:],(10,30), font, 1,color, 2)
                cv2.putText(badim1,metricfinder([metric for metric in efll if filename in metric][0]), (470,30), font, 0.6, color, 2)
                badim2=np.zeros((y,x/2,c)).astype("uint8")
                badim2[...]=255
                cv2.putText(badim2,"File not present for: {}".format(shortdict.keys()[0][shortdict.keys()[0].rfind("/")+1:]),(10,200), font,1,color, 3)#, cv2.LINE_AA)
                badim12=np.concatenate((badim1, badim2), axis=1)
                cv2.putText(badim12,badim[badim.rfind("DREADDs/"):badim.rfind("/"+searchfor1)],(100,535), font, 1.5,color, 2)
                stack[tick,...]=badim12
                tick+=1
    tifffile.imsave(os.path.join(svloc, dr1[dr1.rfind("/")+1:]+"_w_"+dr2[dr2.rfind("/")+1:]+".tif"), stack.astype("uint16"))
    print("Saved {}".format(os.path.join(svloc, dr1[dr1.rfind("/")+1:]+"_w_"+dr2[dr2.rfind("/")+1:]+".tif")))


def summarycombine(dr1, svlocname, filenamesearch=None):
    """quick way to combine summary, could be helpful for different masking parameters.
    Inputs:
            dr1=outputdirectorie of registration
            svlocname=path, filename, extension to save output"""
    if filenamesearch != None:
        searchfor=filenamesearch
    else:
        searchfor="summary"
    fl1=[]
    efl=[]
    for (root, dir, files) in os.walk(dr1):
         for f in files:
             path = os.path.join(root, f)
             if os.path.exists(path) and searchfor in path:
                 fl1.append(path)
             elif os.path.exists(path) and "elastix.log" in path:
                 efl.append(path)

    fl1.sort()
    font = cv2.FONT_HERSHEY_SIMPLEX
    color=3*(0,)
    tick=0
    for names in fl1:
        filename=names[len(dr1):names.rfind("/"+searchfor)]
        print ("Found: {}".format(filename))
        im1=cv2.imread(names)
        cv2.putText(im1,dr1[dr1.rfind("/")+1:],(10,30), font, 1,color, 2)
        cv2.putText(im1,names[names.rfind(dr1[-3:])+4:names.rfind("/"+searchfor)],(10,535), font, 1.5,color, 2)
        cv2.putText(im1,metricfinder([metric for metric in efl if filename in metric][0]), (470,30), font, 0.6, color, 2)
        try:
            stack[tick,...]=im1
        except NameError:
            y,x,c=im1.shape
            stack=np.zeros((len(fl1),y,x,c))
            stack[tick,...]=im1
        tick+=1
    tifffile.imsave(svlocname, stack.astype("uint16"))
    print("Saved {}".format(svlocname))


def metricfinder(dr):
    """Searches elastix filelog; returns final metric value as str"""
    f=open(dr, "r")
    s=mmap.mmap(f.fileno(), 0, access=mmap.ACCESS_READ)
    s.seek(s.rfind("Final metric value")); metric=s.readline(); metric=metric[:metric.rfind("\\")]
    s.close()
    return metric


def gridlineresizer(mv, **kwargs):
    """resizes gridline file to appropraite z dimenisions.
    Returns path to zcorrected stack
    Input:
       movingfile: atlas file used to scale gridlinefile, can be path, np object, or volume class
       savlocation: destination of file"""

    ###determine output image stack size
    if (mv.kind == "volume") == True:
        #mv_resampled=mv.resampled_for_elastix_vol
        gridlinefile=mv.packagedirectory+"/supp_files/gridlines.tif"
        mv=sitk.GetArrayFromImage(sitk.ReadImage(mv.resampled_for_elastix_vol))
        z,y,x=mv.shape
    elif (type(mv).__module__ == np.__name__) == True:
        z,y,x=mv.shape
        gridlinefile=kwargs["gridlinefile"]
    elif isinstance(mv, str):
        mv=sitk.GetArrayFromImage(sitk.ReadImage(mv))
        z,y,x=mv.shape
        gridlinefile=kwargs["gridlinefile"]
   ##load gridline file
    grdfl=tifffile.imread(gridlinefile)
    if len(grdfl.shape)==2:
        grdim=grdfl
    else:
        grdim = grdfl[0,:,:]
   ##########preprocess gridline file: Correcting for z difference between gridline file and atlas
    grdresized=cv2.resize(grdim, (x,y), interpolation=cv2.INTER_LINEAR) #interpolation=cv2.INTER_CUBIC) # #cv2.INTER_LINEAR, cv2.INTER_CUBIC, cv2.INTER_AREA
    grdresizedstck=np.zeros((z, y, x))
    for i in range(z):
        grdresizedstck[i,:,:]=grdresized
    return grdresizedstck.astype("uint8")
def gridcompare(svlc, reg_vol):
    gridfld=os.path.join(svlc, "tmpgridlinefilefortransformix")
    makedir(gridfld)
    tmpgridline=os.path.join(gridfld, "gridline.tif")
    gridlineresized=gridlineresizer(reg_vol)
    tifffile.imsave(tmpgridline, gridlineresized)
    print ("gridline resized to shape {}".format(gridlineresized.shape))
    return gridfld, tmpgridline



def combine_images(movie1, movie2, movie3, movie4, svlc, brainname):
    """Function used to take 4 movies and concatenate them together. It will also output a merged file of movie 2 with movie 4. Finally it will output an RMS video - but I don"t like this one all that much.
    All inputs are strings
    movie1=original non-registered/resampled movie.
    movie2=atlas
    movie3=tranformation gridline file
    movie4=final elastixs output"""
    savefilename="reg_visualization_"+brainname
    ####################LOAD ROUTINE
    m1=sitk.GetArrayFromImage(sitk.RescaleIntensity(sitk.ReadImage(movie1)))
    m2=sitk.GetArrayFromImage(sitk.RescaleIntensity(sitk.ReadImage(movie2)))
    m3=sitk.GetArrayFromImage(sitk.RescaleIntensity(sitk.ReadImage(movie3))) #don"t need to rescale gridline filter
    m4=sitk.GetArrayFromImage(sitk.RescaleIntensity(sitk.ReadImage(movie4)))
   ############## video 2 atop video 3
    z3, y3, x3 = m3.shape
    z2, y2, x2 = m2.shape
    xdif=x2-x3
    zdif=z2-z3
    if zdif < 0:
        m3=np.resize(m3, (z2, y3, x3)) #making m3 have the same z as atlas
    if xdif > 0:
        m23=np.hstack((m2,np.dstack((m3, np.zeros(z3, y3, xdif)))))
    elif xdif < 0:
        m23=np.hstack((np.dstack((m2, np.zeros(z3, y3, abs(xdif)))), m3))
    elif xdif == 0:
        m23=np.hstack((m2,m3))
    z23,y23,x23 = m23.shape
    z4,y4,x4 = m4.shape
    z1, y1, x1 = m1.shape
  ###combine video23 with video4
    ydif=y23-y4
    if ydif > 0:
        m234=np.dstack((m23, np.hstack((m4, np.zeros((z4, ydif, x4))))))
    elif ydif < 0:
        m234=np.dstack((np.hstack(m23, np.zeros((x23, abs(ydif), x23))), m4))
    elif ydif == 0:
        m234 = np.dstack((m23, m4))
    z234, y234, x234=m234.shape
  ####combine video1 with video234
    newydif=y234-y1
    if newydif > 0:
        m1=np.hstack((m1, np.zeros((z1, abs(newydif), x1))))
    elif newydif < 0:
        m234 = np.hstack((m234, np.zeros((z234, abs(newydif), x234))))
#    elif newydif < 0 == True:
#        m1_taller=np.hstack((np.))
    zdif=z1-z4
    if zdif > 0:
        m1234=np.dstack((m1,np.vstack((m234, np.zeros((zdif, y234, x234))))))
    elif zdif < 0:
        m1234=np.dstack((np.vstack((m1, np.zeros((abs(zdif), y1, x1)))), m234))
    elif zdif == 0:
        m1234=np.dstack((m1, m234))
    ##### Folder Creation and SAVE Routine
    makedir(os.path.join(svlc,"combinedmovies"))
    tifffile.imsave(os.path.join(svlc,"combinedmovies", "Concatenated_{}.tif".format(savefilename)), m1234.astype(np.uint8))
    print ("Completed Gridline Registration Visualization as {}".format(savefilename))
    return


# goal is to flatten 6 tiffs.

def flatten(dr, **kwargs):
    """Regex to create zplane dictionary of subdictionaries (keys=channels, values=single zpln list sorted by x and y).
    USED FOR RAW DATA FROM LVBT: 1-2 light sheets (each multipage tiff, where a page represents a horizontal foci)"""
    ####INPUTS:
    if dr == None:
        vols=kwargs["volumes"]
        reg_vol=[xx for xx in vols if xx.ch_type == "regch"][0]
        dr = reg_vol.indr
    outdr=kwargs["outputdirectory"]
    regexpression = kwargs["regexpression"]
    #regexpression=r"(.*)(?P<y>\d{2})(.*)(?P<x>\d{2})(.*C+)(?P<ls>[0-9]{1,2})(.*Z+)(?P<z>[0-9]{1,4})(.*r)(?P<ch>[0-9]{1,4})(.ome.tif)"
    fl=[os.path.join(dr, f) for f in os.listdir(dr) if "raw_DataStack" in f or "raw_RawDataStack" in f] #sorted for raw files
    reg=re.compile(regexpression)
    matches = list(map(reg.match, fl)) #matches[0].groups()
    ##find index of z,y,x,ch in a match str
    z_indx = matches[0].span("z")
    try:
        y_indx=matches[0].span("y")
        x_indx=matches[0].span("x")
        tiling = True
    except IndexError:
        y_indx = 1
        x_indx = 1
        tiling = False
    #ch_indx=matches[0].span("ch")
    ###determine number of channels, sheets, horizontal foci
    #chs=[]; [chs.append(matches[i].group("ch")[-2:]) for i in range(len(matches)) if matches[i].group("ch")[-2:] not in chs]
    chs=[]; [chs.append(matches[i].group("ch")[:]) for i in range(len(matches)) if matches[i].group("ch")[:] not in chs]
    zplns=[]; [zplns.append(matches[i].group("z")) for i in range(len(matches)) if matches[i].group("z") not in zplns]; zplns.sort()
    #zmx=max(zplns)
    with tifffile.TiffFile(os.path.join(dr, "".join(matches[0].groups()))) as tif:
        if len(tif.pages[0].shape)==3: hf,y,x=tif.pages[0].shape
        elif len(tif.pages[0].shape)==2:
            hf=len(tif.pages) #number of horizontal foci
            y,x=tif.pages[0].shape
        tif.close()
    ##make dct consisting of each channel sorted by z plane, then in xy order (topleft-->top right to bottom-right), then sorted for ls(L then R)
    zdct={}; chdct={}
    #chs=[str(ch).zfill(4) for ch in chs]; bd_dct={}
    bd_dct={}
    for ch in chs:
        lst=[]
        try:
            [lst.append("".join(match.groups())) for num,match in enumerate(matches) if ch in match.group("ch")]
        except:
            [lst.append("".join(match.groups())) for num,match in enumerate(matches) if ch in match.group("ch")]
        try:
            srtd=sorted(lst, key=lambda a: (a[z_indx[0]:z_indx[1]],  a[ls_indx[0]:ls_indx[1]], a[y_indx[0]:y_indx[1]], a[x_indx[0]:x_indx[1]])) #sort by z, then ls, then x, then y
        except NameError:
            srtd=sorted(lst, key=lambda a: (a[z_indx[0]:z_indx[1]],  a[y_indx[0]:y_indx[1]], a[x_indx[0]:x_indx[1]])) #sort by z, then x, then y
        if tiling:
            ytile=int(max([yy for f in lst for yy in f[y_indx[0]:y_indx[1]]]))+1 #automatically find the number of tiles
            xtile=int(max([xx for f in lst for xx in f[x_indx[0]:x_indx[1]]]))+1 #automatically find the number of tiles
        elif not tiling:
            ytile = 1
            xtile = 1
        try:
            ls_indx=matches[0].span("ls")
            lsheets=[]; [lsheets.append(matches[i].group("ls")[-2:]) for i in range(len(matches)) if matches[i].group("ls")[-2:] not in lsheets]
            lsheets=int(max([lsh for f in lst for lsh in f[ls_indx[0]:ls_indx[1]]]))+1 #automatically find the number of light sheets
            intvl=xtile*ytile*lsheets
        except IndexError:
            intvl=xtile*ytile
            lsheets=1
        ################find z plns missing tiles and pln to badlst
        test_matches = list(map(reg.match, srtd))
        new_z_indx=test_matches[0].span("z")
        z_lst=[xx[new_z_indx[0]:new_z_indx[1]] for xx in srtd]
        counter=collections.Counter(z_lst)
        bd_dct[ch]=[xx for xx in counter if counter[xx] != intvl]
        ############sort by plane
        ttdct={}
        for plns in zplns:
            try:
                tmp=[]; [tmp.append(xx) for xx in srtd if "Z"+plns in xx]; tmp=sorted(tmp, key=lambda a: (a[z_indx[0]:z_indx[1]],  a[ls_indx[0]:ls_indx[1]], a[y_indx[0]:y_indx[1]], a[x_indx[0]:x_indx[1]]))
            except NameError:
                tmp=[]; [tmp.append(xx) for xx in srtd if "Z"+plns in xx]; tmp=sorted(tmp, key=lambda a: (a[z_indx[0]:z_indx[1]],  a[y_indx[0]:y_indx[1]], a[x_indx[0]:x_indx[1]]))
            ttdct[plns]=tmp
        ########key=channel; values=dictionary of tiles/pln
        chdct[ch[-2:]]=ttdct
    ###max zpln
    mx_zpln=max([len(chdct[xx]) for xx in chdct])
    ###zdct: keys=pln, values=dictionary of channels with subvalues being tiles/lightsheet
    for xx in range(mx_zpln-1):
        tmpdct={}
        [tmpdct.update(dict([(chann, chdct[chann][str(xx).zfill(4)])])) for chann in chdct]
        zdct[str(xx).zfill(4)]=tmpdct
    ################################################################################################
    ###REMOVE ENTIRE PLANE, ALL CHANNEL WHERE THERE IS MISSING FILES; THIS MIGHT NEED TO BE REVISITED
    for chann in bd_dct:
        if len(bd_dct[chann]) > 0:
            for bdpln in bd_dct[chann]:
                try:
                    del zdct[bdpln]
                except:
                    print("bad plane {} missing".format(bdpln))
    ################################################################################################
    chs=[ch[-2:] for ch in chs]
    ###check to see if all channels have the same length, if not it means LVBT messed up
    if max([len(bd_dct[xxx]) for xxx in bd_dct]) >0:
        writer(outdr, "STEP 0\n")
        writer(outdr, "Unequal_number_of_planes_per_channel_detected...seriously WTF LVBT.\n", flnm="unequal_number_of_planes_per_channel_detected.txt")
        writer(outdr, "\nChannels and planes that were bad {}".format(bd_dct), flnm="unequal_number_of_planes_per_channel_detected.txt")
        writer(outdr, "\nBad planes have been removed from ALL CHANNELS", flnm="unequal_number_of_planes_per_channel_detected.txt")
    #####find full size dimensions in zyx
    writer(outdr, "{} *Complete* Zplanes found for {}\n".format(len(zdct.keys()), dr))
    writer(outdr, "Checking for bad missing files:\n     Bad planes per channel:\n     {}\n".format(bd_dct))
    writer(outdr, "{} Channels found\n".format(len(zdct["0001"].keys())))
    writer(outdr, "{}x by {}y tile scan determined\n".format(xtile, ytile))
    writer(outdr, "{} Light Sheet(s) found. {} Horizontal Focus Determined\n\n".format(lsheets, hf))
    return dict([("zchanneldct", zdct), ("xtile", xtile), ("ytile", ytile), ("channels", chs), ("lightsheets", lsheets), ("horizontalfoci", hf), ("fullsizedimensions", (len(zdct.keys()),(y*ytile),(x*xtile)))])

def flatten_vol(volume_class):
    """Regex to create zplane dictionary of subdictionaries (keys=channels, values=single zpln list sorted by x and y).
    USED FOR RAW DATA FROM LVBT: 1-2 light sheets (each multipage tiff, where a page represents a horizontal foci)"""
    ####INPUTS:
    outdr=volume_class.outdr
    dr=volume_class.indr
    regexpression=volume_class.regex
    ch=str(volume_class.channel).zfill(4)
    #regexpression=r"(.*)(?P<y>\d{2})(.*)(?P<x>\d{2})(.*C+)(?P<ls>[0-9]{1,2})(.*Z+)(?P<z>[0-9]{1,4})(.*r)(?P<ch>[0-9]{1,4})(.ome.tif)"
    fl=[f for f in os.listdir(dr) if "raw_DataStack" in f] #sorted for raw files
    reg=re.compile(regexpression)
    matches = list(map(reg.match, fl)) #matches[0].groups()
    ##find index of z,y,x,ch in a match str
    z_indx=matches[0].span("z")
    y_indx=matches[0].span("y")
    x_indx=matches[0].span("x")
    ch_indx=matches[0].span("ch")
    ###determine number of channels, sheets, horizontal foci
    #chs=[]; [chs.append(matches[i].group("ch")[-2:]) for i in range(len(matches)) if matches[i].group("ch")[-2:] not in chs]
    chs=[]; [chs.append(matches[i].group("ch")[:]) for i in range(len(matches)) if matches[i].group("ch")[:] not in chs]
    assert str(volume_class.channel).zfill(4) in chs
    zplns=[]; [zplns.append(matches[i].group("z")) for i in range(len(matches)) if matches[i].group("z") not in zplns]; zplns.sort()
    zmx=max(zplns)
    with tifffile.TiffFile(os.path.join(dr, "".join(matches[0].groups()))) as tif:
        hf=len(tif.pages) #number of horizontal foci
        y,x=tif.pages[0].shape
        tif.close()
    ##make dct consisting of each channel sorted by z plane, then in xy order (topleft-->top right to bottom-right), then sorted for ls(L then R)
    zdct={}; chdct={}
    #chs=[str(ch).zfill(4) for ch in chs]; bd_dct={}
    bd_dct={}
    for ch in chs:
        lst=[]
        try:
            [lst.append("".join(match.groups())) for num,match in enumerate(matches) if ch in match.group("ch")]
        except:
            [lst.append("".join(match.groups())) for num,match in enumerate(matches) if ch in match.group("ch")]
        try:
            srtd=sorted(lst, key=lambda a: (a[z_indx[0]:z_indx[1]],  a[ls_indx[0]:ls_indx[1]], a[y_indx[0]:y_indx[1]], a[x_indx[0]:x_indx[1]])) #sort by z, then ls, then x, then y
        except NameError:
            srtd=sorted(lst, key=lambda a: (a[z_indx[0]:z_indx[1]],  a[y_indx[0]:y_indx[1]], a[x_indx[0]:x_indx[1]])) #sort by z, then x, then y
        ytile=int(max([yy for f in lst for yy in f[y_indx[0]:y_indx[1]]]))+1 #automatically find the number of tiles
        xtile=int(max([xx for f in lst for xx in f[x_indx[0]:x_indx[1]]]))+1 #automatically find the number of tiles
        try:
            ls_indx=matches[0].span("ls")
            lsheets=[]; [lsheets.append(matches[i].group("ls")[-2:]) for i in range(len(matches)) if matches[i].group("ls")[-2:] not in lsheets]
            lsheets=int(max([lsh for f in lst for lsh in f[ls_indx[0]:ls_indx[1]]]))+1 #automatically find the number of light sheets
            intvl=xtile*ytile*lsheets
        except IndexError:
            intvl=xtile*ytile
            lsheets=1
        ################find z plns missing tiles and pln to badlst
        test_matches = list(map(reg.match, srtd))
        new_z_indx=test_matches[0].span("z")
        z_lst=[xx[new_z_indx[0]:new_z_indx[1]] for xx in srtd]
        counter=collections.Counter(z_lst)
        bd_dct[ch]=[xx for xx in counter if counter[xx] != intvl]
        ############sort by plane
        ttdct={}
        for plns in zplns:
            try:
                tmp=[]; [tmp.append(xx) for xx in srtd if "Z"+plns in xx]; tmp=sorted(tmp, key=lambda a: (a[z_indx[0]:z_indx[1]],  a[ls_indx[0]:ls_indx[1]], a[y_indx[0]:y_indx[1]], a[x_indx[0]:x_indx[1]]))
            except NameError:
                tmp=[]; [tmp.append(xx) for xx in srtd if "Z"+plns in xx]; tmp=sorted(tmp, key=lambda a: (a[z_indx[0]:z_indx[1]],  a[y_indx[0]:y_indx[1]], a[x_indx[0]:x_indx[1]]))
            ttdct[plns]=tmp
        ########key=channel; values=dictionary of tiles/pln
        chdct[ch[-2:]]=ttdct
    ###max zpln
    mx_zpln=max([len(chdct[xx]) for xx in chdct])
    ###zdct: keys=pln, values=dictionary of channels with subvalues being tiles/lightsheet
    for xx in range(mx_zpln-1):
        tmpdct={}
        [tmpdct.update(dict([(chann, chdct[chann][str(xx).zfill(4)])])) for chann in chdct]
        zdct[str(xx).zfill(4)]=tmpdct
    ################################################################################################
    ###REMOVE ENTIRE PLANE, ALL CHANNEL WHERE THERE IS MISSING FILES; THIS MIGHT NEED TO BE REVISITED
    print ("Len of bd_dct = {}".format(len(bd_dct)))
    for chann in bd_dct:
        if len(bd_dct[chann]) > 0:
            for bdpln in bd_dct[chann]:
                del zdct[bdpln]
    ################################################################################################
    chs=[ch[-2:] for ch in chs]
    ###check to see if all channels have the same length, if not it means LVBT messed up
    if max([len(bd_dct[xxx]) for xxx in bd_dct]) >0:
        writer(outdr, "Unequal_number_of_planes_per_channel_detected...seriously WTF LVBT.\n", flnm="unequal_number_of_planes_per_channel_detected.txt")
        writer(outdr, "\nChannels and planes that were bad {}".format(bd_dct), flnm="unequal_number_of_planes_per_channel_detected.txt")
        writer(outdr, "\nBad planes have been removed from ALL CHANNELS", flnm="unequal_number_of_planes_per_channel_detected.txt")
    #####find full size dimensions in zyx
    print ("{} *Complete* Zplanes found".format(len(zdct.keys())))
    print ("{} Channels found".format(len(zdct["0000"].keys())))
    print ("{}x by {}y tile scan determined".format(xtile, ytile))
    print ("{} Light Sheet(s) found. {} Horizontal Focus Determined".format(lsheets, hf))
    return dict([("zchanneldct", zdct), ("xtile", xtile), ("ytile", ytile), ("channels", chs), ("lightsheets", lsheets), ("horizontalfoci", hf), ("fullsizedimensions", (len(zdct.keys()),(y*ytile),(x*xtile)))])


def flatten_stitcher(cores, outdr, ovlp, xtile, ytile, zpln, dct, lightsheets, **kwargs):
    """return numpy arrays of     """
    ###zpln="0500"; dct=zdct[zpln]
    #easy way to set ch and zplnlst
    ["stitching for ch_{}".format(ch[-2:]) for ch, zplnlst in dct.items()] #cheating way to set ch and zplnlst

    ###dim setup
    ch = list(dct.keys())[0]
    zplnlst = dct[list(dct.keys())[0]] #changed for py37
    ydim, xdim = cv2.imread(zplnlst[0], -1).shape
    xpxovlp = int(ovlp*xdim)
    ypxovlp = int(ovlp*ydim)
    tiles = len(zplnlst) ##number of tiles

    #check for optional kwargs if not passed explicitly:
    blndtype = kwargs["blendtype"] if "blendtype" in kwargs else False
    intensitycorrection = kwargs["intensitycorrection"] if "intensitycorrection" in kwargs else False
    blendfactor = kwargs["blendfactor"] if "blendfactor" in kwargs else 4

    ### blending setup
    if blndtype == "linear":
        alpha = np.tile(np.linspace(0, 1, num=int(xpxovlp)), (ydim, 1)) ###might need to change 0-1 to 0-255?
        yalpha = np.swapaxes(np.tile(np.linspace(0, 1, num=int(ypxovlp)), (int(xdim+((1-ovlp)*xdim*(xtile-1))),1)), 0, 1)
    elif blndtype == "sigmoidal":
        alpha = np.tile(scipy.stats.logistic.cdf(np.linspace(-blendfactor, blendfactor,num=int(xpxovlp))), (ydim, 1)) ###might need to change 0-1 to 0-255?
        yalpha = np.swapaxes(np.tile(scipy.stats.logistic.cdf(np.linspace(-blendfactor, blendfactor, num=int(ypxovlp))), (int(xdim+((1-ovlp)*xdim*(xtile-1))),1)), 0, 1)
    elif blndtype == False or blndtype == None: #No blending: generate np array with 0 for half of overlap and 1 for other.
        alpha = np.zeros((ydim, xpxovlp)); alpha[:, xpxovlp/2:] = 1
        yalpha = np.zeros((int(ypxovlp), int((xdim+((1-ovlp)*xdim*(xtile-1)))))); yalpha[int(ypxovlp/2):, :] = 1
    else:
        alpha = np.tile(scipy.stats.logistic.cdf(np.linspace(-blendfactor, blendfactor,num=int(xpxovlp))), (ydim, 1)) ###might need to change 0-1 to 0-255?
        yalpha = np.swapaxes(np.tile(scipy.stats.logistic.cdf(np.linspace(-blendfactor, blendfactor, num=int(ypxovlp))), (int(xdim+((1-ovlp)*xdim*(xtile-1))),1)), 0, 1)

    #process

    if len(dct)<=1:
        stitchlst=[flatten_xystitcher(xdim, ydim, xtile, ytile, ovlp, xpxovlp, ypxovlp, tiles, alpha, yalpha, zpln, ch[-2:], zplnlst, intensitycorrection, lightsheets) for ch, zplnlst in dct.items()]
    else:
        ##parallel processing
        p = mp.Pool(cores)
        iterlst = [(xdim, ydim, xtile, ytile, ovlp, xpxovlp, ypxovlp, tiles, alpha, yalpha, zpln, ch[-2:], zplnlst, intensitycorrection, lightsheets) for ch, zplnlst in dct.items()]
        try:
            stitchlst = p.starmap(flatten_xystitcher, iterlst)
        except ValueError as e:
            raise ValueError("{}\n...This is likely due to not setting the correct 'tiling_overlap' in run_tracing.py params dictionary".format(e))
        p.terminate(); del iterlst
    stitchdct = {}; [stitchdct.update(i) for i in stitchlst]
    
    return stitchdct
###

def flatten_xystitcher(xdim, ydim, xtile, ytile, ovlp, xpxovlp, ypxovlp, 
                       tiles, alpha, yalpha, zpln, ch, zplnlst, intensitycorrection, lightsheets):
    """
    #helper function for parallelization of stitcher; this version used with raw data.
    Currently, independent calculations for each channels mean pixel shift per tile. 
    Ideally this compensates for variable photobleaching in different channels
    """
    
    #list of images going Lshet01, Rsheet01, Lsheet02
    lst=[tifffile.imread("".join(im), multifile=False) for im in zplnlst] #cleaned up 201802 - if breaks change imread as float64
    
    #max project HF
    if len(lst[0].shape) > 2: lst=[np.amax(xx, 0) for xx in lst]
    
    #account for one or two light sheets
    if lightsheets == 2:
        l_ls_imlst=lst[:int(len(lst)/2)] #left ims
        r_ls_imlst=lst[int(len(lst)/2):] #r ims
        lsts=[l_ls_imlst, r_ls_imlst]
    else:
        lsts=[lst]
    
    #stitch
    stitchedlst=[]
    if intensitycorrection:
        for lst in lsts:
            warnings.warn("depreciated", DeprecationWarning)
            ytick=0; xtick=0; tick=0
            while ytick < ytile:
                while xtick < xtile:
                    im=lst[tick]; tick+=1
                    if xtick != 0:
                        #correct for overall image intensity difference by shifting higher mean overlapping pixels towards smaller
                        l_im_ovlp=xfrm[:,-(xpxovlp):]; r_im_ovlp=im[:,:(xpxovlp)]
                        l_im_mean=np.mean(l_im_ovlp.ravel()); r_im_mean=np.mean(r_im_ovlp.ravel()); mean_dif=abs(r_im_mean - l_im_mean)
                        #move the means two images closer to each other
                        if l_im_mean < r_im_mean:
                            im=im-mean_dif; im=im.clip(min=0)
                        elif r_im_mean < l_im_mean:
                            xfrm=xfrm - mean_dif; xfrm=xfrm.clip(min=0)
                        ##stitch
                        xblend = (xfrm[:,-(xpxovlp):]*(1-alpha) + im[:,:(xpxovlp)] * (alpha))    #1-alhpa alpha
                        xfrm=np.concatenate((xfrm[:,:-(xpxovlp)], xblend, im[:,xpxovlp:]), axis=1)
                    else:
                        xfrm=im
                    xtick+=1
                xtick=0
                if ytick !=0:
                    #correct for overall image intensity difference by shifting higher mean overlapping pixels towards smaller
                    t_im_ovlp=xyfrm[-(ypxovlp):,:]; b_im_ovlp=xfrm[:ypxovlp,:]
                    t_im_mean=np.mean(t_im_ovlp.ravel()); b_im_mean=np.mean(b_im_ovlp.ravel()); mean_dif=abs(t_im_mean - b_im_mean)
                    #move the means two images closer to each other
                    if t_im_mean < b_im_mean:
                        xfrm=xfrm-mean_dif; xfrm=xfrm.clip(min=0)
                    elif b_im_mean < t_im_mean:
                        xyfrm=xyfrm - mean_dif; xyfrm=xyfrm.clip(min=0)
                    #stitch
                    yblend=(xyfrm[-(ypxovlp):,:]*(1-yalpha) + xfrm[:(ypxovlp),:] * yalpha)
                    xyfrm=np.concatenate((xyfrm[:-(ypxovlp),:], yblend, xfrm[ypxovlp:,:]), axis=0)
                else:
                    xyfrm=xfrm
                ytick+=1
            stitchedlst.append(xyfrm)
            
    elif not intensitycorrection:
        for lst in lsts:
            warnings.warn("depreciated", DeprecationWarning)
            ytick=0; xtick=0; tick=0
            while ytick < ytile:
                while xtick < xtile:
                    im=lst[tick]
                    tick+=1
                    if xtick != 0:
                        ##stitch
                        xblend = (xfrm[:,-(xpxovlp):]*(1-alpha) + im[:,:(xpxovlp)] * (alpha))    #1-alhpa alpha
                        xfrm=np.concatenate((xfrm[:,:-(xpxovlp)], xblend, im[:,xpxovlp:]), axis=1)
                    else:
                        xfrm=im
                    xtick+=1
                xtick=0
                if ytick !=0:
                    #stitch
                    yblend=(xyfrm[-(ypxovlp):,:]*(1-yalpha) + xfrm[:(ypxovlp),:] * yalpha)
                    xyfrm=np.concatenate((xyfrm[:-(ypxovlp),:], yblend, xfrm[ypxovlp:,:]), axis=0)
                else:
                    xyfrm=xfrm
                ytick+=1
            stitchedlst.append(xyfrm)
    try:
        del l_ls_imlst, r_ls_imlst, lsts
    except:
        pass
    if lightsheets==2:
        ydm,xdm=xyfrm.shape
        n_alpha=np.tile(scipy.stats.logistic.cdf(np.linspace(-10, 10,num=xdm)), (ydm, 1))
        s_blend = (stitchedlst[0]*(1-n_alpha) + stitchedlst[1]* (n_alpha))
        return dict([(ch, s_blend)])
    else:
        return dict([(ch, stitchedlst[0])])