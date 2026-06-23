import numpy as np
import pandas as pd
import h5py
import matplotlib.pyplot as plt
import os,sys,random
import illustris_python as il
from astropy.cosmology import FlatLambdaCDM
import time

def get_subhalo_flags_snap(basePath,outPath,sim,snap):
    '''Create dataframe of all subhaloes with their flags.'''
    flag_info_file = f'{outPath}/Catalogues/Flags/{sim}_SubhaloFlag_{snap:03}.csv'
    if os.access(flag_info_file,0):
        return pd.read_csv(flag_info_file)
    fields = ['SubhaloMassType','SubhaloFlag']
    ptNumStars = il.snapshot.partTypeNum('stars')
    ptNumGas = il.snapshot.partTypeNum('gas')
    ptNumDM = il.snapshot.partTypeNum('dm')
    df_out = pd.DataFrame()
    cat = il.groupcat.loadSubhalos(basePath,snap,fields=fields)
    subs = np.arange(cat['count'],dtype=int)
    df_out['SubfindID']=subs
    df_out['SubhaloMassType_stars']=cat['SubhaloMassType'][:,ptNumStars]
    df_out['SubhaloFlag']=cat['SubhaloFlag']
    df_out.to_csv(flag_info_file,index=False)
    return pd.read_csv(flag_info_file)

def main():
    
    from astropy.cosmology import Planck15 as cosmo
    sim = os.getenv('SIM')

    basePath = f'/lustre/work/connor.bottrell/Simulations/IllustrisTNG/{sim}/output'
    outPath = f'/lustre/work/connor.bottrell/Simulations/IllustrisTNG/Scripts/Mergers/illustris_mergers'
    
    snaps = np.arange(100,dtype=int)
    task_idx = int(sys.argv[1])
    get_subhalo_flags_snap(basePath,outPath,sim,snaps[task_idx])
    

if __name__=='__main__':

    main()
    
