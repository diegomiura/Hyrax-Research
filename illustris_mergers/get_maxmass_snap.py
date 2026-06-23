import numpy as np
import pandas as pd
import h5py
import matplotlib.pyplot as plt
import os,sys,random
import illustris_python as il
import time

# def get_subhalos(basePath,snap,cosmo,mstar_lower=0,mstar_upper=np.inf):
#     '''
#     mstar_lower and mstar_upper have log10(Mstar/Msun) units and are physical (i.e. Msun, not Msun/h).
#     '''
#     little_h = cosmo.H0.value/100.
#     ptNumStars = il.snapshot.partTypeNum('stars')
#     fields = ['SubhaloMassType','SubhaloFlag']
#     subs = il.groupcat.loadSubhalos(basePath,snap,fields=fields)

#     mstar = subs['SubhaloMassType'][:,ptNumStars]
#     flags = subs['SubhaloFlag']
#     subs = np.arange(subs['count'],dtype=int)

#     # convert to units used by TNG (1e10 Msun/h)
#     mstar_lower = 10**(mstar_lower)/1e10*little_h
#     mstar_upper = 10**(mstar_upper)/1e10*little_h
#     subs = subs[(flags!=0)*(mstar>=mstar_lower)*(mstar<=mstar_upper)]

#     return subs,mstar[subs]

def get_subhalos(basePath,outPath,sim,snap,cosmo,
                 mstar_lower=0,mstar_upper=np.inf):
    '''
    mstar_lower and mstar_upper have log10(Mstar/Msun) units and are physical 
    (i.e. not typical Msun/h simulation units).
    '''
    little_h = cosmo.H0.value/100.
    # full set of subhalos for snapshot (+ flags & mstar)
    df_subs = get_subhalo_flags_snap(basePath,outPath,sim,snap)
    mstar = df_subs['SubhaloMassType_stars']
    flags = df_subs['SubhaloFlag']
    subs  = df_subs['SubfindID']
    # convert limits to units used by TNG (1e10 Msun/h)
    mstar_lower = 10**(mstar_lower)/1e10*little_h
    mstar_upper = 10**(mstar_upper)/1e10*little_h
    # screen for flagged subhalos and mass limits
    subs = subs[(flags!=0) & (mstar>=mstar_lower) & (mstar<=mstar_upper)]
    return df_subs.iloc[subs]

def max_past_mass_limit_snap(basePath,outPath,fileName,sim,snap,cosmo,limit=0.5):
    '''
    Get dataframe of maximum progenitor masses in [limit] Gyr for all subhalos in snapshot.
    
    Only consider subhalos in with non-zero masses ()
    '''
    out_file = f'{outPath}/Catalogues/MassMax/{fileName}'
    if os.access(out_file,0):
        return pd.read_csv(out_file)
    df_subs = get_subhalos(basePath,outPath,sim,snap,cosmo,mstar_lower=0,mstar_upper=np.inf)
    for mass_def in ['SubhaloMassType','SubhaloMassInRadType']:
        max_masses = np.array([max_past_mass_limit(
            basePath,outPath,sim,snap,sub,cosmo,limit=limit,
            mass_def=mass_def) for sub in df_subs['SubfindID']])
        df_subs[f'Max{mass_def}_stars']=max_masses[:,0]
        df_subs[f'SnapMax{mass_def}_stars']=max_masses[:,1].astype(int)
    df_subs.to_csv(out_file,index=False)
    return pd.read_csv(out_file)
    
def max_past_mass_limit(basePath,outPath,sim,snap,sub,cosmo,limit=0.5,
                        index=0,mass_def='SubhaloMassType'):
    '''
    Get maximum past stellar mass in the past [limit] Gyr along MPB. Also return the snapshot in which the maximum mass is found.
    
    Sometimes a the progenitor of a galaxy is skipped in SubLink when it passes close to other objects in a snapshot. This can create a cavity in the snapshot list. For example, see the link between halos 6 and 7 in the SubLink description on the TNG data access page. This code accounts for these skips.
    
    mass_def sets the definition of mass for the purposes of these calculations (e.g. "SubhaloMassType" or "SubhaloMassInRadType").
    '''
    partNum = il.snapshot.partTypeNum('stars')
    time_dict = get_snap_time_dict(basePath,outPath,sim,cosmo)
    fields = ['SubhaloID','MainLeafProgenitorID',mass_def,'SnapNum','SubfindID']
    tree = il.sublink.loadTree(basePath,snap,sub,fields=fields)
    try:
        # identify all subhalos with the same main leaf progenitor as the root
        mpb_mask = tree['MainLeafProgenitorID']==tree['MainLeafProgenitorID'][index]
        mpb_snaps = tree['SnapNum'][mpb_mask]
        mpb_times = [time_dict[f'{mpb_snap:03}'] for mpb_snap in mpb_snaps]
        mpb_mass = tree[mass_def][mpb_mask,partNum]
        # only consider progenitors from snapshots in [limit] Gyr (including root)
        branch_limit = sum((mpb_times[0]-mpb_times)<limit)
        idx_max = np.argmax(mpb_mass[:branch_limit])
        return [mpb_mass[:branch_limit][idx_max],mpb_snaps[:branch_limit][idx_max]]
    except:
        # no tree exists or tree broken
        return [-1,-1]

def get_snap_time_dict(basePath,outPath,sim,cosmo):
    '''
    Create dictionary of snapshot times for easy use.
    '''
    filename = f'{outPath}/Catalogues/Misc/{sim}_SnapInfo.npz'
    if os.access(filename,0):
        return np.load(f'Catalogues/Misc/{sim}_SnapInfo.npz',
                       allow_pickle=True)['time_dict'][()]
    snap_df = get_snapshot_df(basePath,outPath,sim,cosmo)
    snap_times = snap_df['Age'].values
    snaps = snap_df['Snapshot'].values
    time_dict = {f'{snaps[i]:03}':snap_times[i] for i in range(len(snap_times))}
    np.savez(f'Catalogues/Misc/{sim}_SnapInfo.npz',time_dict = time_dict)
    return np.load(f'Catalogues/Misc/{sim}_SnapInfo.npz',
                   allow_pickle=True)['time_dict'][()]

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
    
def get_snapshot_df(basePath,outPath,sim,cosmo):
    '''Create csv with redshifts,times for each snapshot.'''
    snap_info_file = f'{outPath}/Catalogues/Misc/{sim}_SnapInfo.csv'
    if os.access(snap_info_file,0):
        return pd.read_csv(snap_info_file)
    snaps = np.arange(100,dtype=int)
    redshifts = np.zeros(len(snaps))
    df = pd.DataFrame({'Snapshot':snaps,})
    for i,snap in enumerate(snaps):
        hdr = il.groupcat.loadHeader(basePath,snap)
        redshifts[i] = hdr['Redshift']
    df['Redshift']=redshifts
    df['ScaleFactor']=1/(1+redshifts)
    df['Age']=cosmo.age(redshifts).value
    df['LookbackTime']=cosmo.lookback_time(redshifts).value
    df.to_csv(snap_info_file,index=False)
    return pd.read_csv(snap_info_file)

def get_merger_info():
    
    return

def main():
    
    from astropy.cosmology import Planck15 as cosmo
    sim = os.getenv('SIM')
    snap = int(os.getenv('SNAP'))

    basePath = f'/lustre/work/connor.bottrell/Simulations/IllustrisTNG/{sim}/output'
    outPath = f'/lustre/work/connor.bottrell/Simulations/IllustrisTNG/Scripts/Mergers/illustris_mergers'

    fileName = f'{sim}_MassMaxInf_{snap:03}.csv'

    max_past_mass_limit_snap(basePath,outPath,fileName,
                             sim,snap,cosmo,limit=np.inf)

    
if __name__=='__main__':

    main()
