import numpy as np
import pandas as pd
import h5py
import matplotlib.pyplot as plt
import os,sys,random
import illustris_python as il
import time

def get_subhalos(basePath,outPath,sim,snap,cosmo,
                 mstar_lower=0,mstar_upper=np.inf):
    '''
    mstar_lower and mstar_upper have log10(Mstar/Msun) units and are physical (i.e. not typical Msun/h simulation units).
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

def max_past_mass_limit_snap(basePath,outPath,sim,snap,cosmo,limit=0.5):
    '''
    Get dataframe of maximum progenitor masses in [limit] Gyr for all subhalos in snapshot.
    
    Only consider subhalos in with non-zero masses and whose cosmological flag is 1.
    '''
    out_file = f'{outPath}/Catalogues/MassMax/{sim}_MassMaxInf_{snap:03}.csv'
    if os.access(out_file,0):
        return pd.read_csv(out_file)
    df_subs = get_subhalos(basePath,outPath,sim,snap,cosmo,
                           mstar_lower=0,mstar_upper=np.inf)
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
    Get maximum past stellar mass in the past [limit] Gyr along MPB.
    
    Sometimes a the progenitor of a galaxy is skipped in SubLink when it passes close to other objects in a snapshot. This can create a cavity in the snapshot list. For example, see the link between halos 6 and 7 in the SubLink description on the TNG data access page. This code accounts for these skips.
    
    mass_def is the sets the definition of mass for the purposes of these calculations (e.g. "SubhaloMassType" or "SubhaloMassInRadType").
    '''
    # check if non-cosmological subhalo (e.g. stellar clump)
    rec = il.groupcat.loadSingle(basePath,snap,subhaloID=sub)
    flag = rec['SubhaloFlag']
    if flag==False:
        return [-1,-1]
        
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
        return [mpb_mass[:branch_limit][idx_max],
                mpb_snaps[:branch_limit][idx_max]]
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
    df['ScaleFactor']=1./(1+redshifts)
    df['Age']=cosmo.age(redshifts).value
    df['LookbackTime']=cosmo.lookback_time(redshifts).value
    df.to_csv(snap_info_file,index=False)
    return pd.read_csv(snap_info_file)

def get_root_descendant_tree(basePath,snap,sub):
    '''
    Get the full tree (from root descendant) for a given SnapNum and SubfindID.
    '''
    tree = il.sublink.loadTree(
        basePath,snap,sub,
        fields=['SnapNum','SubfindID','SubhaloID'],
        onlyMDB=True
    )

    fields = [
        'SubhaloID', 'MainLeafProgenitorID', 'NextProgenitorID', 
        'SubhaloMassType', 'SubhaloMassInRadType', 'SnapNum', 'SubfindID', 
        'RootDescendantID', 'DescendantID', 'FirstProgenitorID',
    ]
    
    if tree == None:
        rd_id = None
        rd_snap = snap
        rd_sub = sub
        tree = {field:None for field in fields}
        
    elif len(tree['SubhaloID'])>0:
        rd_id = tree['SubhaloID'][0]
        rd_snap = tree['SnapNum'][0]
        rd_sub = tree['SubfindID'][0]
        # get full tree of root descendant with all fields
        tree = il.sublink.loadTree(
            basePath,rd_snap,rd_sub,
            fields=fields,
        )
    else:
        rd_snap = snap
        rd_sub = sub
        # get full tree of root descendant with all fields
        tree = il.sublink.loadTree(
            basePath,rd_snap,rd_sub,
            fields=fields,
        )
        rd_idx = np.argwhere(
            (tree['SnapNum']==snap)*(tree['SubfindID']==sub))[0][0]
        rd_id = tree['SubhaloID'][rd_idx]
    return tree

def get_root_descendant_tree_worker(args):
    '''
    Multiprocessing worker for getting root descendant tree.
    Returns as dictionary element d['{sub}'] = tree.
    '''
    basePath,snap,sub = args
    return(f'{sub}',get_root_descendant_tree(basePath,snap,sub))

def get_merger_history(
    basePath,outPath,sim,snap,sub,cosmo,mu_lower=0.1,mu_upper=1.0,
    mass_def='SubhaloMassType',limit=0.5,np_lim=np.inf,tree=None):
    '''
    Identify mergers along main progenitor branch within mass ratio limits: mu_lower <= mu <= mu_upper. Mass ratios are defined such that Mcomp/Mprim < 1.0 always (accounting also for mass histories).
    
    The mass ratio is determined at the snapshot in which the next progenitor mass is greatest (for both the main progenitor and next progenitor) to limit the effects of numerical stripping. The mass history is limited to [limit] Gyr from a given next progenitor to limit the effects of physical stripping (Patton+2020 and Hani+2021).
    
    Only consider next progenitors who are less than [np_lim] pointers away from the main progenitor branch. Searching the full breadth can be lenthy for large halos.
    '''
    partNumStars = il.snapshot.partTypeNum('stars')
    partNumGas = il.snapshot.partTypeNum('gas')
    
    if tree is None:
        tree = get_root_descendant_tree(
            basePath,snap,sub
        )
        
    if tree['SubhaloID'] is None:
        # tree broken
        tree_flag = 1
        merger_info = {

            'PastSnapNum' : -1,
            'PastSubfindID' : -1,
            'PastMassRatio' : -1,

            'PastMainProgenitorSnapNum' : -1,
            'PastMainProgenitorSubfindID' : -1,
            'PastMainProgenitorMaxMass_stars' : -1,
            'PastMainProgenitorMaxMass_gas' : -1,
            'PastMainProgenitorMaxMassInRad_stars' : -1,
            'PastMainProgenitorMaxMassInRad_gas' : -1,

            'PastNextProgenitorSnapNum' : -1,
            'PastNextProgenitorSubfindID' : -1,
            'PastNextProgenitorMaxMass_stars' : -1,
            'PastNextProgenitorMaxMass_gas' : -1,
            'PastNextProgenitorMaxMassInRad_stars' : -1,
            'PastNextProgenitorMaxMassInRad_gas' : -1,

            'FutureSnapNum' : -1,
            'FutureSubfindID' : -1,
            'FutureMassRatio' : -1,

            'FutureMainProgenitorSnapNum' : -1,
            'FutureMainProgenitorSubfindID' : -1,
            'FutureMainProgenitorMaxMass_stars' : -1,
            'FutureMainProgenitorMaxMass_gas' : -1,
            'FutureMainProgenitorMaxMassInRad_stars' : -1, 
            'FutureMainProgenitorMaxMassInRad_gas' : -1,

            'FutureNextProgenitorSnapNum' : -1,
            'FutureNextProgenitorSubfindID': -1,
            'FutureNextProgenitorMaxMass_stars' : -1,
            'FutureNextProgenitorMaxMass_gas' : -1,
            'FutureNextProgenitorMaxMassInRad_stars' : -1, 
            'FutureNextProgenitorMaxMassInRad_gas' : -1,

            'CountSinceMainLeafProgenitor' : -1,
            'CountSinceHalfScaleFactor' : -1,
            'CountSince250Myr' : -1,
            'CountSince500Myr' : -1,
            'CountSince1Gyr' : -1,
            'CountSince2Gyr' : -1,
            'CountSince3Gyr' : -1,
            'FreqencySinceMainLeafProgenitor' : -1,
            'TimeSinceMerger' : -1,
            
            'CountUntilRootDescendent': -1,
            'CountUntil250Myr' : -1,
            'CountUntil500Myr' : -1,
            'CountUntil1Gyr' : -1,
            'CountUntil2Gyr' : -1,
            'CountUntil3Gyr' : -1,
            'TimeUntilMerger' : -1,
            
            'TreeFlag': tree_flag,
        }
        return merger_info
    else:
        # tree unbroken
        tree_flag = 0
        rd_id = tree['SubhaloID'][0]
        rd_snap = tree['SnapNum'][0]
        rd_sub = tree['SubfindID'][0]
        
    # past merger params
    merger_num = 0
    merger_snaps = []
    merger_subs = []
    merger_mpb_snaps = []
    merger_mpb_subs = []
    merger_np_snaps = []
    merger_np_subs = []
    merger_mus = []
    merger_mpb_massmax = []
    merger_np_massmax = []
    merger_mpb_massradmax = []
    merger_np_massradmax = []
    merger_mpb_massmax_gas = []
    merger_np_massmax_gas = []
    merger_mpb_massradmax_gas = []
    merger_np_massradmax_gas = []
    tpostmerger = []
    
    # future merger params
    future_num = 0
    future_snaps = []
    future_subs = []
    future_mus = []
    future_mpb_snaps = []
    future_mpb_subs = []
    future_np_snaps = []
    future_np_subs = []
    future_mpb_massmax = []
    future_np_massmax = []
    future_mpb_massmax_gas = []
    future_np_massmax_gas = []
    future_mpb_massradmax = []
    future_np_massradmax = []
    future_mpb_massradmax_gas = []
    future_np_massradmax_gas = []
    tuntilmerger = []
    
    # get index of target galaxy
    tar_idx = np.argwhere((tree['SnapNum']==snap)*
                            (tree['SubfindID']==sub))[0][0]
    tar_snap = snap
    tar_sub = sub
    tar_id = tree['SubhaloID'][tar_idx]
    
    # past mergers

    # get indices of main progenitors (starting from tar_idx)

    mlp_id = tree['MainLeafProgenitorID'][tar_idx]
    mlp_idx = mlp_id - rd_id
    mlp_snap = tree['SnapNum'][mlp_idx]
    mlp_sub = tree['SubfindID'][mlp_idx]
    
    mpb_idxs = np.arange(tar_idx,mlp_idx+1)
    
    # now exclude root and search breadth for next progenitors
    for mpb_idx in mpb_idxs[1:]:
        merger_snap = tree['SnapNum'][mpb_idx-1]
        merger_sub = tree['SubfindID'][mpb_idx-1]
        mpb_snap = tree['SnapNum'][mpb_idx]
        mpb_sub = tree['SubfindID'][mpb_idx]
        np_id = tree['NextProgenitorID'][mpb_idx]
        np_num = 0
        
        # continue search along breadth until 
        while np_id != -1 and np_num<np_lim:
            np_idx = np_id - rd_id
            np_snap = tree['SnapNum'][np_idx]
            np_sub = tree['SubfindID'][np_idx]
            # get maximum next progentor mass in time limit
            np_mass_max,np_snap_max = max_past_mass_limit(
                basePath,outPath,sim,np_snap,np_sub,
                cosmo,limit=limit,mass_def=mass_def
            )

            # get main progenitor stellar mass and gas mass
            # where next progenitor mass is max in limit (w/ skips)
            mpb_mass_max = tree['SubhaloMassType'][
                mpb_idx + (np_snap - np_snap_max)][partNumStars]
            mpb_massrad_max = tree['SubhaloMassInRadType'][
                mpb_idx + (np_snap - np_snap_max)][partNumStars]
            mpb_mass_max_gas = tree['SubhaloMassType'][
                mpb_idx + (np_snap - np_snap_max)][partNumGas]
            mpb_massrad_max_gas = tree['SubhaloMassInRadType'][
                mpb_idx + (np_snap - np_snap_max)][partNumGas]
            
            # get mass to be used in comparison with NP for mass ratio
            mpb_mass_max_def = tree[mass_def][
                mpb_idx + (np_snap - np_snap_max)][partNumStars]
            
            if mpb_mass_max_def!=0 and np_mass_max!=0:
                mass_ratio = mpb_mass_max_def/np_mass_max
                # convert mass ratio to 0 < mu < 1 
                # (possibly np max mass > mpb max mass)
                if mass_ratio>1.0: 
                    mass_ratio=1./mass_ratio
                # only count if both masses are within mass ratio range
                if mass_ratio>mu_lower and mass_ratio<=mu_upper:
                    merger_num+=1
                    merger_snaps.append(merger_snap)
                    merger_subs.append(merger_sub)
                    merger_mpb_snaps.append(mpb_snap)
                    merger_mpb_subs.append(mpb_sub)
                    merger_np_snaps.append(np_snap)
                    merger_np_subs.append(np_sub)
                    merger_mus.append(mass_ratio)
                    merger_mpb_massmax.append(mpb_mass_max)
                    merger_mpb_massradmax.append(mpb_massrad_max)
                    merger_mpb_massmax_gas.append(mpb_mass_max_gas)
                    merger_mpb_massradmax_gas.append(mpb_massrad_max_gas)
                    # get other properties from next progenitor at max snap
                    npb_mask = tree['MainLeafProgenitorID'] == tree['MainLeafProgenitorID'][np_idx]
                    npb_snaps = tree['SnapNum'][npb_mask]
                    npb_mass = tree['SubhaloMassType'][npb_mask]
                    npb_massrad = tree['SubhaloMassInRadType'][npb_mask]
                    np_massrad_max = npb_massrad[
                        npb_snaps==np_snap_max][0][partNumStars]
                    np_mass_max_gas = npb_mass[
                        npb_snaps==np_snap_max][0][partNumGas]
                    np_massrad_max_gas = npb_massrad[
                        npb_snaps==np_snap_max][0][partNumGas]
                    merger_np_massmax.append(np_mass_max)
                    merger_np_massradmax.append(np_massrad_max)
                    merger_np_massmax_gas.append(np_mass_max_gas)
                    merger_np_massradmax_gas.append(np_massrad_max_gas)

            # get next progenitor pointer from current next progenitor       
            np_id = tree['NextProgenitorID'][np_idx]
            np_num += 1
            
    df_snaps = get_snapshot_df(basePath,outPath,sim,cosmo)
    # get snapshot info for root and mlp (lookback, scale factor, etc.)
    tar_record = df_snaps.loc[df_snaps['Snapshot']==tar_snap]
    tar_lookback = tar_record['LookbackTime'].values[0]
    mlp_record  = df_snaps.loc[df_snaps['Snapshot']==mlp_snap]
    mlp_lookback = mlp_record['LookbackTime'].values[0]
    
    # for each merger, get characteristics
    for i in range(merger_num):
        merger_record = df_snaps.loc[df_snaps['Snapshot']==merger_snaps[i]]
        tpostmerger.append(merger_record['LookbackTime'].values[0]-tar_lookback)
    
    # frequency of mergers since main leaf progenitor formation
    merger_freq_mlp = merger_num/mlp_lookback
    # number of mergers since the size of the universe doubled
    tar_scalefactor = tar_record['ScaleFactor'].values[0]
    # find snapshot for which universe was half size relative to root
    ahalf_idx = np.argmin(np.abs(df_snaps['ScaleFactor'].values/tar_scalefactor-0.5))
    ahalf_snap = df_snaps['Snapshot'].iloc[ahalf_idx]
    # get number of mergers since Universe was half the size
    merger_num_ahalf = sum(merger_snaps>=ahalf_snap)
    
    tpost = np.array(tpostmerger)
    merger_num_3Gyr = len(tpost[tpost<=3.0])
    merger_num_2Gyr = len(tpost[tpost<=2.0])
    merger_num_1Gyr = len(tpost[tpost<=1.0])
    merger_num_500Myr = len(tpost[tpost<=0.5])
    merger_num_250Myr = len(tpost[tpost<=0.25])
    
    # future mergers

    mdb_id = tar_id    
    # move up tree until root descendant
    while mdb_id != rd_id:
        mdb_idx = mdb_id - rd_id
        mdb_snap = tree['SnapNum'][mdb_idx]
        mdb_sub = tree['SubfindID'][mdb_idx] 
        # move up 1 descendant then down to find main progenitor
        tmp_id = tree['DescendantID'][mdb_idx]
        tmp_idx = tmp_id - rd_id
        # main progenitor at same level as target (possibly same as target)
        mpb_id = tree['FirstProgenitorID'][tmp_idx]
        mpb_idx = mpb_id - rd_id
        mpb_snap = tree['SnapNum'][mpb_idx]
        mpb_sub = tree['SubfindID'][mpb_idx]
        
        # search breadth for next progenitors
        np_id = tree['NextProgenitorID'][mpb_idx]
        np_num = 0
        while np_id != -1 and np_num<np_lim:
            
            np_idx = np_id - rd_id
            np_snap = tree['SnapNum'][np_idx]
            np_sub = tree['SubfindID'][np_idx]
            
            # only consider if mdb_id is one of the progenitors
            if (mdb_id != mpb_id) and (mdb_id != np_id):
                np_num+=1
                np_id = tree['NextProgenitorID'][np_idx]
                continue
                
            # get maximum next progentor mass in time limit
            np_mass_max,np_snap_max = max_past_mass_limit(
                basePath,outPath,sim,np_snap,np_sub,
                cosmo,limit=limit,mass_def=mass_def
            )
            # get main progenitor stellar mass and gas mass
            # where next progenitor mass is max in limit (w/ skips)
            mpb_mass_max = tree['SubhaloMassType'][
                mpb_idx + (np_snap - np_snap_max)][partNumStars]
            mpb_massrad_max = tree['SubhaloMassInRadType'][
                mpb_idx + (np_snap - np_snap_max)][partNumStars]
            mpb_mass_max_gas = tree['SubhaloMassType'][
                mpb_idx + (np_snap - np_snap_max)][partNumGas]
            mpb_massrad_max_gas = tree['SubhaloMassInRadType'][
                mpb_idx + (np_snap - np_snap_max)][partNumGas]
           
            # get mass to be used in comparison with NP for mass ratio
            mpb_mass_max_def = tree[mass_def][
                mpb_idx + (np_snap - np_snap_max)][partNumStars]
            
            if mpb_mass_max_def!=0 and np_mass_max!=0:
                mass_ratio = mpb_mass_max_def/np_mass_max
                # convert mass ratio to 0 < mu < 1 
                # (possibly np max mass > mpb max mass)
                if mass_ratio>1.0: 
                    mass_ratio=1./mass_ratio
                # only count if both masses are within mass ratio range
                if mass_ratio>mu_lower and mass_ratio<=mu_upper:
                    future_num+=1
                    # post-coalescence info
                    future_snaps.append(tree['SnapNum'][tmp_idx])
                    future_subs.append(tree['SubfindID'][tmp_idx])
                    # progenitor info
                    future_mus.append(mass_ratio)
                    future_mpb_snaps.append(mpb_snap)
                    future_mpb_subs.append(mpb_sub)
                    future_mpb_massmax.append(mpb_mass_max)
                    future_mpb_massmax_gas.append(mpb_mass_max_gas)
                    future_mpb_massradmax.append(mpb_massrad_max)
                    future_mpb_massradmax_gas.append(mpb_massrad_max_gas)
                    future_np_snaps.append(np_snap)
                    future_np_subs.append(np_sub)
                    # properties from next progenitor at max snap
                    npb_mask = tree['MainLeafProgenitorID'] == tree['MainLeafProgenitorID'][np_idx]
                    npb_snaps = tree['SnapNum'][npb_mask]
                    npb_mass = tree['SubhaloMassType'][npb_mask]
                    npb_massrad = tree['SubhaloMassInRadType'][npb_mask]
                    np_massrad_max = npb_massrad[
                        npb_snaps==np_snap_max][0][partNumStars]
                    np_mass_max_gas = npb_mass[
                        npb_snaps==np_snap_max][0][partNumGas]
                    np_massrad_max_gas = npb_massrad[
                        npb_snaps==np_snap_max][0][partNumGas]
                    future_np_massmax.append(np_mass_max)
                    future_np_massmax_gas.append(np_mass_max_gas)
                    future_np_massradmax.append(np_massrad_max)
                    future_np_massradmax_gas.append(np_massrad_max_gas)
       
            if np_id == mdb_id:
                break

            np_num+=1
            np_id = tree['NextProgenitorID'][np_idx]

        mdb_id = tree['DescendantID'][mdb_idx]
    
    # for each merger, get characteristics
    for i in range(future_num):
        future_record = df_snaps.loc[df_snaps['Snapshot']==future_snaps[i]]
        tuntilmerger.append(tar_lookback - future_record['LookbackTime'].values[0])
    
    tuntil = np.array(tuntilmerger)
    future_num_3Gyr = len(tuntil[tuntil<=3.0])
    future_num_2Gyr = len(tuntil[tuntil<=2.0])
    future_num_1Gyr = len(tuntil[tuntil<=1.0])
    future_num_500Myr = len(tuntil[tuntil<=0.5])
    future_num_250Myr = len(tuntil[tuntil<=0.25])
    
    # enter data into catalogue
    merger_info = {
            
        'PastSnapNum' : merger_snaps,
        'PastSubfindID' : merger_subs,
        'PastMassRatio' : merger_mus,
        
        'PastMainProgenitorSnapNum' : merger_mpb_snaps,
        'PastMainProgenitorSubfindID' : merger_mpb_subs,
        'PastMainProgenitorMaxMass_stars' : merger_mpb_massmax,
        'PastMainProgenitorMaxMass_gas' : merger_mpb_massmax_gas,
        'PastMainProgenitorMaxMassInRad_stars' : merger_mpb_massradmax,
        'PastMainProgenitorMaxMassInRad_gas' : merger_mpb_massradmax_gas,
        
        'PastNextProgenitorSnapNum' : merger_np_snaps,
        'PastNextProgenitorSubfindID' : merger_np_subs,
        'PastNextProgenitorMaxMass_stars' : merger_np_massmax,
        'PastNextProgenitorMaxMass_gas' : merger_np_massmax_gas,
        'PastNextProgenitorMaxMassInRad_stars' : merger_np_massradmax,
        'PastNextProgenitorMaxMassInRad_gas' : merger_np_massradmax_gas,
        
        'FutureSnapNum' : future_snaps,
        'FutureSubfindID' : future_subs,
        'FutureMassRatio' : future_mus,
        
        'FutureMainProgenitorSnapNum' : future_mpb_snaps,
        'FutureMainProgenitorSubfindID' : future_mpb_subs,
        'FutureMainProgenitorMaxMass_stars' : future_mpb_massmax,
        'FutureMainProgenitorMaxMass_gas' : future_mpb_massmax_gas,
        'FutureMainProgenitorMaxMassInRad_stars' : future_mpb_massradmax, 
        'FutureMainProgenitorMaxMassInRad_gas' : future_mpb_massradmax_gas,
        
        'FutureNextProgenitorSnapNum' : future_np_snaps,
        'FutureNextProgenitorSubfindID': future_np_subs,
        'FutureNextProgenitorMaxMass_stars' : future_np_massmax,
        'FutureNextProgenitorMaxMass_gas' : future_np_massmax_gas,
        'FutureNextProgenitorMaxMassInRad_stars' : future_np_massradmax, 
        'FutureNextProgenitorMaxMassInRad_gas' : future_np_massradmax_gas,
        
        'CountSinceMainLeafProgenitor' : merger_num,
        'CountSinceHalfScaleFactor' : merger_num_ahalf,
        'CountSince250Myr' : merger_num_250Myr,
        'CountSince500Myr' : merger_num_500Myr,
        'CountSince1Gyr' : merger_num_1Gyr,
        'CountSince2Gyr' : merger_num_2Gyr,
        'CountSince3Gyr' : merger_num_3Gyr,
        'FreqencySinceMainLeafProgenitor' : merger_freq_mlp,
        'TimeSinceMerger' : tpostmerger,
        
        'CountUntilRootDescendent': future_num,
        'CountUntil250Myr' : future_num_250Myr,
        'CountUntil500Myr' : future_num_500Myr,
        'CountUntil1Gyr' : future_num_1Gyr,
        'CountUntil2Gyr' : future_num_2Gyr,
        'CountUntil3Gyr' : future_num_3Gyr,
        'TimeUntilMerger' : tuntilmerger,
        
        'TreeFlag': tree_flag,
    }
    return merger_info

def get_merger_history_worker(args):
    sub,tree,common_args = args
    return (f'{sub}',get_merger_history(sub=sub,tree=tree,**common_args))
    # sub,common_args = args
    # return (f'{sub}',get_merger_history(sub=sub,tree=None,**common_args))
        
def get_merger_history_snap(basePath,outPath,sim,snap,cosmo,limit=0.5,
                            mstar_lower=0,mstar_upper=np.inf,
                            mass_def='SubhaloMassType'):
    
    filename = f'{outPath}/Catalogues/Mergers/{sim}_MergersInf_{snap:03}.npz'
    if os.access(filename,0):
        return np.load(filename,allow_pickle=True)
    
    little_h = cosmo.H0.value/100.
    mstar_lower = 10**mstar_lower*little_h/1e10
    mstar_upper = 10**mstar_upper*little_h/1e10
    
    df = max_past_mass_limit_snap(basePath,outPath,sim,snap,cosmo,limit=limit)
    subs = df['SubfindID'].values
    mstar = df['MaxSubhaloMassType_stars'].values
    indices = np.argwhere((mstar>=mstar_lower)*(mstar<=mstar_upper)).flatten()
    subs = subs[indices]

    import multiprocessing as mp
    
    nthreads = os.getenv('JOB_CPUS_PER_TASK')
    nthreads = (int(nthreads) if nthreads else 1)
    
    common_args = {
        'basePath':basePath,
        'outPath':outPath,
        'sim':sim,
        'snap':snap,
        'cosmo':cosmo,
        'limit':limit,
        'mass_def':mass_def
    }
    
    # get root descendant trees
    argList = [(basePath,snap,sub) for sub in subs]
    with mp.Pool(nthreads) as pool:
        trees = dict(pool.map(get_root_descendant_tree_worker,argList))
    
    tiny  = {}
    minor = {}
    major = {}
    
    # fill tiny, minor, major dictionaries 
    
    # tiny
    common_args['mu_lower']=0.01
    common_args['mu_upper']=0.1
    argList = [(sub,trees[f'{sub}'],common_args) for sub in subs]
    # argList = [(sub,trees[f'{sub}'],common_args) for sub in subs] 
    with mp.Pool(nthreads) as pool:
        tiny = dict(pool.map(get_merger_history_worker,argList))
        
    # minor
    common_args['mu_lower']=0.1
    common_args['mu_upper']=0.25
    argList = [(sub,trees[f'{sub}'],common_args) for sub in subs]
    # argList = [(sub,trees[f'{sub}'],common_args) for sub in subs] 
    with mp.Pool(nthreads) as pool:
        minor = dict(pool.map(get_merger_history_worker,argList))
        
    # major
    common_args['mu_lower']=0.25
    common_args['mu_upper']=1.0
    argList = [(sub,trees[f'{sub}'],common_args) for sub in subs]
    # argList = [(sub,trees[f'{sub}'],common_args) for sub in subs] 
    with mp.Pool(nthreads) as pool:
        major = dict(pool.map(get_merger_history_worker,argList))

    np.savez(filename,tiny=tiny,minor=minor,major=major)
    return np.load(filename,allow_pickle=True)
    
def main():
    
    from astropy.cosmology import Planck15 as cosmo
    sim = os.getenv('SIM')
    snap = int(os.getenv('SNAP'))
    mstar_lower=float(os.getenv('MSTAR_LOWER'))

    basePath = f'/lustre/work/connor.bottrell/Simulations/IllustrisTNG/{sim}/output'
    outPath = f'/lustre/work/connor.bottrell/Simulations/IllustrisTNG/Scripts/Mergers/illustris_mergers'

    get_merger_history_snap(basePath,outPath,sim,snap,cosmo,
                            limit=np.inf,mstar_lower=mstar_lower,
                            mass_def='SubhaloMassInRadType')

if __name__=='__main__':
    
    main()
