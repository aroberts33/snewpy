# -*- coding: utf-8 -*-
"""The ``snewpy.snowglobes`` module contains functions for interacting with SNOwGLoBES.

`SNOwGLoBES <https://github.com/SNOwGLoBES/snowglobes>`_ can estimate detected
event rates from a given input supernova neutrino flux. It supports many
different neutrino detectors, detector materials and interaction channels.
There are three basic steps to using SNOwGLoBES from SNEWPY:

* **Generating input files for SNOwGLoBES:**
    There are two ways to do this, either generate a time series or a fluence file. This is done taking as input the supernova simulation model.
    The first will evaluate the neutrino flux at each time step, the latter will compute the integrated neutrino flux (fluence) in the time bin.
    The result is a compressed .tar file containing all individual input files.
* **Running SNOwGLoBES:**
    This step convolves the fluence generated in the previous step with the cross-sections for the interaction channels happening in various detectors supported by SNOwGLoBES.
    It takes into account the effective mass of the detector as well as a smearing matrix describing the energy-dependent detection efficiency.
    The output gives the number of events detected as a function of energy for each interaction channel, integrated in a given time window (or time bin), or in a snapshot in time.
* **Collating SNOwGLoBES outputs:**
    This step puts together all the interaction channels and time bins evaluated by SNOwGLoBES in a single file (for each detector and for each time bin).
    The output tables allow to build the detected neutrino energy spectrum and neutrino time distribution, for each reaction channel or the sum of them.
"""

import logging
import os
import re
import tarfile
from pathlib import Path
from tempfile import TemporaryDirectory

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from astropy import units as u
from warnings import warn

import snewpy.models
from snewpy.flavor_transformation import *
from snewpy.neutrino import MassHierarchy
from snewpy.rate_calculator import RateCalculator, center
from snewpy.flux import Container
logger = logging.getLogger(__name__)

def generate_time_series(model_path, model_type, transformation_type, d, output_filename=None, ntbins=30, deltat=None, snmodel_dict={}):
    """Generate time series files in SNOwGLoBES format.

    This version will subsample the times in a supernova model, produce energy
    tables expected by SNOwGLoBES, and compress the output into a tarfile.

    Parameters
    ----------
    model_path : str
        Input file containing neutrino flux information from supernova model.
    model_type : str
        Format of input file. Matches the name of the corresponding class in :py:mod:`snewpy.models`.
    transformation_type : str
        Name of flavor transformation. See snewpy.flavor_transformation documentation for possible values.
    d : int or float
        Distance to supernova in kpc.
    output_filename : str or None
        Name of output file. If ``None``, will be based on input file name.
    ntbins : int
        Number of time slices. Will be ignored if ``deltat`` is also given.
    deltat : astropy.Quantity or None
        Length of time slices.
    snmodel_dict : dict
        Additional arguments for setting up the supernova model. See documentation of relevant ``SupernovaModel`` subclass for available options. (Optional)

    Returns
    -------
    str
        Path of NumPy archive file with neutrino fluence data.
    """
    print(model_type)
    model_class = getattr(snewpy.models.ccsn, model_type)

    # Choose flavor transformation. Use dict to associate the transformation name with its class.
    flavor_transformation_dict = {'NoTransformation': NoTransformation(), 'AdiabaticMSW_NMO': AdiabaticMSW(mh=MassHierarchy.NORMAL), 'AdiabaticMSW_IMO': AdiabaticMSW(mh=MassHierarchy.INVERTED), 'NonAdiabaticMSWH_NMO': NonAdiabaticMSWH(mh=MassHierarchy.NORMAL), 'NonAdiabaticMSWH_IMO': NonAdiabaticMSWH(mh=MassHierarchy.INVERTED), 'TwoFlavorDecoherence': TwoFlavorDecoherence(), 'ThreeFlavorDecoherence': ThreeFlavorDecoherence(), 'NeutrinoDecay_NMO': NeutrinoDecay(mh=MassHierarchy.NORMAL), 'NeutrinoDecay_IMO': NeutrinoDecay(mh=MassHierarchy.INVERTED), 'QuantumDecoherence_NMO': QuantumDecoherence(mh=MassHierarchy.NORMAL), 'QuantumDecoherence_IMO': QuantumDecoherence(mh=MassHierarchy.INVERTED)}
    flavor_transformation = flavor_transformation_dict[transformation_type]

    model_dir, model_file = os.path.split(os.path.abspath(model_path))
    snmodel = model_class(model_path, **snmodel_dict)

    # Subsample the model time. Default to 30 time slices.
    tmin = snmodel.get_time()[0]
    tmax = snmodel.get_time()[-1]
    if deltat is not None:
        dt = deltat
        ntbins = int((tmax-tmin)/dt)
    else:
        dt = (tmax - tmin) / (ntbins+1)

    times = np.arange(tmin/u.s, tmax/u.s, dt/u.s)*u.s
    energy = np.linspace(0, 100, 501) * u.MeV
    flux = snmodel.get_flux(t=times, E=energy,  distance=d, flavor_xform=flavor_transformation)
    fluence = flux.integrate('time', limits = times).integrate('energy', limits = energy)
    #save resulting fluence to file
    if output_filename is not None:
        tfname = output_filename + '.npz'
    else:
        model_file_root, _ = os.path.splitext(model_file)  # strip extension (if present)
        tfname = f'{model_file_root}.{transformation_type}.{tmin:.3f},{tmax:.3f},{ntbins:d}-{d:.1f}.npz'
    fluence.save(tfname)
    return tfname

def generate_fluence(model_path, model_type, transformation_type, d, output_filename=None, tstart=None, tend=None, snmodel_dict={}):
    """Generate fluence files in SNOwGLoBES format.

    This version will subsample the times in a supernova model, produce energy
    tables expected by SNOwGLoBES, and compress the output into a tarfile.

    Parameters
    ----------
    model_path : str
        Input file containing neutrino flux information from supernova model.
    model_type : str
        Format of input file. Matches the name of the corresponding class in :py:mod:`snewpy.models`.
    transformation_type : str
        Name of flavor transformation. See snewpy.flavor_transformation documentation for possible values.
    d : int or float
        Distance to supernova in kpc.
    output_filename : str or None
        Name of output file. If ``None``, will be based on input file name.
    tstart : astropy.Quantity or None
        Start of time interval to integrate over, or list of start times of the time series bins.
    tend : astropy.Quantity or None
        End of time interval to integrate over, or list of end times of the time series bins.
    snmodel_dict : dict
        Additional arguments for setting up the supernova model. See documentation of relevant ``SupernovaModel`` subclass for available options. (Optional)

    Returns
    -------
    str
        Path of NumPy archive file with neutrino fluence data.
    """
    model_class = getattr(snewpy.models.ccsn, model_type)

    # Choose flavor transformation. Use dict to associate the transformation name with its class.
    flavor_transformation_dict = {'NoTransformation': NoTransformation(), 'AdiabaticMSW_NMO': AdiabaticMSW(mh=MassHierarchy.NORMAL), 'AdiabaticMSW_IMO': AdiabaticMSW(mh=MassHierarchy.INVERTED), 'NonAdiabaticMSWH_NMO': NonAdiabaticMSWH(mh=MassHierarchy.NORMAL), 'NonAdiabaticMSWH_IMO': NonAdiabaticMSWH(mh=MassHierarchy.INVERTED), 'TwoFlavorDecoherence': TwoFlavorDecoherence(), 'ThreeFlavorDecoherence': ThreeFlavorDecoherence(), 'NeutrinoDecay_NMO': NeutrinoDecay(mh=MassHierarchy.NORMAL), 'NeutrinoDecay_IMO': NeutrinoDecay(mh=MassHierarchy.INVERTED), 'QuantumDecoherence_NMO': QuantumDecoherence(mh=MassHierarchy.NORMAL), 'QuantumDecoherence_IMO': QuantumDecoherence(mh=MassHierarchy.INVERTED)}
    flavor_transformation = flavor_transformation_dict[transformation_type]

    model_dir, model_file = os.path.split(os.path.abspath(model_path))

    # Current line in your snewpy/snowglobes.py:
    snmodel = model_class(model_path, **snmodel_dict)

    # # PROPOSED CHANGE:
    # if snmodel_dict: # If specific model parameters are provided in snmodel_dict
    #     # Prioritize initializing the model with these parameters.
    #     # The model's __init__ (e.g., Zha_2021.__init__) should handle constructing
    #     # the internal relative filename needed for its loader.
    #     try:
    #         snmodel = model_class(**snmodel_dict)
    #     except TypeError as e:
    #         # This might happen if model_class still expects 'filename' due to legacy wrappers
    #         # and snmodel_dict doesn't provide a dummy one.
    #         # Or if snmodel_dict is missing a required parameter for parameter-based init.
    #         print(f"Attempting parameter-based init for {model_class.__name__} failed: {e}")
    #         print(f"Falling back to trying init with model_path: {model_path}")
    #         # Fallback to original behavior if direct parameter init fails (might re-trigger old error)
    #         # Or, be more strict: if snmodel_dict is present, it MUST work this way.
    #         # For now, let's try to be smart. The Zha_2021 __init__ only takes progenitor_mass.
    #         # The legacy wrapper adds 'filename'.
    #         # The get_model(model_type) returns the class *after* legacy_filename_initialization.
    #         # So model_class() will expect filename=None for param-based path.
    #         try:
    #             # Try calling it as if filename was None initially in the legacy wrapper
    #             snmodel = model_class(filename=None, **snmodel_dict)
    #         except Exception as e2:
    #             print(f"Secondary attempt with filename=None also failed: {e2}")
    #             print("Re-trying original call that led to error, for debugging.")
    #             snmodel = model_class(model_path, **snmodel_dict) # This will likely re-trigger the error
    # else:
    #     # If no snmodel_dict, assume model_path is the primary identifier (old way)
    #     snmodel = model_class(model_path)

    #set the timings up
    #default if inputs are None: full time window of the model
    times = None
    if tstart is not None and tend is not None:
        try:
            #in case we have arrays: join them together
            times = np.append(tstart, tend)
            #and get rid of the duplicates with 1e-10 tolerance
            times = np.unique(times.round(decimals=10))
        except:
            #in case we have single values
            times = u.Quantity([tstart,tend])
        times.sort()

    #energy with 0.2 MeV binning
    energy   = np.arange(0, 101, 0.2) << u.MeV
    #energy bins similar to SNOwGLoBES
    energy_t = (np.linspace(0, 100, 201)+0.25) << u.MeV
    flux = snmodel.get_flux(t=snmodel.get_time(), E=energy,  distance=d, flavor_xform=flavor_transformation)
    fluence = flux.integrate('time', limits = times).integrate('energy', limits = energy_t)
    times = fluence.time
    #store the energy bin centers instead of the edges
    if output_filename is not None:
        tfname = output_filename+'.npz'
    else:
        model_file_root, _ = os.path.splitext(model_file)  # strip extension (if present)
        tfname = f'{model_file_root}.{transformation_type}.{times[0]:.3f},{times[1]:.3f},{len(times)-1:d}-{d:.1f}.npz'

    fluence.save(tfname)
    return tfname

def simulate(SNOwGLoBESdir, tarball_path, detector_input="all", verbose=False, *, detector_effects=True):
    """Calculate expected event rates for the given neutrino flux files and the given (set of) SNOwGLoBES detector(s).
    These event rates are given as a function of the neutrino energy and time, for each interaction channel.

    Parameters
    ----------
    SNOwGLoBESdir : str or None
        Path to SNOwGLoBES directory. Set to ``None`` to automatically use the latest supported SNOwGLoBES release.
    tarball_path : str
        Path of compressed .tar file produced e.g. by ``generate_time_series()`` or ``generate_fluence()``.
    detector_input : str
        Name of detector. If ``"all"``, will use all detectors supported by SNOwGLoBES.
    verbose : bool
        [DEPRECATED, DO NOT USE.]
    detector_effects : bool
         Whether to account for detector smearing and efficiency.
    """
    if verbose:  # Deprecated since SNEWPY v1.2
        warn(f"The 'verbose' parameter to 'snewpy.snowglobes.simulate()' is deprecated and should not be used.", FutureWarning)

    rc = RateCalculator(base_dir=SNOwGLoBESdir)
    if detector_input == 'all':
        detector_input = list(rc.detectors)
    if(isinstance(detector_input,str)):
        detector_input=[detector_input]
    rates_dict = {}
    #read the fluence
    fluence = Container.load(tarball_path)
    for det in detector_input:
        rates_smeared=rc.run(fluence, det, detector_effects=True)
        rates_unsmeared=rc.run(fluence, det, detector_effects=False)
        #collect everything to pandas DataFrame, to make the output similar to previous
        rates_dict[det]={'weighted':{'unsmeared':rates_unsmeared,
                                 'smeared':rates_smeared,
                                }}
    # reorder results to produce the same format as before:
    #    {detector: {time_bin:{'weighted':{smeared/unsmeared: [rate vs energy bins]}}}}
    result = {}
    fname_base = tarball_path[:tarball_path.rfind('.')]
    for det in rates_dict:
        #get the time bins
        rates_smeared   = rates_dict[det]['weighted']['smeared']
        rates_unsmeared = rates_dict[det]['weighted']['unsmeared']

        #get the first rate from the dict to access the energy and time binning
        some_rate = list(rates_smeared.values())[0]
        tbins = center(some_rate.time)
        ebins = center(some_rate.energy)
        result[det] = {}
        for n_bin, t_bin in enumerate(tbins):
            data = {**{(chan,'unsmeared','weighted'): rate.array[0,n_bin,:]
                      for chan,rate in rates_unsmeared.items()},
                    **{(chan,'smeared','weighted'): rate.array[0,n_bin,:]
                      for chan,rate in rates_smeared.items()}}

            df = pd.DataFrame(data, index = ebins)
            df.index.rename('E', inplace=True)
            df.columns.rename(['channel','is_smeared','is_weighted'], inplace=True)
            df = df.reorder_levels([2,1,0], axis='columns')
            if len(tbins) > 1:
                result[det][f'{fname_base}_{n_bin:01d}'] = df
            else:
                result[det][f'{fname_base}'] = df

    # save result to file for re-use in collate()
    cache_file = f'{fname_base}.npy'
    logging.info(f'Saving simulation results to {cache_file}')
    np.save(cache_file, result)
    return result


re_chan_label = re.compile(r'nu(e|mu|tau)(bar|)_([A-Z][a-z]*)(\d*)_?(.*)')
def get_channel_label(c):
    mapp = {'nc':'NeutralCurrent',
            'ibd':'Inverse Beta Decay',
            'e':r'${\nu}_x+e^-$'}
    def gen_label(m):
        flv,bar,Nuc,num,res = m.groups()
        if flv!='e':
            flv='\\'+flv
        if bar:
            bar='\\'+bar
        s = f'${bar}{{\\nu}}_{flv}$ '+f'${{}}^{{{num}}}{Nuc}$ '+res
        return s

    if c in mapp:
        return mapp[c]
    else:
        return re_chan_label.sub(gen_label, c)

def collate(SNOwGLoBESdir, tarball_path, detector_input="", skip_plots=False, verbose=False, remove_generated_files=True, *, smearing=True):
    """Collates SNOwGLoBES output files and generates plots or returns a data table.

    Parameters
    ----------
    SNOwGLoBESdir : str or None
        [DEPRECATED, DO NOT USE.]
    tarball_path : str
        Path of compressed .tar file produced e.g. by ``generate_time_series()`` or ``generate_fluence()``.
    detector_input : str
        [DEPRECATED, DO NOT USE. SNEWPY will use all detectors included in the tarball.]
    skip_plots: bool
        If False, it gives as output the plot of the energy distribution for each time bin and for each interaction channel.
    verbose : bool
        [DEPRECATED, DO NOT USE.]
    remove_generated_files: bool
        [DEPRECATED, DO NOT USE.]
    smearing: bool
        Also consider results with smearing effects.

    Returns
    -------
    dict
        Dictionary of data tables: One table per time bin; each table contains in the first column the energy bins, in the remaining columns the number of events for each interaction channel in the detector.
    """
    if verbose:  # Deprecated since SNEWPY v1.2
        warn(f"The 'verbose' parameter to 'snewpy.snowglobes.collate()' is deprecated and should not be used.", FutureWarning)
    if detector_input:  # Deprecated since SNEWPY v1.2
        warn(f"The 'detector_input' parameter to 'snewpy.snowglobes.collate()' is deprecated and should not be used.", FutureWarning)
    if not remove_generated_files:  # Deprecated since SNEWPY v1.2
        warn(f"The 'remove_generated_files' parameter to 'snewpy.snowglobes.collate()' is deprecated and should not be used.", FutureWarning)

    # Inside your existing aggregate_channels in snowglobes.py

    def aggregate_channels(table, **patterns):
        levels = list(table.columns.names)
        levels_to_stack = [lvl for lvl in levels if lvl != 'channel'] # Should be ['is_weighted', 'is_smeared']
        
        print(f"AGGREGATE_CHANNELS: Input table columns: {table.columns}")
        print(f"AGGREGATE_CHANNELS: Input table index: {table.index}")

        # It's important to know if future_stack makes a difference to the intermediate 't'
        # For now, let's use what's in your file, but be aware of the warning.
        t = table.stack(levels) # This is t_stacked in my previous comments
        
        print(f"AGGREGATE_CHANNELS: After stack (now 't'):")
        if isinstance(t, pd.Series):
            print(f"  t is a Series, name: {t.name}, shape: {t.shape}, index names: {t.index.names}")
            # If it's a Series, convert to DataFrame for consistent processing
            t = t.to_frame(name=t.name if t.name is not None else 'channel_0')
            print(f"  t converted to DataFrame, columns: {t.columns}")
        else:
            print(f"  t is a DataFrame, columns: {t.columns}, shape: {t.shape}, index names: {t.index.names}")

        # Original aggregation loop from your SNEWPY fork
        # This loop modifies 't' inplace if it's a DataFrame
        for name, pattern in patterns.items():
            # Ensure t is a DataFrame for filter(axis=1) and drop(columns=...)
            if isinstance(t, pd.Series): # Should have been converted above, but as a safeguard
                print(f"WARNING: 't' became a Series before pattern loop for pattern '{pattern}'")
                # Cannot easily proceed with .filter(like=pattern) on columns or .drop
                continue 

            t_sel = t.filter(like=pattern, axis=1) # Filter columns of t
            print(f"  Pattern '{pattern}': t_sel is empty: {t_sel.empty}, t_sel shape: {t_sel.shape}")
            if not t_sel.empty:
                print(f"    t_sel columns: {t_sel.columns}")
            
            t_agg = t_sel.sum(axis='columns')
            
            if not t_sel.empty: # Only drop if columns were found
                t.drop(t_sel.columns, axis='columns', inplace=True, errors='ignore') # errors='ignore' if some cols not found
            
            t[name] = t_agg # Add the new aggregated column

        print(f"AGGREGATE_CHANNELS: t BEFORE UNSTACK:")
        print(f"  Columns: {t.columns}")
        print(f"  Index names: {t.index.names}")
        print(f"  Shape: {t.shape}")
        if not t.empty:
            print(f"  First few rows of t:\n{t.head()}") # See if data (even NaNs) is present
        else:
            print("  t is empty (no rows or no columns)!")


        # THE CRITICAL UNSTACK OPERATION
        if not t.columns.empty and t.index.nlevels == len(levels_to_stack) + 1: # Ensure t has columns and a suitable index
            t_unstacked = t.unstack(levels_to_stack)
        else:
            print(f"SKIPPING UNSTACK: 't' has no columns or unsuitable index. t.columns: {t.columns}, t.index.nlevels: {t.index.nlevels}, expected index levels for unstack: {len(levels_to_stack)}")
            # Create an empty DataFrame with the expected column structure if t is problematic
            # This is a palliative measure; the root cause is t becoming columnless
            empty_mi = pd.MultiIndex(levels=[[]]*len(table.columns.names),
                                    codes=[[]]*len(table.columns.names),
                                    names=table.columns.names)
            t_unstacked = pd.DataFrame(columns=empty_mi, index=table.index.get_level_values(0).unique())


        print(f"AGGREGATE_CHANNELS: t AFTER UNSTACK (now 't_unstacked'):")
        print(f"  Columns: {t_unstacked.columns}")
        print(f"  Column names: {t_unstacked.columns.names}")
        print(f"  Shape: {t_unstacked.shape}")
        if not t_unstacked.empty:
            print(f"  First few rows of t_unstacked:\n{t_unstacked.head()}")

        # Reorder levels
        # The names of the levels in t_unstacked.columns should ideally be
        # [name_of_innermost_cols_of_t, levels_to_stack[0]_name, levels_to_stack[1]_name, ...]
        # original_col_names was ['is_weighted', 'is_smeared', 'channel']
        # We expect t_unstacked.columns.names to be something like ['channel', 'is_weighted', 'is_smeared']
        # if names are preserved correctly.
        
        # Check if t_unstacked has any columns before trying to reorder
        if t_unstacked.columns.empty:
            print("AGGREGATE_CHANNELS: t_unstacked has no columns, cannot reorder. Returning as is.")
            # It will have the empty MultiIndex([], names=['is_weighted', 'is_smeared', 'channel']) from earlier if it hit that.
            # Or if unstack produced MultiIndex([], names=THE_ACTUAL_NAMES_AFTER_UNSTACK)
            # We need to ensure it has the *target* names for the calling collate function.
            if not isinstance(t_unstacked.columns, pd.MultiIndex) or not t_unstacked.columns.names == original_col_names:
                t_unstacked.columns = pd.MultiIndex(levels=[[]]*len(original_col_names),
                                                    codes=[[]]*len(original_col_names),
                                                    names=original_col_names)
            return t_unstacked

        # Attempt reordering if columns exist
        # Ensure the names in original_col_names actually exist in t_unstacked.columns.names
        # or reorder by position if names are lost/None
        current_names = list(t_unstacked.columns.names)
        if all(name in current_names for name in original_col_names) and len(current_names) == len(original_col_names):
            t_reordered = t_unstacked.reorder_levels(original_col_names, axis=1)
        else:
            print(f"AGGREGATE_CHANNELS: Column level names mismatch or count mismatch for reordering. Current: {current_names}, Original Target: {original_col_names}. Returning t_unstacked without reordering levels by name.")
            # If reorder_levels by name fails, the structure might be t_unstacked['weighted'] later on.
            # For now, just return what unstack gave if names are weird.
            t_reordered = t_unstacked


        print(f"AGGREGATE_CHANNELS: t FINAL (after reorder):")
        print(f"  Columns: {t_reordered.columns}")
        print(f"  Column names: {t_reordered.columns.names}")

        return t_reordered

    def do_plot(table, params):
        #plotting the events from given table
        flux,det,weighted,smeared = params
        for c in table.columns:
            if table[c].max() > 0.1:
                plt.plot(table[c],drawstyle='steps',label=get_channel_label(c), lw=1)
        plt.xlim(right=0.10)
        plt.ylim(bottom=0.10)
        plt.yscale('log')
        plt.legend(bbox_to_anchor=(0.5, 0.5, 0.5, 0.5), loc='best', borderaxespad=0)  # formats complete graph
        smear_title = 'Interaction' if smeared=='unsmeared' else 'Detected'
        plt.title(f'{flux} {det.capitalize()} {weighted.capitalize()} {smear_title} Events')
        if smeared=='smeared':
            plt.xlabel('Detected Energy (GeV)')
            plt.ylabel('Events')
        else:
            plt.xlabel('Neutrino Energy (GeV)')
            plt.ylabel('Interaction Events')

    #read the results from storage
    cache_file = tarball_path[:tarball_path.rfind('.')] + '.npy'
    logging.info(f'Reading tables from {cache_file}')
    tables = np.load(cache_file, allow_pickle=True).tolist()
    #This output is similar to what produced by:
    #tables = simulate(SNOwGLoBESdir, tarball_path,detector_input)

    #dict for old-style results, for backward compatibiity
    results = {}
    smearing_options = ['smeared','unsmeared'] if smearing else ['unsmeared']
    #save collated files:
    with TemporaryDirectory(prefix='snowglobes') as tempdir:
        tempdir = Path(tempdir)
        for det in tables:
            results[det] = {}
            for flux,t in tables[det].items():
                t = aggregate_channels(t,nc='nc_',e='_e')
                print(f"--- Debugging snowglobes.collate for flux: {flux} ---")
                print(f"Columns of t (output of aggregate_channels): {t.columns}")
                print(f"Column names of t: {t.columns.names}")
                if isinstance(t.columns, pd.MultiIndex):
                    print(f"Level 0 values of t.columns: {t.columns.get_level_values(0).unique()}")
                    if t.columns.nlevels > 1:
                        print(f"Level 1 values of t.columns: {t.columns.get_level_values(1).unique()}")
                    if t.columns.nlevels > 2:
                        print(f"Level 2 values of t.columns: {t.columns.get_level_values(2).unique()}")
                print("--- End Debugging ---")

                for w in ['weighted']:
                    for s in smearing_options:
                        table = t[w][s]
                        filename_base = f'{flux}_{det}_events_{s}_{w}'
                        filename = tempdir/f'Collated_{filename_base}.dat'
                        #save results to text files
                        with open(filename,'w') as f:
                            f.write(table.to_string(float_format='%23.15g'))
                        #format the results for the output
                        header = 'Energy '+' '.join(list(table.columns))
                        data = table.to_numpy().T
                        index = table.index.to_numpy()
                        data = np.concatenate([[index],data])
                        results[filename.name] = {'header':header,'data':data}
                        #optionally plot the results
                        if skip_plots is False:
                            plt.figure(dpi=300)
                            do_plot(table,(flux,det,w,s))
                            filename = tempdir/f'{filename_base}_log_plot.png'
                            plt.savefig(filename, dpi=300, bbox_inches='tight')
                            plt.close()
        #Make a tarfile with the condensed data files and plots
        output_name = Path(tarball_path).stem
        output_name = output_name[:output_name.rfind('.tar')]+'_SNOprocessed'
        output_path = Path(tarball_path).parent/(output_name+'.tar.gz')
        with tarfile.open(output_path, "w:gz") as tar:
            for file in tempdir.iterdir():
                tar.add(file,arcname=output_name+'/'+file.name)
        logging.info(f'Created archive: {output_path}')
    return results
