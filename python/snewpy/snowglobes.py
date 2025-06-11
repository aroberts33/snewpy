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
    print(f"INFO: SNEWPY is using supernova model file: {getattr(snmodel, 'filename', 'N/A')}")

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
    # energy   = np.arange(0, 101, 0.2) << u.MeV
    energy = (np.arange(505) * 0.2 + 0.1) * u.MeV  # 0.1 MeV bin centers, 0.2 MeV bin width
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
            df.replace([np.inf, -np.inf], np.nan, inplace=True) # First, ensure infs are NaNs
            df.fillna(0.0, inplace=True) # Now, fill NaNs with 0.0

            if len(tbins) > 1:
                result[det][f'{fname_base}_{n_bin:01d}'] = df
            else:
                result[det][f'{fname_base}'] = df
           
    # save result to file for re-use in collate()
    cache_file = f'{fname_base}.npy'
    logging.info(f'Saving simulation results to {cache_file}')
    np.save(cache_file, result) # result will now contain DataFrames with 0s instead of NaNs
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

    # 
    
    # In your /home/aroberts/SN_project/snewpy_fork/snewpy/python/snewpy/snowglobes.py

    def aggregate_channels(table: pd.DataFrame, **patterns: str) -> pd.DataFrame:
        if table.empty:
            return table # Return early if input is already empty

        original_col_names_from_input = list(table.columns.names) # Save for later reordering
        levels_to_stack = [name for name in original_col_names_from_input if name != 'channel']

        # Use future_stack=True to be explicit and avoid the warning
        # This stacks ['is_weighted', 'is_smeared'] levels into the index.
        # The columns of df_stacked will be the original 'channel' names.
        df_stacked = table.stack(levels_to_stack, future_stack=True, ) # Use dropna=False explicitly

        # If df_stacked becomes a Series (e.g., only one unique original channel string)
        # convert it to a DataFrame for consistent processing.
        if isinstance(df_stacked, pd.Series):
            s_name = df_stacked.name
            df_stacked = df_stacked.to_frame(name=s_name if s_name is not None else 'unknown_channel_placeholder')
            # After to_frame, df_stacked.columns is Index(['unknown_channel_placeholder'], dtype='object')
            # And df_stacked.columns.name is None

        # df_stacked is now a DataFrame. Its columns are the original channel names.
        # Its index is (original_index_levels..., is_weighted_val, is_smeared_val)
        # Index names might be (original_index_name, 'is_weighted', 'is_smeared')

        # --- Aggregation Logic ---
        # Create a copy to modify, or build a list of series to concat
        processed_df = df_stacked.copy() # Start with all original channels that were columns in df_stacked

        for agg_name, pattern in patterns.items():
            # Select columns (original channels) that match the pattern
            # Ensure columns are strings for .str.contains if they are not already
            cols_to_aggregate_mask = processed_df.columns.astype(str).str.contains(pattern, regex=True)
            cols_to_aggregate = processed_df.columns[cols_to_aggregate_mask].tolist()

            if cols_to_aggregate:
                processed_df[agg_name] = processed_df[cols_to_aggregate].sum(axis=1)
                processed_df.drop(columns=cols_to_aggregate, inplace=True)
        # --- End Aggregation Logic ---

        # Now, processed_df contains non-aggregated channels + new aggregated channels (like 'nc', 'e')
        # Its index is still (original_index_levels..., is_weighted_val, is_smeared_val)

        if processed_df.empty and processed_df.columns.empty : # If NO columns are left (e.g. all aggregated away AND originals dropped)
            # This might happen if input 'table' had only channels that matched patterns,
            # and they were all summed and dropped, leaving only new aggregated columns.
            # If 'processed_df' is truly empty (0x0), unstacking might be an issue.
            # However, if it has an index but 0 columns, unstacking specific index levels
            # should produce a DataFrame with 0 columns but the correct new column MultiIndex structure.
            # Let's assume unstack handles DataFrames with 0 columns but a valid index to unstack.
            pass


        df_unstacked = processed_df.unstack(levels_to_stack)
        # After unstack, the column levels will be:
        # (current_columns_of_processed_df, values_from_is_weighted, values_from_is_smeared)
        # Their names will be (processed_df.columns.name, 'is_weighted', 'is_smeared')

        # Reorder to the original desired output structure: ('is_weighted', 'is_smeared', 'channel')
        # The 'channel' here refers to the new set of channels (original un-aggregated + new aggregated)
        
        # Get current names after unstack
        current_unstacked_names = list(df_unstacked.columns.names)
        target_names_for_reorder = []
        
        # Build the target order for reorder_levels based on original_col_names_from_input
        if 'is_weighted' in current_unstacked_names: target_names_for_reorder.append('is_weighted')
        if 'is_smeared' in current_unstacked_names: target_names_for_reorder.append('is_smeared')
        
        # The remaining level name should be the one that held the channel strings
        # This was the .columns.name of processed_df before unstacking.
        # Often this might be None or the original 'channel' if preserved.
        channel_level_name_after_unstack = [n for n in current_unstacked_names if n not in ['is_weighted', 'is_smeared']]
        if channel_level_name_after_unstack:
            target_names_for_reorder.append(channel_level_name_after_unstack[0])
        elif 'channel' in original_col_names_from_input and len(current_unstacked_names) == len(original_col_names_from_input):
            # If names were lost, but count matches, assume 'channel' is the one missing from target
            target_names_for_reorder.append(original_col_names_from_input[original_col_names_from_input.index('channel')])


        if len(target_names_for_reorder) == df_unstacked.columns.nlevels and not df_unstacked.columns.empty:
            try:
                df_reordered = df_unstacked.reorder_levels(target_names_for_reorder, axis=1)
                # Also ensure the names themselves are set correctly if reorder_levels doesn't do it
                df_reordered.columns.names = original_col_names_from_input 
                return df_reordered
            except Exception as e:
                logger.error(f"Error during reorder_levels in aggregate_channels: {e}. Current names: {current_unstacked_names}, Target names: {target_names_for_reorder}")
                # Fallback: return df_unstacked, but ensure it has the target names if possible, even if order is wrong
                if len(df_unstacked.columns.names) == len(original_col_names_from_input):
                    try_renaming = dict(zip(df_unstacked.columns.names, original_col_names_from_input))
                    df_unstacked = df_unstacked.rename_axis(columns=try_renaming)

                return df_unstacked # Return un-reordered if reordering failed but columns exist
        elif df_unstacked.columns.empty: # If unstack resulted in empty columns (should have names from my fallback)
            # Ensure it has the correct empty MultiIndex structure expected by the caller
            df_unstacked.columns = pd.MultiIndex(levels=[[]]*len(original_col_names_from_input),
                                                codes=[[]]*len(original_col_names_from_input),
                                                names=original_col_names_from_input)
            return df_unstacked
        else: # Mismatch in number of levels or some other issue
            logger.warning(f"aggregate_channels: Could not reorder levels. Returning df_unstacked. Dims: {df_unstacked.columns.nlevels} vs {len(original_col_names_from_input)}")
            return df_unstacked

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
