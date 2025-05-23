# -*- coding: utf-8 -*-
"""
The submodule ``snewpy.models.ccsn_loaders`` contains classes to load core-collapse
supernova models from files stored on disk.
"""

import logging
import os
import re
import sys
import tarfile

from astropy import units as u
from astropy.table import Table, join
from astropy.io import ascii, fits
import h5py
import numpy as np
from scipy.special import gamma, lpmv

try:
    import healpy as hp
except ImportError:
    pass

from snewpy.models.base import PinchedModel, SupernovaModel
from snewpy.neutrino import Flavor
from snewpy import _model_downloader

class GarchingArchiveModel(PinchedModel):
    """Subclass that reads models in the format used in the
    `Garching Supernova Archive <https://wwwmpa.mpa-garching.mpg.de/ccsnarchive/>`_."""
    def __init__(self, filename, eos='LS220', metadata={}):
        """Model Initialization.

        Parameters
        ----------
        filename : str
            Absolute or relative path to file with model data, we add nue/nuebar/nux.  This argument will be deprecated.
        eos: str
            Equation of state. Valid value is 'LS220'. This argument will be deprecated.

        Other Parameters
        ----------------
        progenitor_mass: astropy.units.Quantity
            Mass of model progenitor in units Msun. Valid values are {progenitor_mass}.
        Raises
        ------
        FileNotFoundError
            If a file for the chosen model parameters cannot be found
        ValueError
            If a combination of parameters is invalid when loading from parameters
        """
        # Read through the several ASCII files for the chosen simulation and
        # merge the data into one giant table.
        mergtab = None
        for flavor in Flavor:
            _flav = Flavor.NU_X if flavor == Flavor.NU_X_BAR else flavor
            _sfx = _flav.name.replace('_', '').lower()
            _filename = '{}_{}_{}'.format(filename, eos, _sfx)
            _lname = 'L_{}'.format(flavor.name)
            _ename = 'E_{}'.format(flavor.name)
            _e2name = 'E2_{}'.format(flavor.name)
            _aname = 'ALPHA_{}'.format(flavor.name)

            # Open the requested filename using the model downloader.
            datafile = self.request_file(_filename)

            simtab = Table.read(datafile,
                                names=['TIME', _lname, _ename, _e2name],
                                format='ascii')
            simtab['TIME'].unit = 's'
            simtab[_lname].unit = '1e51 erg/s'
            simtab[_aname] = (2*simtab[_ename]**2 - simtab[_e2name]) / (simtab[_e2name] - simtab[_ename]**2)
            simtab[_ename].unit = 'MeV'
            del simtab[_e2name]

            if mergtab is None:
                mergtab = simtab
            else:
                mergtab = join(mergtab, simtab, keys='TIME', join_type='left')
                mergtab[_lname].fill_value = 0.
                mergtab[_ename].fill_value = 0.
                mergtab[_aname].fill_value = 0.
        simtab = mergtab.filled()
        if not metadata:
            metadata = {
                'Progenitor mass': float(os.path.basename(filename).split('s')[1].split('c')[0]) * u.Msun,
                'EOS': eos,
            }
        super().__init__(simtab, metadata)

class Nakazato_2013(PinchedModel):
    def __init__(self, filename, metadata={}):
        """Model initialization.

        Parameters
        ----------
        filename : str
            Absolute or relative path to FITS file with model data.

        Raises
        ------
        FileNotFoundError
            If a file for the chosen model parameters cannot be found
        """
        # Open the requested filename using the model downloader.
        datafile = self.request_file(filename)
        # Read FITS table using the astropy reader.
        simtab = Table.read(datafile)

        self.filename = os.path.basename(filename)
        super().__init__(simtab, metadata)


class Sukhbold_2015(Nakazato_2013):
    pass


class Tamborra_2014(GarchingArchiveModel):
    pass


class Bollig_2016(GarchingArchiveModel):
    pass


class Walk_2018(GarchingArchiveModel):
    pass


class Walk_2019(GarchingArchiveModel):
    pass


class OConnor_2013(PinchedModel):
    """Model based on the black hole formation simulation in `O'Connor & Ott (2013) <https://arxiv.org/abs/1207.1100>`_.
    """

    def __init__(self, filename, metadata={}):
        """
        Parameters
        ----------
        filename : str
            Absolute or relative path to FITS file with model data.
        """
        datafile = self.request_file(filename)
        # Open luminosity file.
        with tarfile.open(datafile) as tf:
            # Extract luminosity data.
            dataname = 's{:d}_{}_timeseries.dat'.format(int(metadata['Progenitor mass'].value), metadata['EOS'])
            # Read FITS table using the astropy reader.
            simtab = ascii.read(tf.extractfile(dataname), names=['TIME', 'L_NU_E', 'L_NU_E_BAR', 'L_NU_X',
                                                                    'E_NU_E', 'E_NU_E_BAR', 'E_NU_X',
                                                                    'RMS_NU_E', 'RMS_NU_E_BAR', 'RMS_NU_X'])

        simtab['ALPHA_NU_E'] = (2.0 * simtab['E_NU_E'] ** 2 - simtab['RMS_NU_E'] ** 2) / (
                simtab['RMS_NU_E'] ** 2 - simtab['E_NU_E'] ** 2)
        simtab['ALPHA_NU_E_BAR'] = (2.0 * simtab['E_NU_E_BAR'] ** 2 - simtab['RMS_NU_E_BAR'] ** 2) / (
                simtab['RMS_NU_E_BAR'] ** 2 - simtab['E_NU_E_BAR'] ** 2)
        simtab['ALPHA_NU_X'] = (2.0 * simtab['E_NU_X'] ** 2 - simtab['RMS_NU_X'] ** 2) / (
                simtab['RMS_NU_X'] ** 2 - simtab['E_NU_X'] ** 2)

        # note, here L_NU_X is already divided by 4
        super().__init__(simtab, metadata)


class OConnor_2015(PinchedModel):
    """Model based on the black hole formation simulation in `O'Connor (2015) <https://arxiv.org/abs/1411.7058>`_.
    """

    def __init__(self, filename, metadata={}):
        """
        Parameters
        ----------
        filename : str
            Absolute or relative path to FITS file with model data.
        """

        datafile = self.request_file(filename)
        simtab = Table.read(datafile,
                            names=['TIME', 'L_NU_E', 'L_NU_E_BAR', 'L_NU_X',
                                    'E_NU_E', 'E_NU_E_BAR', 'E_NU_X',
                                    'RMS_NU_E', 'RMS_NU_E_BAR', 'RMS_NU_X'],
                            format='ascii')

        header = ascii.read(simtab.meta['comments'], delimiter='=', format='no_header', names=['key', 'val'])
        tbounce = float(header['val'][0])
        simtab['TIME'] -= tbounce

        simtab['ALPHA_NU_E'] = (2.0*simtab['E_NU_E']**2 - simtab['RMS_NU_E']**2) / \
            (simtab['RMS_NU_E']**2 - simtab['E_NU_E']**2)
        simtab['ALPHA_NU_E_BAR'] = (2.0*simtab['E_NU_E_BAR']**2 - simtab['RMS_NU_E_BAR']**2) / \
            (simtab['RMS_NU_E_BAR']**2 - simtab['E_NU_E_BAR']**2)
        simtab['ALPHA_NU_X'] = (2.0*simtab['E_NU_X']**2 - simtab['RMS_NU_X']**2) / \
            (simtab['RMS_NU_X']**2 - simtab['E_NU_X']**2)

        # SYB: double-check on this factor of 4. Should be factor of 2?
        simtab['L_NU_X'] /= 4.0

        # prevent negative lums
        simtab['L_NU_E'][simtab['L_NU_E'] < 0] = 1
        simtab['L_NU_E_BAR'][simtab['L_NU_E_BAR'] < 0] = 1
        simtab['L_NU_X'][simtab['L_NU_X'] < 0] = 1

        self.filename = os.path.basename(filename)

        super().__init__(simtab, metadata)


class Zha_2021(OConnor_2015):
    pass

class Warren_2020(PinchedModel):
    def __init__(self, filename, metadata={}):
        """
        Parameters
        ----------
        filename : str
            Absolute or relative path to file prefix, we add nue/nuebar/nux
        """
        # Open the requested filename using the model downloader.
        datafile = self.request_file(filename)

        # Open luminosity file.
        # Read data from HDF5 files, then store.
        f = h5py.File(datafile, 'r')

        simtab = Table()

        for i in range(len(f['nue_data']['lum'])):
            if f['sim_data']['shock_radius'][i][1] > 0.00001:
                bounce = f['sim_data']['shock_radius'][i][0]
                break

        simtab['TIME'] = f['nue_data']['lum'][:, 0] - bounce
        simtab['L_NU_E'] = f['nue_data']['lum'][:, 1] * 1e51
        simtab['L_NU_E_BAR'] = f['nuae_data']['lum'][:, 1] * 1e51
        simtab['L_NU_X'] = f['nux_data']['lum'][:, 1] * 1e51
        simtab['E_NU_E'] = f['nue_data']['avg_energy'][:, 1]
        simtab['E_NU_E_BAR'] = f['nuae_data']['avg_energy'][:, 1]
        simtab['E_NU_X'] = f['nux_data']['avg_energy'][:, 1]
        simtab['RMS_NU_E'] = f['nue_data']['rms_energy'][:, 1]
        simtab['RMS_NU_E_BAR'] = f['nuae_data']['rms_energy'][:, 1]
        simtab['RMS_NU_X'] = f['nux_data']['rms_energy'][:, 1]

        simtab['ALPHA_NU_E'] = (2.0 * simtab['E_NU_E'] ** 2 - simtab['RMS_NU_E'] ** 2) / \
            (simtab['RMS_NU_E'] ** 2 - simtab['E_NU_E'] ** 2)
        simtab['ALPHA_NU_E_BAR'] = (2.0 * simtab['E_NU_E_BAR'] ** 2 - simtab['RMS_NU_E_BAR']
                                    ** 2) / (simtab['RMS_NU_E_BAR'] ** 2 - simtab['E_NU_E_BAR'] ** 2)
        simtab['ALPHA_NU_X'] = (2.0 * simtab['E_NU_X'] ** 2 - simtab['RMS_NU_X'] ** 2) / \
            (simtab['RMS_NU_X'] ** 2 - simtab['E_NU_X'] ** 2)

        # Set model metadata.
        self.filename = os.path.basename(filename)

        super().__init__(simtab, metadata)


class Kuroda_2020(PinchedModel):
    def __init__(self, filename, metadata={}):
        """
        Parameters
        ----------
        filename : str
            Absolute or relative path to file prefix, we add nue/nuebar/nux
        """

        # Open the requested filename using the model downloader.
        datafile = self.request_file(filename)
        # Read ASCII data.
        simtab = Table.read(datafile, format='ascii')

        # Get grid of model times.
        simtab['TIME'] = simtab['Tpb[ms]'] << u.ms
        for f in [Flavor.NU_E, Flavor.NU_E_BAR, Flavor.NU_X]:
            fkey = re.sub('(E|X)_BAR', r'A\g<1>', f.name).lower()
            simtab[f'L_{f.name}'] = simtab[f'<L{fkey}>'] * 1e51 << u.erg / u.s
            simtab[f'E_{f.name}'] = simtab[f'<E{fkey}>'] << u.MeV
            # There is no pinch parameter so use alpha=2.0.
            simtab[f'ALPHA_{f.name}'] = np.full_like(simtab[f'E_{f.name}'].value, 2.)

        self.filename = os.path.basename(filename)

        super().__init__(simtab, metadata)


class Fornax_2019(SupernovaModel):
    def __init__(self, filename, metadata={}, cache_flux=False):
        """
        Parameters
        ----------
        filename : str
            Absolute or relative path to FITS file with model data.
        cache_flux : bool
            If true, pre-compute the flux on a fixed angular grid and store the values in a FITS file.
        """
        # Set up model metadata.
        self.filename = filename
        self.metadata = metadata

        self.fluxunit = 1e50 * u.erg/(u.s*u.MeV)
        self.time = None

        # Read a cached flux file in FITS format or generate one.
        self.is_cached = cache_flux and 'healpy' in sys.modules
        if cache_flux and not 'healpy' in sys.modules:
            logger = logging.getLogger()
            logger.warning("No module named 'healpy'. Cannot enable caching.")

        if self.is_cached:

            self.E = {}
            self.dE = {}
            self.dLdE = {}
            self.luminosity = {}

            # Check if we're initializing on a FITS file or not.
            if filename.endswith('.fits'):
                fitsfile = filename
            else:
                fitsfile = filename.replace('h5', 'fits')

            if os.path.exists(fitsfile):
                self._read_fits(fitsfile)
                ntim, nene, npix = self.dLdE[Flavor.NU_E].shape
                self.npix = npix
                self.nside = hp.npix2nside(npix)
            else:
                with h5py.File(filename, 'r') as _h5file:
                    # Conversion of flavor to key name in the model HDF5 file.
                    self._flavorkeys = {Flavor.NU_E: 'nu0',
                                        Flavor.NU_E_BAR: 'nu1',
                                        Flavor.NU_X: 'nu2',
                                        Flavor.NU_X_BAR: 'nu2'}

                    if self.time is None:
                        self.time = _h5file['nu0']['g0'].attrs['time'] * u.s

                    # Use a HEALPix grid with nside=4 (192 pixels) to cache the
                    # values of Y_lm(theta, phi).
                    self.nside = 4
                    self.npix = hp.nside2npix(self.nside)
                    thetac, phic = hp.pix2ang(self.nside, np.arange(self.npix))

                    Ylm = {}
                    for l in range(3):
                        Ylm[l] = {}
                        for m in range(-l, l+1):
                            Ylm[l][m] = self._real_sph_harm(l, m, thetac, phic)

                    # Store 3D tables of dL/dE for each flavor.
                    logger = logging.getLogger()
                    for flavor in Flavor:

                        key = self._flavorkeys[flavor]
                        logger.info('Caching {} for {} ({})'.format(filename, str(flavor), key))

                        # HDF5 file only contains NU_E, NU_E_BAR, and NU_X.
                        if flavor == Flavor.NU_X_BAR:
                            self.E[flavor] = self.E[Flavor.NU_X]
                            self.dE[flavor] = self.dE[Flavor.NU_X]
                            self.dLdE[flavor] = self.dLdE[Flavor.NU_X]
                            self.luminosity[flavor] = self.luminosity[Flavor.NU_X]
                            continue

                        self.E[flavor] = _h5file[key]['egroup'][()] * u.MeV
                        self.dE[flavor] = _h5file[key]['degroup'][()] * u.MeV

                        ntim, nene = self.E[flavor].shape
                        self.dLdE[flavor] = np.zeros((ntim, nene, self.npix), dtype=float)
                        # Loop over time bins.
                        for i in range(ntim):
                            # Loop over energy bins.
                            for j in range(nene):
                                dLdE_ij = 0.
                                # Sum over multipole moments.
                                for l in range(3):
                                    for m in range(-l, l+1):
                                        dLdE_ij += _h5file[key]['g{}'.format(j)
                                                                ]['l={} m={}'.format(l, m)][i] * Ylm[l][m]
                                self.dLdE[flavor][i][j] = dLdE_ij

                        # Integrate over energy to get L(t).
                        factor = 1. if flavor.is_electron else 0.25
                        self.dLdE[flavor] = self.dLdE[flavor] * factor * self.fluxunit
                        self.dLdE[flavor] = self.dLdE[flavor].to('erg/(s*MeV)')

                        self.luminosity[flavor] = np.sum(self.dLdE[flavor] * self.dE[flavor][:, :, np.newaxis], axis=1)

                    # Write output to FITS.
                    self._write_fits(fitsfile, overwrite=True)
        else:
            # Conversion of flavor to key name in the model HDF5 file.
            self._flavorkeys = {Flavor.NU_E: 'nu0',
                                Flavor.NU_E_BAR: 'nu1',
                                Flavor.NU_X: 'nu2',
                                Flavor.NU_X_BAR: 'nu2'}

            # Open the requested filename using the model downloader.
            datafile = self.request_file(filename)
            # Open HDF5 data file.
            self._h5file = h5py.File(datafile, 'r')

            # Get grid of model times in seconds.
            self.time = self._h5file['nu0']['g0'].attrs['time'] * u.s

    def _read_fits(self, filename):
        """Read cached angular data from FITS.

        Parameters
        ----------
        filename : str
            Input filename.
        """
        hdus = fits.open(filename)

        self.time = hdus['TIME'].data * u.Unit(hdus['TIME'].header['BUNIT'])

        for flavor in Flavor:
            name = str(flavor).split('.')[-1]

            ext = '{}_ENERGY'.format(name)
            self.E[flavor] = hdus[ext].data * u.Unit(hdus[ext].header['BUNIT'])

            ext = '{}_DE'.format(name)
            self.dE[flavor] = hdus[ext].data * u.Unit(hdus[ext].header['BUNIT'])

            ext = '{}_FLUX'.format(name)
            self.dLdE[flavor] = hdus[ext].data * u.Unit(hdus[ext].header['BUNIT'])
            self.dLdE[flavor] = self.dLdE[flavor].to('erg/(s*MeV)')

            self.luminosity[flavor] = np.sum(self.dLdE[flavor] * self.dE[flavor][:, :, np.newaxis], axis=1)

    def _write_fits(self, filename, overwrite=False):
        """Write angular-dependent calculated flux in FITS format.

        Parameters
        ----------
        filename : str
            Output filename.
        """
        hx = fits.HDUList()

        hdu_time = fits.PrimaryHDU(self.time.to_value('s'))
        hdu_time.header['EXTNAME'] = 'TIME'
        hdu_time.header['BUNIT'] = 'second'
        hx.append(hdu_time)

        for flavor in Flavor:
            name = str(flavor).split('.')[-1]

            hdu_E = fits.ImageHDU(self.E[flavor].to_value('MeV'))
            hdu_E.header['EXTNAME'] = '{}_ENERGY'.format(name)
            hdu_E.header['BUNIT'] = 'MeV'
            hx.append(hdu_E)

            hdu_dE = fits.ImageHDU(self.dE[flavor].to_value('MeV'))
            hdu_dE.header['EXTNAME'] = '{}_DE'.format(name)
            hdu_dE.header['BUNIT'] = 'MeV'
            hx.append(hdu_dE)

            hdu_flux = fits.ImageHDU(self.dLdE[flavor].to_value(str(self.fluxunit)))
            hdu_flux.header['EXTNAME'] = '{}_FLUX'.format(name)
            hdu_flux.header['BUNIT'] = str(self.fluxunit)
            hx.append(hdu_flux)

        hx.writeto(filename, overwrite=overwrite)

    def _fact(self, n):
        """Calculate n!.

        Parameters
        ----------
        n : int or float
            Input for computing n factorial.

        Returns
        -------
        factorial : float
            Factorial n!, computed as Gamma(n+1).
        """
        return gamma(n + 1.)

    def _real_sph_harm(self, l, m, theta, phi):
        """Compute orthonormalized real (tesseral) spherical harmonics Y_lm.

        Parameters
        ----------
        l : int
            Degree of the spherical harmonics.
        m : int
            Order of the spherical harmonics.
        theta : float or ndarray
            Input zenith angles.
        phi : float or ndarray
            Input azimuth angles.

        Returns
        -------
        Y_lm : float or ndarray
            Real-valued spherical harmonic function at theta, phi.
        """
        if m < 0:
            norm = np.sqrt((2*l + 1.)/(2*np.pi)*self._fact(l + m)/self._fact(l - m))
            return norm * lpmv(-m, l, np.cos(theta)) * np.sin(-m*phi)
        elif m == 0:
            norm = np.sqrt((2*l + 1.)/(4*np.pi))
            return norm * lpmv(0, l, np.cos(theta)) * np.ones_like(phi)
        else:
            norm = np.sqrt((2*l + 1.)/(2*np.pi)*self._fact(l - m)/self._fact(l + m))
            return norm * lpmv(m, l, np.cos(theta)) * np.cos(m*phi)


    # def _get_binnedspectra(self, t, theta=None, phi=None): # Made theta, phi optional
    #     E = {}
    #     dE = {}
    #     binspec = {}

    #     # convert input time to a time index.
    #     current_time_val = t
    #     if hasattr(t, 'shape') and t.shape: # if t is an array, take the first element
    #         current_time_val = t[0] 

    #     j = (np.abs(current_time_val.to(self.time.unit) - self.time)).argmin()

    #     is_angle_averaged_calculation = (theta is None or phi is None)

    #     for flavor in Flavor:
    #         if self.is_cached:
    #             _theta_val_rad_for_cache = 0.0
    #             _phi_val_rad_for_cache = 0.0
    #             if is_angle_averaged_calculation:
    #                 # For cached data, if true angle-average isn't cached as a pixel, this defaults to a specific direction.
    #                 # This might need a more sophisticated handling if cached 3D data is to be used for 1D analysis.
    #                 logger.warning(f"Fornax_2019._get_binnedspectra: theta/phi not provided for cached data. Using default direction (theta=0, phi=0). This may not be a true angle average.")
    #             else:
    #                 _theta_val_rad_for_cache = theta.to_value('radian')
    #                 _phi_val_rad_for_cache = phi.to_value('radian')

    #             k = hp.ang2pix(self.nside, _theta_val_rad_for_cache, _phi_val_rad_for_cache)
    #             E[flavor] = self.E[flavor][j]
    #             dE[flavor] = self.dE[flavor][j]
    #             binspec[flavor] = self.dLdE[flavor][j, :, k]
    #         else: # Read the HDF5 input file directly
    #             if flavor == Flavor.NU_X_BAR:
    #                 E[flavor] = E[Flavor.NU_X]
    #                 dE[flavor] = dE[Flavor.NU_X]
    #                 binspec[flavor] = binspec[Flavor.NU_X]
    #                 continue

    #             key = self._flavorkeys[flavor]
    #             E[flavor] = self._h5file[key]['egroup'][j] * u.MeV
    #             dE[flavor] = self._h5file[key]['degroup'][j] * u.MeV
    #             dLdE_values = np.zeros(len(E[flavor]), dtype=float)

    #             for ebin in range(len(E[flavor])):
    #                 if is_angle_averaged_calculation:
    #                     # Calculate angle-averaged dL/dE using F_00 coefficient
    #                     # L(E) = sqrt(4*pi) * F_00(E)
    #                     F_00_at_ebin_j = self._h5file[key]['g{}'.format(ebin)]['l=0 m=0'][j]
    #                     dLdE_values[ebin] = F_00_at_ebin_j * np.sqrt(4 * np.pi)
    #                 else:
    #                     # Directional calculation if theta and phi are provided
    #                     _theta_val_rad_sph = theta.to_value('radian')
    #                     _phi_val_rad_sph = phi.to_value('radian')
    #                     dLdE_ebin_val = 0.0
    #                     for l_val in range(3): # Sum up to l=2
    #                         for m_val in range(-l_val, l_val + 1):
    #                             Ylm_val = self._real_sph_harm(l_val, m_val, _theta_val_rad_sph, _phi_val_rad_sph)
    #                             dLdE_ebin_val += self._h5file[key]['g{}'.format(ebin)]['l={} m={}'.format(l_val, m_val)][j] * Ylm_val
    #                     dLdE_values[ebin] = dLdE_ebin_val

    #             factor = 1. if flavor.is_electron else 0.25 # Account for nu_x representing 4 species vs 1.
    #             binspec[flavor] = dLdE_values * factor * self.fluxunit
    #             binspec[flavor] = binspec[flavor].to('erg/(s*MeV)')
    #     return E, dE, binspec

    def _get_binned_spectra_at_single_time(self, t_scalar_astropy, theta=None, phi=None):
        """
        Helper function to get binned spectra for a single scalar time point.
        Returns E_dict, dE_dict, binspec_dict where values are 1D arrays (N_model_energy_bins).
        """
        E_single_t = {}
        dE_single_t = {} # You might not need dE if your model energies are fixed per flavor
        binspec_single_t = {}

        # Find the closest model time index 'j' for the given scalar t_scalar_astropy
        j = (np.abs(t_scalar_astropy.to(self.time.unit) - self.time)).argmin()

        is_angle_averaged_calculation = (theta is None or phi is None)

        for flavor in Flavor:
            if self.is_cached:
                _theta_val_rad_for_cache = 0.0
                _phi_val_rad_for_cache = 0.0
                if is_angle_averaged_calculation:
                    logger.warning(f"Fornax_2019._get_binned_spectra_at_single_time: theta/phi not provided for cached data. Using default direction (theta=0, phi=0). This may not be a true angle average.")
                else:
                    _theta_val_rad_for_cache = theta.to_value('radian')
                    _phi_val_rad_for_cache = phi.to_value('radian')
                
                k_pix = hp.ang2pix(self.nside, _theta_val_rad_for_cache, _phi_val_rad_for_cache)
                # self.E[flavor] and self.dLdE[flavor] are (Ntime_model, Nenergy_model_bins, [Npix_cached])
                E_single_t[flavor] = self.E[flavor][j] 
                dE_single_t[flavor] = self.dE[flavor][j] # Assuming dE is also (Ntime_model, Nenergy_model_bins)
                binspec_single_t[flavor] = self.dLdE[flavor][j, :, k_pix] # This should be 1D (Nenergy_model_bins)
            else: # Read HDF5
                if flavor == Flavor.NU_X_BAR:
                    E_single_t[flavor] = E_single_t[Flavor.NU_X]
                    dE_single_t[flavor] = dE_single_t[Flavor.NU_X]
                    binspec_single_t[flavor] = binspec_single_t[Flavor.NU_X]
                    continue

                key = self._flavorkeys[flavor]
                # self._h5file[key]['egroup'] is (Ntime_model, Nenergy_model_bins)
                E_single_t[flavor] = self._h5file[key]['egroup'][j] * u.MeV 
                dE_single_t[flavor] = self._h5file[key]['degroup'][j] * u.MeV
                dLdE_values_for_ebins = np.zeros(len(E_single_t[flavor]), dtype=float) # For this time j, iterate ebins

                for ebin in range(len(E_single_t[flavor])): # E_single_t[flavor] is 1D for current time j
                    if is_angle_averaged_calculation:
                        F_00_at_ebin_j = self._h5file[key]['g{}'.format(ebin)]['l=0 m=0'][j]
                        dLdE_values_for_ebins[ebin] = F_00_at_ebin_j * np.sqrt(4 * np.pi)
                    else:
                        _theta_val_rad_sph = theta.to_value('radian')
                        _phi_val_rad_sph = phi.to_value('radian')
                        dLdE_ebin_val_sum = 0.0
                        # Coefficients g{ebin}['l=L m=M'] are (Ntime_model,)
                        for l_val in range(3):
                            for m_val in range(-l_val, l_val + 1):
                                Ylm_val = self._real_sph_harm(l_val, m_val, _theta_val_rad_sph, _phi_val_rad_sph)
                                dLdE_ebin_val_sum += self._h5file[key]['g{}'.format(ebin)]['l={} m={}'.format(l_val, m_val)][j] * Ylm_val
                        dLdE_values_for_ebins[ebin] = dLdE_ebin_val_sum
                
                factor = 1. if flavor.is_electron else 0.25
                binspec_single_t[flavor] = dLdE_values_for_ebins * factor * self.fluxunit # This is now 1D (Nenergy_model_bins)
                # binspec_single_t[flavor] = binspec_single_t[flavor].to('erg/(s*MeV)') # Ensure unit
        return E_single_t, dE_single_t, binspec_single_t


    def get_initial_spectra(self, t, E, theta = None, phi = None, flavors=Flavor, interpolation='linear'):
        """Get neutrino spectra/luminosity curves before flavor transformation.

        Parameters
        ----------
        t : astropy.Quantity
            Time to evaluate initial spectra.
        E : astropy.Quantity or ndarray of astropy.Quantity
            Energies to evaluate the initial spectra.
        theta : astropy.Quantity
            Zenith angle of the spectral emission.
        phi : astropy.Quantity
            Azimuth angle of the spectral emission.
        flavors: iterable of snewpy.neutrino.Flavor
            Return spectra for these flavors only (default: all)
        interpolation : str
            Scheme to interpolate in spectra ('nearest', 'linear').

        Returns
        -------
        initialspectra : dict
            Dictionary of model spectra, keyed by neutrino flavor.
        """
        # Ensure t is an array for iteration, even if a scalar is passed initially
        t_array = t if hasattr(t, 'shape') and t.shape and len(t.shape)>0 else u.Quantity([t])
        
        ntimes = len(t_array)
        nenergies_target = len(E) # Target number of energy bins for output

        # Initialize dictionary to hold the (Ntime x Nenergy_target) arrays for each flavor
        initialspectra_all_times = {f: u.Quantity(np.zeros((ntimes, nenergies_target)), self.fluxunit.unit) for f in flavors}

        E_target_MeV_np = E.to_value(u.MeV) # Target E bins for interpolation (1D numpy array)
        logE_target_np = np.log10(np.where(E_target_MeV_np == 0, np.finfo(float).eps, E_target_MeV_np))

        for i, t_scalar_i in enumerate(t_array): # Iterate over each requested time point
            # Get the model's binned spectra (1D arrays) for this specific time t_scalar_i
            _E_model_dict_single_t, _, _spec_model_dict_single_t = self._get_binned_spectra_at_single_time(t_scalar_i, theta, phi)

            for flavor in flavors:
                _E_model_flavor_np = _E_model_dict_single_t[flavor].to_value('MeV') # 1D (N_model_energy_bins)
                _spec_model_flavor_np = _spec_model_dict_single_t[flavor].to_value(self.fluxunit.unit) # 1D (N_model_energy_bins)

                if interpolation.lower() == 'linear':
                    _logE_model_np = np.log10(np.where(_E_model_flavor_np == 0, np.finfo(float).eps, _E_model_flavor_np))
                    
                    # Pad for interpolation to handle energies outside model's E range
                    _logE_padded = np.concatenate(([np.log10(np.finfo(float).eps)], _logE_model_np, [_logE_model_np[-1] + (_logE_model_np[-1]-_logE_model_np[-2] if len(_logE_model_np)>1 else 1.0)]))
                    _spec_padded = np.concatenate(([0.], _spec_model_flavor_np, [0.]))
                    
                    interpolated_flux_at_time_i = np.interp(logE_target_np, _logE_padded, _spec_padded)
                    initialspectra_all_times[flavor][i, :] = interpolated_flux_at_time_i * self.fluxunit.unit
                
                elif interpolation.lower() == 'nearest':
                    _logE_model_np = np.log10(np.where(_E_model_flavor_np == 0, np.finfo(float).eps, _E_model_flavor_np))
                    
                    if len(_logE_model_np) > 1:
                        _dlogE_half_np = np.diff(_logE_model_np) / 2.0
                        _logE_bin_edges_np = np.concatenate(([_logE_model_np[0] - _dlogE_half_np[0]], _logE_model_np[:-1] + _dlogE_half_np, [_logE_model_np[-1] + _dlogE_half_np[-1]]))
                    elif len(_logE_model_np) == 1: # Single energy bin in model
                        _logE_bin_edges_np = np.array([_logE_model_np[0] - 0.1, _logE_model_np[0] + 0.1]) # Arbitrary small range
                    else: # No energy bins in model (empty spectrum)
                        _logE_bin_edges_np = np.array([np.log10(np.finfo(float).eps), np.log10(np.finfo(float).eps*2)])

                    indices = np.searchsorted(_logE_bin_edges_np, logE_target_np, side='right') - 1
                    indices = np.clip(indices, 0, len(_spec_model_flavor_np)-1 if len(_spec_model_flavor_np)>0 else 0)

                    if len(_spec_model_flavor_np) > 0:
                        selected_flux_values = _spec_model_flavor_np[indices]
                    else:
                        selected_flux_values = np.zeros(nenergies_target)
                        
                    initialspectra_all_times[flavor][i, :] = selected_flux_values * self.fluxunit.unit
                else:
                    raise ValueError(f'Unrecognized interpolation type "{interpolation}"')

        # If the original t was scalar, we should return 1D arrays (Nenergy,)
        # The calling `get_transformed_spectra` in base.py usually handles the stacking if t was an array.
        # Let's ensure this function returns what get_transformed_spectra expects.
        # If t (original input) was scalar, then t_array had len 1. So return [0,:] slice.
        if t.isscalar: # Check original t
            return {f: initialspectra_all_times[f][0,:] for f in flavors}
        else:
            return initialspectra_all_times


# class Fornax_2022(Fornax_2021):
#     def __init__(self, filename, metadata={}):
#         """
#         Parameters
#         ----------
#         filename : str
#             Absolute or relative path to HDF5 file with model data.
#         """
#         # Open the requested filename using the model downloader.
#         datafile = self.request_file(filename)
#         # Set up model metadata.
#         self.progenitor = os.path.splitext(os.path.basename(filename))[0].split('_')[2]
#         self.progenitor_mass = float(self.progenitor[:-3])*u.Msun if self.progenitor.endswith('bh') else float(self.progenitor)*u.Msun
#
#         self.metadata = metadata
#
#         # Open HDF5 data file.
#         _h5file = h5py.File(datafile, 'r')
#
#         self.metadata['PNS mass'] = _h5file.attrs['Mpns'] * u.Msun
#         self.time = _h5file['nu0'].attrs['time'] * u.s
#
#         self.luminosity = {}
#         self._E = {}
#         self._dLdE = {}
#         for flavor in Flavor:
#             # Convert flavor to key name in the model HDF5 file
#             key = {Flavor.NU_E: 'nu0',
#                    Flavor.NU_E_BAR: 'nu1',
#                    Flavor.NU_X: 'nu2',
#                    Flavor.NU_X_BAR: 'nu2'}[flavor]
#
#             self._E[flavor] = np.asarray(_h5file[key]['egroup'])
#             self._dLdE[flavor] = {f"g{i}": np.asarray(_h5file[key][f'g{i}']) for i in range(12)}
#
#             # Compute luminosity by integrating over model energy bins.
#             dE = np.asarray(_h5file[key]['degroup'])
#             n = len(dE[0])
#             dLdE = np.zeros((len(self.time), n), dtype=float)
#             for i in range(n):
#                 dLdE[:, i] = self._dLdE[flavor][f"g{i}"]
#
#             # Note factor of 0.25 in nu_x and nu_x_bar.
#             factor = 1. if flavor.is_electron else 0.25
#             self.luminosity[flavor] = np.sum(dLdE*dE, axis=1) * factor * 1e50 * u.erg/u.s

class Fornax_2022(Fornax_2019):
    def __init__(self, filename, metadata={}):
        """
        Parameters
        ----------
        filename : str
            Absolute or relative path to model data file (.txt or .h5).
        metadata : dict
            Additional metadata for the model.
        """
        # determine file type (.txt or .h5)
        file_extension = os.path.splitext(filename)[-1]

        if file_extension == '.txt':
            # for .txt files, use the new data loading method
            self.load_txt_data(filename)
        elif file_extension == '.h5':
            # for HDF5 files, use the original Fornax loading mechanism
            self.load_hdf5_data(filename)

        self.metadata = metadata

    def load_txt_data(self, filename, distance=3.086e22):
        """
        Load data from a .txt file format with time and strain values.

        Parameters
        ----------
        filename : str
            Path to the .txt file with model data.
        distance : float
            Source distance in cm (default is 10 kpc).
        """

        # Read data from the text file, assuming there is space/tab separation
        # Each row corresponds to [time, hp_nu0, hp_nu1, hp_nu2, hx_nu0, hx_nu1, hx_nu2]
        data = np.loadtxt(filename, skiprows=1)

        # Extract the time and strain values from the datafile

        self.time = data[:, 0] * u.s # time in seconds
        hp_values = data[:, 1:4] # hp strain values for nu0, nu1, nu2
        hx_values = data[:, 4:7] # hx strain values for nu0, nu1, nu2

        # Normalize by dividing these strains by source distance
        self.hp_strain = hp_values / distance
        self.hx_strain = hx_values / distance

        # Option to store strain values in Dictionary
        self.strain = {
            Flavor.NU_E: self.hp_strain[:, 0],   # nu0 hp strain
            Flavor.NU_E_BAR: self.hp_strain[:, 1],  # nu1 hp strain
            Flavor.NU_X: self.hp_strain[:, 2],  # nu2 hp strain
            Flavor.NU_X_BAR: self.hx_strain[:, 2],  # nu2 hx strain (considered as a placeholder for both nu_x and nu_x_bar)
        }




        ## WORK IN PROGRESS

    def load_hdf5_data(self, filename):
        """
        Load data from an HDF5 file (original Fornax data format).

        Parameters
        ----------
        filename : str
            Path to the HDF5 file with model data.
        """
        ## COPIED FROM ABOVE
        # open the requested filename using the model downloader.
        datafile = self.request_file(filename)

        # set up model metadata
        self.progenitor = os.path.splitext(os.path.basename(filename))[0].split('_')[2]
        self.progenitor_mass = float(self.progenitor[:-3])*u.Msun if self.progenitor.endswith('bh') else float(self.progenitor)*u.Msun

        # open HDF5 data file
        _h5file = h5py.File(datafile, 'r')

        self.metadata['PNS mass'] = _h5file.attrs['Mpns'] * u.Msun
        self.time = _h5file['nu0'].attrs['time'] * u.s

        self.luminosity = {}
        self._E = {}
        self._dLdE = {}
        for flavor in Flavor:
            # convert flavor to key name in the model HDF5 file
            key = {Flavor.NU_E: 'nu0',
                   Flavor.NU_E_BAR: 'nu1',
                   Flavor.NU_X: 'nu2',
                   Flavor.NU_X_BAR: 'nu2'}[flavor]

            self._E[flavor] = np.asarray(_h5file[key]['egroup'])
            self._dLdE[flavor] = {f"g{i}": np.asarray(_h5file[key][f'g{i}']) for i in range(12)}

            # compute luminosity by integrating over model energy bins
            dE = np.asarray(_h5file[key]['degroup'])
            n = len(dE[0])
            dLdE = np.zeros((len(self.time), n), dtype=float)
            for i in range(n):
                dLdE[:, i] = self._dLdE[flavor][f"g{i}"]

            # note factor of 0.25 in nu_x and nu_x_bar
            factor = 1. if flavor.is_electron else 0.25
            self.luminosity[flavor] = np.sum(dLdE*dE, axis=1) * factor * 1e50 * u.erg/u.s

    def get_strain(self, polarization='hp', neutrino_type='nu0'):
        """
        Retrieve strain values (hp or hx) for the specified neutrino type.

        Parameters
        ----------
        polarization : str
            Either 'hp' or 'hx' (gravitational wave polarization).
        neutrino_type : str
            One of 'nu0', 'nu1', or 'nu2' (neutrino species).

        Returns
        -------
        ndarray : Strain values for the given polarization and neutrino type.
        """
        # Map neutrino type to the correct key in the strain dictionary
        if polarization == 'hp':
            if neutrino_type == 'nu0':
                return self.strain[Flavor.NU_E]
            elif neutrino_type == 'nu1':
                return self.strain[Flavor.NU_E_BAR]
            elif neutrino_type == 'nu2':
                return self.strain[Flavor.NU_X]
        elif polarization == 'hx':
            if neutrino_type == 'nu0':
                return self.hx_strain[:, 0]
            elif neutrino_type == 'nu1':
                return self.hx_strain[:, 1]
            elif neutrino_type == 'nu2':
                return self.hx_strain[:, 2]

        ## WORK IN PROGRESS



class Mori_2023(PinchedModel):
    def __init__(self, filename, metadata={}):
        """
        Parameters
        ----------
        filename : str
            Absolute or relative path to file prefix.
        """
        # Open the requested filename using the model downloader.
        datafile = self.request_file(filename)

        self.metadata = metadata

        # Read ASCII data.
        simtab = Table.read(datafile, format='ascii')

        # Remove the first table row, which appears to have zero input.
        simtab = simtab[simtab['1:t_sim[s]'] > 0]

        # Get grid of model times.
        simtab['TIME'] = simtab['2:t_pb[s]'] << u.s
        for j, (f, fkey) in enumerate(zip([Flavor.NU_E, Flavor.NU_E_BAR, Flavor.NU_X], 'ebx')):
            simtab[f'L_{f.name}'] = simtab[f'{6+j}:Le{fkey}[e/s]'] << u.erg / u.s
            # Compute the pinch parameter from E_rms and E_avg
            # <E^2> / <E>^2 = (2+a)/(1+a), where
            # E_rms^2 = <E^2> - <E>^2.
            Eavg = simtab[f'{9+j}:Em{fkey}[MeV]']
            Erms = simtab[f'{12+j}:Er{fkey}[MeV]']
            E2 = Erms**2 + Eavg**2
            x = E2 / Eavg**2
            alpha = (2-x) / (x-1)

            simtab[f'E_{f.name}'] = Eavg << u.MeV
            simtab[f'E2_{f.name}'] = E2 << u.MeV**2
            simtab[f'ALPHA_{f.name}'] = alpha

#            simtab[f'E_{f.name}'] = simtab[f'{9+j}:Em{fkey}[MeV]'] << u.MeV
#            Erms = simtab[f'{12+j}:Er{fkey}[MeV]'] * u.MeV
#
#            # Compute the pinch parameter from E_rms and E_avg
#            simtab[f'E2_{f.name}'] = Erms**2 + simtab[f'E_{f.name}']**2
#            x = simtab[f'E2_{f.name}'] / simtab[f'E_{f.name}']**2
#            simtab[f'ALPHA_{f.name}'] = (2-x) / (x-1)

        self.filename = os.path.basename(filename)

        super().__init__(simtab, metadata)


class Bugli_2021(PinchedModel):
    """Model based on `Buggli (2021) <https://arxiv.org/abs/2105.00665>`_.
    """

    def __init__(self, filename, metadata={}):
        """
        Parameters
        ----------
        filename : str
            Absolute or relative path to FITS file with model data.
        """

        datafile = self.request_file(filename)
        simtab = Table.read(datafile,
                            names=['TIME', 'L_NU_E', 'L_NU_E_BAR', 'L_NU_X',
                                    'E_NU_E', 'E_NU_E_BAR', 'E_NU_X',
                                    'RMS_NU_E', 'RMS_NU_E_BAR', 'RMS_NU_X'],
                            format='ascii')

        simtab['ALPHA_NU_E'] = (2.0*simtab['E_NU_E']**2 - simtab['RMS_NU_E']**2) / \
            (simtab['RMS_NU_E']**2 - simtab['E_NU_E']**2)
        simtab['ALPHA_NU_E_BAR'] = (2.0*simtab['E_NU_E_BAR']**2 - simtab['RMS_NU_E_BAR']**2) / \
            (simtab['RMS_NU_E_BAR']**2 - simtab['E_NU_E_BAR']**2)
        simtab['ALPHA_NU_X'] = (2.0*simtab['E_NU_X']**2 - simtab['RMS_NU_X']**2) / \
            (simtab['RMS_NU_X']**2 - simtab['E_NU_X']**2)

        self.filename = os.path.basename(filename)

        super().__init__(simtab, metadata)


class Fischer_2020(PinchedModel):
    def __init__(self, filename, metadata={}):
        """
        Parameters
        ----------
        filename : str
            Absolute or relative path to file
        """
        # Open the requested filename using the model downloader.
        datafile = self.request_file(filename)
        self.metadata = metadata

        # Open the requested filename using the model downloader.
        # datafile = _model_downloader.get_model_data(self.__class__.__name__, filename)
        # self.filename = os.path.basename(filename)

        simtab = Table()

        tf = tarfile.open(datafile)

        # Open luminosity file
        with tf.extractfile("luminosity.dat") as Lfile:
             Ldata = np.genfromtxt(Lfile, skip_header=2)

             simtab['TIME'] = Ldata[:, 0]

             simtab['L_NU_E'] = Ldata[:, 1]
             simtab['L_NU_E_BAR'] = Ldata[:, 2]
             simtab['L_NU_X'] = Ldata[:, 3]
             simtab['L_NU_X_BAR'] = Ldata[:, 4]

             Lfile.close()

        # Open mean energy file
        with tf.extractfile("menergy.dat") as Efile:
            Edata = np.genfromtxt(Efile, skip_header=2)

            simtab['E_NU_E'] = Edata[:, 1] << u.MeV
            simtab['E_NU_E_BAR'] = Edata[:, 2] << u.MeV
            simtab['E_NU_X'] = Edata[:, 3] << u.MeV
            simtab['E_NU_X_BAR'] = Edata[:, 4] << u.MeV

            Efile.close()

        # Open rms energy file
        with tf.extractfile("rmsenergy.dat") as RMSEfile:
            RMSEdata = np.genfromtxt(RMSEfile, skip_header=2)

            simtab['RMS_NU_E'] = RMSEdata[:, 1] << u.MeV
            simtab['RMS_NU_E_BAR'] = RMSEdata[:, 2] << u.MeV
            simtab['RMS_NU_X'] = RMSEdata[:, 3] << u.MeV
            simtab['RMS_NU_X_BAR'] = RMSEdata[:, 4] << u.MeV

            RMSEfile.close()

        simtab['ALPHA_NU_E'] = (2.0 * simtab['E_NU_E'] ** 2 - simtab['RMS_NU_E'] ** 2) / \
            (simtab['RMS_NU_E'] ** 2 - simtab['E_NU_E'] ** 2)
        simtab['ALPHA_NU_E_BAR'] = (2.0 * simtab['E_NU_E_BAR'] ** 2 - simtab['RMS_NU_E_BAR'] ** 2) / \
            (simtab['RMS_NU_E_BAR'] ** 2 - simtab['E_NU_E_BAR'] ** 2)
        simtab['ALPHA_NU_X'] = (2.0 * simtab['E_NU_X'] ** 2 - simtab['RMS_NU_X'] ** 2) / \
            (simtab['RMS_NU_X'] ** 2 - simtab['E_NU_X'] ** 2)
        simtab['ALPHA_NU_X_BAR'] = (2.0 * simtab['E_NU_X_BAR'] ** 2 - simtab['RMS_NU_X_BAR'] ** 2) / \
            (simtab['RMS_NU_X_BAR'] ** 2 - simtab['E_NU_X_BAR'] ** 2)

        tf.close()

        super().__init__(simtab, metadata)
