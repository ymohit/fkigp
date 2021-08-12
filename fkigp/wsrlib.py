import numpy as np
import pyart
import warnings
from scipy.interpolate import interp1d, RegularGridInterpolator

import os.path
# import boto3
import tempfile


def aws_parse(name):
    '''
    Parse AWS key into constituent parts

    s = aws_parse(name)

    Parameters
    ----------
    name: string
        The name part of a key, e.g., KBGM20170421_025222 or KBGM20170421_025222_V06
        or KBGM20170421_025222_V06.gz

    Returns
    -------
    s: dict
        A dictionary with fields: station, year, month, day, hour, minute, second.


    See Also
    --------
    aws_key

    Note: the suffix (e.g., '_V06' or '_V06.gz') is deduced from the portion
    of the key that is given and may not be the actual file suffix.
    '''

    name = os.path.basename(name)
    name, ext = os.path.splitext(name)

    # example: KBGM20170421_025222
    return {
        'station': name[0:4],
        'year': int(name[4:8]),
        'month': int(name[8:10]),
        'day': int(name[10:12]),
        'hour': int(name[13:15]),
        'minute': int(name[15:17]),
        'second': int(name[17:19]),
        'suffix': name[19:] + ext
    }


def aws_key(s, suffix=''):
    '''
    Get key for scan

    key, path, name = aws_key(s, suffix)

    Parameters
    ----------
    s: string or struct
        The short name, e.g., KBGM20170421_025222. This can also be a
        dictionary returned by aws_parse
    suffix: string
        Optionally append this to the returned name and key

    Returns
    -------
    key: string
        The full key, e.g., 2017/04/21/KBGM/KBGM20170421_025222
    path: string
        The path, e.g., 2017/04/21/KBGM
    name: string
        The name, e.g., KBGM20170421_025222

    See Also
    --------
    aws_parse
    '''

    if isinstance(s, str):
        s = aws_parse(s)

    path = '%4d/%02d/%02d/%s' % (s['year'],
                                 s['month'],
                                 s['day'],
                                 s['station'])

    name = '%s%04d%02d%02d_%02d%02d%02d' % (s['station'],
                                            s['year'],
                                            s['month'],
                                            s['day'],
                                            s['hour'],
                                            s['minute'],
                                            s['second']);

    suff = suffix or s['suffix']

    key = '%s/%s%s' % (path, name, suff)

    return key


def db(x):
    '''
    Compute decibel transform

    dbx = db( x )

    dbz = 10.*log10(z)
    '''

    return 10 * np.log10(x)


def idb(dbx):
    '''
    Inverse decibel (convert from decibels to linear units)

    x = idb( dbx )

    x = 10**(dbx/10)
    '''
    return 10 ** (dbx / 10)


def z_to_refl(z, wavelength=0.1071):
    '''
    Convert reflectivity factor (Z) to reflectivity (eta)

    eta, db_eta = z_to_refl( z, wavelength )

    Parameters
    ----------
    z: array
        Vector of Z values (reflectivity factor; units: mm^6/m^3)
    wavelength: scalar
        Radar wavelength (units: meters; default = 0.1071 )

    Returns
    -------
    eta: vector
        Reflectivity values (units: cm^2/km^3 )
    db_eta: vector
        Decibels of eta (10.^(eta/10))

    See Also
    --------
    refl_to_z

    Reference:
      Chilson, P. B., W. F. Frick, P. M. Stepanian, J. R. Shipley, T. H. Kunz,
      and J. F. Kelly. 2012. Estimating animal densities in the aerosphere
      using weather radar: To Z or not to Z? Ecosphere 3(8):72.
      http://dx.doi.org/10.1890/ ES12-00027.1


    UNITS
        Z units = mm^6 / m^3
                = 1e-18 m^6 / m^3
                = 1e-18 m^3

        lambda units = m

        eta units = cm^2 / km^3
                  = 1e-4 m^2 / 1e9 m^3
                  = 1e-13 m^-1

    Equation is

               lambda^4
       Z_e = -------------- eta    (units 1e-18 m^3)
              pi^5 |K_m|^2


              pi^5 |K_m|^2
       eta = -------------- Z_e    (units 1e-13 m^-1)
               lambda^4
    '''

    K_m_squared = 0.93
    log_eta = np.log10(z) + 5 * np.log10(np.pi) + np.log10(K_m_squared) - 4 * np.log10(wavelength)

    '''
    Current units: Z / lambda^4 = 1e-18 m^3 / 1 m^4 
                                = 1e-18 m^3 / 1 m^4
                                = 1e-18 m^-1

    Divide by 10^5 to get units 1e-13
    '''

    log_eta = log_eta - 5  # Divide by 10^5

    db_eta = 10 * log_eta
    eta = 10 ** (log_eta)

    return eta, db_eta


def refl_to_z(eta, wavelength=0.1071):
    '''
    Convert reflectivity (eta) to reflectivity factor (Z)

    z, dbz = refl_to_z( eta, wavelength )

    Parameters
    ----------
    eta: vector
        Reflectivity values (units: cm^2/km^3 )
    wavelength: scalar
        Radar wavelength (units: meters; default = 0.1071 )

    Returns
    -------
    z: array
        Vector of Z values (reflectivity factor; units: mm^6/m^3)
    dbz: vector
        Decibels of z (10.^(z/10))

    For details of conversion see refl_to_z documentation

    See Also
    --------
    refl_to_z
    '''

    K_m_squared = 0.93

    log_z = np.log10(eta) + 4 * np.log10(wavelength) - 5 * np.log10(np.pi) - np.log10(K_m_squared)

    '''
    Current units: eta * lambda^4 = 1e-13 m^-1 * 1 m^4 
                                  = 1e-13 m^3 
    Multiply by 10^5 to get units 1e-18
    '''

    log_z = log_z + 5  # Multiply by 10^5

    dbz = 10 * log_z
    z = 10 ** (log_z)

    return z, dbz


# def test_conversions():
#     dbz = np.linspace(-15, 70, 100)
#     z = idb(dbz)
#     print(dbz - db(z))

#     eta, _ = z_to_refl(z)
#     z2, _ = refl_to_z(eta)
#     print(z - z2)

def cart2pol(x, y):
    '''
    Convert from Cartesian coordinates to polar coordinate

    theta, rho = cart2pol( x, y)

    Parameters
    ----------
    x, y: array-like
        Horizontal coordinate and vertical coordinate

    Returns
    -------
    theta, rho: array-like
        Input arrays: angle in radians, distance from origin

    See Also
    --------
    pol2cart
    '''
    theta = np.arctan2(y, x)
    rho = np.hypot(x, y)
    return theta, rho


def pol2cart(theta, rho):
    '''Convert from polar coordinate to Cartesian coordinates

    Parameters
    ----------
    theta, rho: array-like
        Input arrays: angle in radians, distance from origin

    Returns
    -------
    x, y: array-like
        Horizontal coordinate and vertical coordinate

    See Also
    --------
    cart2pol
    '''
    x = rho * np.cos(theta)
    y = rho * np.sin(theta)
    return x, y


def pol2cmp(theta):
    '''Convert from mathematical angle to compass bearing

    Parameters
    ----------
    theta: array-like
        angle in radians counter-clockwise from positive x-axis

    Returns
    -------
    bearing: array-like
        angle in degrees clockwise from north

    See Also
    --------
    cmp2pol
    '''
    bearing = np.rad2deg(np.pi / 2 - theta)
    bearing = np.mod(bearing, 360)
    return bearing


def cmp2pol(bearing):
    '''Convert from compass bearing to mathematical angle

    Parameters
    ----------
    bearing: array-like
        Angle measured in degrees clockwise from north

    Returns
    -------
    theta: array-like
        angle in radians counter-clockwise from positive x-axis

    See Also
    --------
    pol2cmp
    '''
    theta = np.deg2rad(90 - bearing)
    theta = np.mod(theta, 2 * np.pi)
    return theta


def slant2ground(r, theta):
    '''
    Convert from slant range and elevation to ground range and height.

    Parameters
    ----------
    r: array
        Range along radar path in m
    theta: array
        elevation angle in degrees

    Returns
    -------
    s: array
        Range along ground (great circle distance) in m
    h: array
        Height above earth in m

    Uses spherical earth with radius 6371.2 km

    From Doviak and Zrnic 1993 Eqs. (2.28b) and (2.28c)

    See also
    https://bitbucket.org/deeplycloudy/lmatools/src/3ad332f9171e/coordinateSystems.py?at=default

    See Also
    --------
    pyart.core.antenna_to_cartesian
    '''

    earth_radius = 6371200.0  # from NARR GRIB file
    multiplier = 4.0 / 3.0

    r_e = earth_radius * multiplier  # earth effective radius

    theta = np.deg2rad(theta)  # convert to radians

    z = np.sqrt(r ** 2 + r_e ** 2 + (2 * r_e * r * np.sin(theta))) - r_e
    s = r_e * np.arcsin(r * np.cos(theta) / (r_e + z))

    return s, z


def get_unambiguous_range(self, sweep, check_uniform=True):
    """
    Return the unambiguous range in meters for a given sweep.

    Raises a LookupError if the unambiguous range is not available, an
    Exception is raised if the velocities are not uniform in the sweep
    unless check_uniform is set to False.

    Parameters
    ----------
    sweep : int
        Sweep number to retrieve data for, 0 based.
    check_uniform : bool
        True to check to perform a check on the unambiguous range that
        they are uniform in the sweep, False will skip this check and
        return the velocity of the first ray in the sweep.

    Returns
    -------
    unambiguous_range : float
        Scalar containing the unambiguous in m/s for a given sweep.

    """
    s = self.get_slice(sweep)
    try:
        unambiguous_range = self.instrument_parameters['unambiguous_range']['data'][s]
    except:
        raise LookupError('unambiguous range unavailable')
    if check_uniform:
        if np.any(unambiguous_range != unambiguous_range[0]):
            raise Exception('Nyquist velocities are not uniform in sweep')
    return float(unambiguous_range[0])


# Get unique sweeps

def get_tilts(radar):
    tilts = radar.fixed_angle['data']
    unique_tilts = np.unique(tilts)
    return tilts, unique_tilts


def get_sweeps(radar, field):
    tilts, unique_tilts = get_tilts(radar)

    rng = radar.range['data']

    # list of dicts w/ entries
    #  az, rng, data

    n = len(unique_tilts)

    sweeps = [None] * n

    for i, tilt in enumerate(unique_tilts):
        matches = np.nonzero(tilts == tilt)[0]
        nyq_vels = [radar.get_nyquist_vel(i) for i in matches]

        # non-Doppler fields: pick the one with smallest prf
        if field in ['reflectivity',
                     'differential_reflectivity',
                     'cross_correlation_ratio',
                     'differential_phase']:

            j = matches[np.argmin(nyq_vels)]

            # Doppler fields: pick the one with largest prf
        elif field in ['velocity',
                       'spectrum_width']:

            j = matches[np.argmax(nyq_vels)]

        else:
            raise ValueError("Invalid field")

        elev = radar.get_elevation(j)
        az = radar.get_azimuth(j)
        unambiguous_range = get_unambiguous_range(radar, j)  # not a class method
        data = radar.get_field(j, field)

        # Convert to regular numpy array filled with NaNs
        data = np.ma.filled(data, fill_value=np.nan)

        # Sort by azimuth
        I = np.argsort(az)
        az = az[I]
        elev = elev[I]
        data = data[I, :]

        sweeps[i] = {
            'data': data,
            'az': az,
            'rng': rng,
            'elev': elev,
            'fixed_angle': tilt,
            'unambiguous_range': unambiguous_range,
            'sweepnum': j
        }

    return sweeps


def get_volumes(radar, field='reflectivity', coords='antenna'):
    '''
    Get all sample volumes in a vector, along with coordinates

    x1, x2, x3, data = get_volumes(radar, field)

    Parameters
    ----------
    radar: Radar
        The Py-ART radar object representing the volume scan
    field: string
        Which field to get, e.g., 'reflectivity'
    coords: string
        Return coordinate system ('antenna' | 'cartesian' | 'geographic')

    Returns
    -------
    x1, x2, x3: array
        Coordinate arrays for each sample volume in specified coordinate system
    data: array
        Measurements for requested field for each sample volume

    Dimension orders are:
        antenna:    range, azimuth, elevation
        cartesian:  x, y, z
        geographic: lon, lat, z
    '''

    sweeps = get_sweeps(radar, field)

    n = len(sweeps)

    X1 = [None] * n
    X2 = [None] * n
    X3 = [None] * n
    DATA = [None] * n

    for j, sweep in enumerate(sweeps):

        DATA[j] = sweep['data']

        sweepnum = sweep['sweepnum']

        if coords == 'antenna':
            elev = radar.get_elevation(sweepnum)
            az = radar.get_azimuth(sweepnum)

            # Dimension order is (az, range). Keep this order and ask
            # meshgrid to use 'ij' indexing
            AZ, RNG = np.meshgrid(sweep['az'], sweep['rng'], indexing='ij')
            ELEV = np.full_like(DATA[j], sweep['elev'].reshape(-1, 1))

            X1[j], X2[j], X3[j] = RNG, AZ, ELEV
        elif coords == 'cartesian':
            X, Y, Z = radar.get_gate_x_y_z(sweepnum)
            X1[j], X2[j], X3[j] = X, Y, Z
        elif coords == 'geographic':
            LAT, LON, ALT = radar.get_gate_lat_lon_alt(sweepnum)
            X1[j], X2[j], X3[j] = LON, LAT, ALT
        else:
            raise ValueError('Unrecognized coordinate system: %s' % (coords))

        if X1[j].size != DATA[j].size:
            raise ValueError()

    concat = lambda X: np.concatenate([x.ravel() for x in X])

    X1 = concat(X1)
    X2 = concat(X2)
    X3 = concat(X3)
    DATA = concat(DATA)

    return X1, X2, X3, DATA


def radarInterpolant(data, az, rng, method="nearest"):
    m, n = data.shape

    I = np.argsort(az)
    az = az[I]
    data = data[I, :]

    # Replicate first and last radials on opposite ends of array
    # to correctly handle wrapping
    az = np.hstack((az[-1] - 360, az, az[0] + 360))

    data = np.vstack((data[-1, :],
                      data,
                      data[0, :]))

    # Ensure strict monotonicity
    delta = np.hstack((0, np.diff(az)))  # difference between previous and this

    az = az + np.where(delta == 0, 0.001, 0.0)  # add small amount to each azimuth that
    #  is the same as predecessor

    # Create interpolating function
    return RegularGridInterpolator((az, rng), data,
                                   method=method,
                                   bounds_error=False,
                                   fill_value=np.nan)


VALID_FIELDS = ['reflectivity',
                'velocity',
                'spectrum_width',
                'differential_reflectivity',
                'cross_correlation_ratio',
                'differential_phase']


def radar2mat(radar,
              fields=None,
              coords='polar',
              r_min=2125.0,  # default: first range bin of WSR-88D
              r_max=459875.0,  # default: last range bin
              r_res=250,  # default: super-res gate spacing
              az_res=0.5,  # default: super-res azimuth resolution
              dim=600,  # num pixels on a side in Cartesian rendering
              sweeps=None,
              elevs=np.linspace(0.5, 4.5, 5),
              use_ground_range=True,
              interp_method='nearest'):
    '''
    Input parsing and checking
    '''

    # Get available fields
    available_fields = list(radar.fields.keys())

    # Assemble list of fields to render, with error checking
    if fields is None:
        fields = available_fields

    elif isinstance(fields, (list, np.array)):

        fields = np.array(fields)  # convert to numpy array

        valid = np.in1d(fields, VALID_FIELDS)
        available = np.in1d(fields, available_fields)

        if not (np.all(valid)):
            raise ValueError("fields %s are not valid" % (fields[valid != True]))

        if not (np.all(available)):
            warnings.warn("requested fields %s were not available" % (fields[available != True]))

        fields = fields[available]

    else:
        raise ValueError("fields must be None or a list")

    ''' 
    Get indices of desired sweeps (within unique sweeps), save in "sweeps" variable
    '''
    _, available_elevs = get_tilts(radar)

    if sweeps is not None:
        warnings.warn('Both sweeps and elevs are specified. Using sweeps')
    elif elevs is not None:
        # Use interp1d to map requested elevation to nearest available elevation
        # and report the index
        inds = np.arange(len(available_elevs))
        elev2ind = interp1d(available_elevs, inds, kind='nearest', fill_value="extrapolate")
        sweeps = elev2ind(elevs).astype(int)
    else:
        raise ValueError("must specify either sweeps or elevs")

    '''
    Construct coordinate matrices PHI, R for query points
    '''
    if coords == 'polar':
        # Query points
        r = np.arange(r_min, r_max, r_res)
        phi = np.arange(az_res, 360, az_res)
        PHI, R = np.meshgrid(phi, r)

        # Coordinates of three dimensions in output array
        x1 = elevs
        x2 = r
        x3 = phi

    elif coords == 'cartesian':
        x = y = np.linspace(-r_max, r_max, dim)
        [X, Y] = np.meshgrid(x, y)
        [PHI, R] = cart2pol(X, Y)
        PHI = pol2cmp(PHI)  # convert from radians to compass heading

        # Coordinates of three dimensions in output array
        x1 = elevs
        x2 = y
        x3 = x

    else:
        raise ValueError("inavlid coords: %s" % (coords))

    '''
    Build the output 3D arrays
    '''
    data = dict()

    m, n = PHI.shape
    nsweeps = len(sweeps)

    for field in fields:
        data[field] = np.empty((nsweeps, m, n))

        thesweeps = get_sweeps(radar, field)  # all sweeps

        for i in range(nsweeps):

            # get ith selected sweep
            sweep_num = sweeps[i]
            sweep = thesweeps[sweep_num]

            az = sweep['az']
            rng = sweep['rng']

            if use_ground_range:
                rng, _ = slant2ground(rng, sweep['fixed_angle'])

            F = radarInterpolant(sweep['data'], az, rng, method=interp_method)

            data[field][i, :, :] = F((PHI, R))

    return data, x1, x2, x3
