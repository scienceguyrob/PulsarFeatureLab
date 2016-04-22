"""

**************************************************************************
| CoordinateTransform.py                                                           |
**************************************************************************
| Description:                                                           |
|                                                                        |
| Contains support functions for the codebase. This code runs on python  |
| 2.4 or later.                                                          |
**************************************************************************
| Author: Rob Lyon                                                       |
| Email : robert.lyon@postgrad.manchester.ac.uk                          |
| web   : www.scienceguyrob.com                                          |
**************************************************************************


"""

# Python 2.4 imports.

from astropy.coordinates import ICRS, Galactic
from astropy.coordinates import SkyCoord
import astropy.units as u

"""
Provides functions which transform astronomical coordinates using AstroPy.

"""

# ******************************************************************************************
#
# Functions.
#
# ******************************************************************************************

def transformEquatorialFromDegrees(R,D):
    """
    Transforms RA and DEC in degrees, into HH:MM:SS and
    DD:MM:SS FORMAT.

    Parameters:
    RA     -    the right ascension.
    DEC    -    the declination.

    Returns:
    The RA and DEC.
    """

    RA = float(R)
    DEC = float(D)
    sc = SkyCoord(ra=RA*u.degree, dec=DEC*u.degree, frame='icrs')
    #print sc
    coord = str(sc.to_string('hmsdms'))
    #print coord
    coordinate_str = coord.split(" ")
    RA = coordinate_str[0].replace("h",":").replace("m",":").replace("s","")
    DEC = coordinate_str[1].replace("h",":").replace("m",":").replace("s","").replace("d",":")

    return [RA,DEC]

# ******************************************************************************************
