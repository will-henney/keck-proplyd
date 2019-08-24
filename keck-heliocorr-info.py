import sys
import os
import glob
from astropy.io import fits
import astropy.coordinates as coord
from astropy.wcs import WCS
from astropy.time import Time
import astropy.units as u
sys.path.append('/Users/will/Dropbox/OrionWest')
from helio_utils import helio_topo_from_header

outtab = [['File', 'Date', 'JD', 'ST', 'RA', 'Dec', 'Helio', 'Helio2'], None]
speclist = glob.glob('Keck?/[jp]*[5-8][0-9].fits')
spm = coord.EarthLocation.of_site("Keck Observatory")
for fn in sorted(speclist):
    hdr = fits.open(fn)[0].header
    w = WCS(hdr)
    wav, c = w.pixel_to_world(1, 1, 1)
    time = Time(w.wcs.dateobs)
    heliocorr = c.radial_velocity_correction(
        'barycentric', obstime=time, location=spm)
    heliocorr2 = helio_topo_from_header(hdr)
    id_, _ = os.path.splitext(os.path.basename(fn))
    outtab.append([id_, time.iso.split()[0],
                   time.mjd,
                   hdr.get('ST'), hdr.get('RA'), hdr.get('DEC'),
                   '{:.2f}'.format(heliocorr.to(u.km/u.s).value),
                   '{:.2f}'.format(heliocorr2),
    ])
