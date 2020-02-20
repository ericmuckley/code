import matplotlib._cntr as cntr
import numpy as np
import numpy.ma as ma

def get_contours(x, y, z, level=0):
    x, y = np.meshgrid(x, y)
    z = np.sin(x) + np.cos(y)

    c = cntr.Cntr(x, y, z)
    nlist = c.trace(level, level, 0)
    segs = nlist[:len(nlist)//2]
    print(segs)



# Make your choice of filled contours or contour lines here.
wantFilledContours = True


# Test data.
x = np.arange(0, 10, 1)
y = np.arange(0, 10, 1)
x, y = np.meshgrid(x, y)
z = np.sin(x) + np.cos(y)

z = ma.asarray(z, dtype=np.float64)  # Import if want filled contours.

if wantFilledContours:
    lower_level = 0.5
    upper_level = 0.8
    c = cntr.Cntr(x, y, z.filled())
    nlist = c.trace(lower_level, upper_level, 0)
    nseg = len(nlist)//2
    segs = nlist[:nseg]
    kinds = nlist[nseg:]
    print(segs)    # x,y coords of contour points.
    print(kinds)   # kind codes: 1 = LINETO, 2 = MOVETO, etc.
                  # See lib/matplotlib/path.py for details.
else:
    # Non-filled contours (lines only).
    level = 0.5
    c = cntr.Cntr(x, y, z)
    nlist = c.trace(level, level, 0)
    segs = nlist[:len(nlist)//2]
    print(segs)    # x,y coords of contour points.


