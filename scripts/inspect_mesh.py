import sys, pyvista as pv, pathlib
p = pathlib.Path(sys.argv[1])
m = pv.read(p.as_posix())
xmin,xmax,ymin,ymax,zmin,zmax = m.bounds
cx,cy,cz = (xmin+xmax)/2, (ymin+ymax)/2, (zmin+zmax)/2
print(f"file: {p}")
print(f"bounds: x[{xmin:.3f},{xmax:.3f}]  y[{ymin:.3f},{ymax:.3f}]  z[{zmin:.3f},{zmax:.3f}]")
print(f"center: ({cx:.3f}, {cy:.3f}, {cz:.3f})")
print(f"extent (Lx, Ly, Lz): ({xmax-xmin:.3f}, {ymax-ymin:.3f}, {zmax-zmin:.3f})")
