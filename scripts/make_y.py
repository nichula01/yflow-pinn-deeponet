#!/usr/bin/env python3
"""
Generate a clean 3D Y-junction airway and save STL/VTK/PNG.

Default method = SDF + marching cubes (robust, watertight).
Alternative    = voxel union (--method voxel).

Usage:
  python scripts\make_y.py --angle 60 --rp 2.5 --rd 2.0 --len_parent 30 --len_daughter 20 --name y60 --preview
"""

import argparse, pathlib, sys, traceback
import numpy as np
import pyvista as pv
from skimage.measure import marching_cubes

def log(m): print(f"[make_y] {m}", flush=True)

# ---------- math utils (SDF) ----------
def normalize(v): 
    v = np.asarray(v, float); n = np.linalg.norm(v); 
    return v / (n + 1e-12)

def sdf_capped_cylinder(P, A, B, r):
    """
    Signed distance to a finite cylinder from A->B with radius r.
    P: (...,3), A,B: (3,), r: float
    """
    pa = P - A
    ba = B - A
    h = np.clip((pa * ba).sum(-1) / (ba * ba).sum(), 0.0, 1.0)[..., None]
    d = np.linalg.norm(pa - h*ba, axis=-1) - r
    # outside ends: distance to end-caps
    q = np.maximum(h[...,0]-1.0, 0.0) + np.maximum(-h[...,0], 0.0)
    return np.where(q>0, np.sqrt(d**2 + (q*np.linalg.norm(ba))**2), d)

def smooth_union(d1, d2, k=0.5):
    """Smooth min (union) via log-sum-exp approximation."""
    # smaller k = sharper union; larger k = smoother blend
    m = np.minimum(d1, d2)
    return -k * np.log(np.exp(-(d1 - m)/k) + np.exp(-(d2 - m)/k)) + m

def make_y_sdf(angle_deg=60, rp=2.5, rd=2.0, Lp=30.0, Ld=20.0, grid=0.25, blend=0.6):
    """
    Build Y as SDF and extract surface with marching cubes.
    grid : voxel size of the SDF grid (smaller -> smoother)
    blend: smoothing at junction (0.4..0.8 typical)
    """
    log(f"SDF grid={grid}, blend={blend}")
    ang = np.deg2rad(angle_deg)

    # Axis vectors
    e_z = np.array([0,0,1.0])
    d1  = normalize([ np.sin(ang/2), 0.0, np.cos(ang/2) ])
    d2  = normalize([-np.sin(ang/2), 0.0, np.cos(ang/2) ])

    # Segment endpoints (centerlines)
    A  = np.array([0,0,0.0])
    B  = np.array([0,0,Lp])                  # parent along +Z
    split_z = Lp*0.8
    S  = np.array([0,0,split_z])

    Lvec1 = d1 * Ld
    Lvec2 = d2 * Ld
    B1 = S + Lvec1
    B2 = S + Lvec2
    # small lateral offsets to avoid coplanar degeneracy
    off = rp*0.15
    A1  = S + np.array([ off*np.sign(d1[0]), 0, 0])
    A2  = S + np.array([-off*np.sign(d2[0]), 0, 0])

    # Compute bounds
    pts = np.vstack([A,B,A1,B1,A2,B2])
    pad = max(rp, rd) + 4.0
    lo  = (pts.min(0) - pad).astype(float)
    hi  = (pts.max(0) + pad).astype(float)

    xs = np.arange(lo[0], hi[0]+grid, grid)
    ys = np.arange(lo[1], hi[1]+grid, grid)
    zs = np.arange(lo[2], hi[2]+grid, grid)
    X,Y,Z = np.meshgrid(xs, ys, zs, indexing="ij")
    P = np.stack([X,Y,Z], axis=-1)

    # SDFs
    Dp  = sdf_capped_cylinder(P, A, B, rp)
    D1  = sdf_capped_cylinder(P, A1, B1, rd)
    D2  = sdf_capped_cylinder(P, A2, B2, rd)

    # smooth unions
    D = smooth_union(Dp, D1, k=blend)
    D = smooth_union(D,  D2, k=blend)

    # Extract iso-surface at 0
    log("marching cubes…")
    verts, faces, _, _ = marching_cubes(D.transpose(2,1,0), level=0.0, spacing=(grid,grid,grid))
    # marching_cubes returns z,y,x order; we compensated via transpose

    # shift to world coordinates
    verts[:,0] += lo[0]
    verts[:,1] += lo[1]
    verts[:,2] += lo[2]

    # build PolyData
    faces_flat = np.hstack([np.full((faces.shape[0],1),3), faces]).astype(np.int32).ravel()
    mesh = pv.PolyData(verts, faces_flat).triangulate().clean()
    mesh = mesh.smooth(n_iter=50, relaxation_factor=0.2)
    mesh = mesh.compute_normals(auto_orient_normals=True, inplace=False)
    return mesh

# ---------- voxel fallback (optional) ----------
def make_y_voxel(angle_deg=60, rp=2.5, rd=2.0, Lp=30.0, Ld=20.0, density=None, fillet=True):
    ang = np.deg2rad(angle_deg)
    parent = pv.Cylinder(center=(0,0,Lp/2), direction=(0,0,1), radius=rp, height=Lp)
    d1 = ( np.sin(ang/2), 0, np.cos(ang/2) )
    d2 = (-np.sin(ang/2), 0, np.cos(ang/2) )
    S  = Lp*0.8
    c1 = ( rp*np.sin(ang/2), 0, S )
    c2 = (-rp*np.sin(ang/2), 0, S )
    b1 = pv.Cylinder(center=c1, direction=d1, radius=rd, height=Ld)
    b2 = pv.Cylinder(center=c2, direction=d2, radius=rd, height=Ld)
    poly = parent.triangulate().clean().merge(b1.triangulate().clean()).merge(b2.triangulate().clean()).clean()
    if density is None or density<=0: density = max(rp/3.0, 0.3)
    grid = poly.voxelize(density)
    mesh = grid.extract_surface().triangulate().clean()
    if fillet: mesh = mesh.smooth(n_iter=60, relaxation_factor=0.2)
    return mesh.compute_normals(auto_orient_normals=True, inplace=False)

def save_all(mesh: pv.PolyData, outstem: pathlib.Path, preview=False):
    outstem.parent.mkdir(parents=True, exist_ok=True)
    vtk = outstem.with_suffix(".vtk")
    stl = outstem.with_suffix(".stl")
    png = outstem.with_suffix(".png")
    for p in (vtk,stl,png):
        if p.exists(): p.unlink()
    mesh.save(vtk.as_posix()); mesh.save(stl.as_posix())
    if preview:
        pl = pv.Plotter(off_screen=True)
        pl.add_mesh(mesh, smooth_shading=True, show_edges=False, color="lightgray")
        pl.show_bounds(grid='front', location='outer')
        pl.set_background("white"); pl.camera_position="iso"
        pl.show(screenshot=str(png))
        log(f"preview saved: {png.name}")
    log(f"saved: {vtk.name}, {stl.name}")

def main():
    ap = argparse.ArgumentParser(description="Generate a 3D Y-junction airway.")
    ap.add_argument("--angle", type=float, default=60.0)
    ap.add_argument("--rp", type=float, default=2.5)
    ap.add_argument("--rd", type=float, default=2.0)
    ap.add_argument("--len_parent", type=float, default=30.0)
    ap.add_argument("--len_daughter", type=float, default=20.0)
    ap.add_argument("--name", type=str, default=None)
    ap.add_argument("--preview", action="store_true")
    ap.add_argument("--method", choices=["sdf","voxel"], default="sdf")
    ap.add_argument("--grid", type=float, default=0.25, help="SDF grid spacing (sdf method)")
    ap.add_argument("--blend", type=float, default=0.6, help="Junction smoothing (sdf method)")
    ap.add_argument("--density", type=float, default=None, help="voxel size (voxel method)")
    ap.add_argument("--no-fillet", action="store_true", help="disable smoothing (voxel method only)")
    args = ap.parse_args()

    outname = args.name or f"y{int(round(args.angle))}_rp{str(args.rp).replace('.','p')}_rd{str(args.rd).replace('.','p')}"
    outstem = pathlib.Path("geom")/outname
    try:
        if args.method == "sdf":
            mesh = make_y_sdf(args.angle, args.rp, args.rd, args.len_parent, args.len_daughter, grid=args.grid, blend=args.blend)
        else:
            mesh = make_y_voxel(args.angle, args.rp, args.rd, args.len_parent, args.len_daughter,
                                density=args.density, fillet=not args.no-fillet)
        save_all(mesh, outstem, preview=args.preview)
    except Exception as e:
        log(f"ERROR: {e}")
        traceback.print_exc(); sys.exit(1)

if __name__ == "__main__":
    main()
