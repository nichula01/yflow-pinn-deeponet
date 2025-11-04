#!/usr/bin/env python3
"""
Robust Y-junction generator with verbose logging, optional preview, and auto-clean on errors.

Usage (from repo root):
  python scripts\make_y.py --angle 60 --rp 2.5 --rd 2.0 --len_parent 30 --len_daughter 20 --name y60_rp2p5_rd2p0 --preview
  python scripts\make_y.py --angle 60 --no-fillet --debug
"""

import argparse
import pathlib
import numpy as np
import pyvista as pv
import sys
import traceback


def log(msg: str):
    print(f"[make_y] {msg}", flush=True)


def make_y(angle_deg=60.0, rp=2.5, rd=2.0, len_parent=30.0, len_daughter=20.0, fillet=True, debug=False):
    log(f"params: angle={angle_deg}° rp={rp} rd={rd} Lp={len_parent} Ld={len_daughter} fillet={fillet} debug={debug}")
    angle = np.deg2rad(angle_deg)

    # Parent along +Z, centered so it spans z in [0, len_parent]
    parent = pv.Cylinder(center=(0, 0, len_parent / 2.0), direction=(0, 0, 1), radius=rp, height=len_parent)
    # Daughters directions
    dir1 = ( np.sin(angle / 2.0), 0.0, np.cos(angle / 2.0) )
    dir2 = (-np.sin(angle / 2.0), 0.0, np.cos(angle / 2.0) )
    # Split near top of parent
    split_z = len_parent * 0.8
    c1 = ( rp * np.sin(angle / 2.0), 0.0, split_z )
    c2 = (-rp * np.sin(angle / 2.0), 0.0, split_z )

    b1 = pv.Cylinder(center=c1, direction=dir1, radius=rd, height=len_daughter)
    b2 = pv.Cylinder(center=c2, direction=dir2, radius=rd, height=len_daughter)

    if debug:
        dbg = pathlib.Path("geom") / "_debug"
        dbg.mkdir(parents=True, exist_ok=True)
        parent.save((dbg / "parent.vtk").as_posix())
        b1.save((dbg / "b1.vtk").as_posix())
        b2.save((dbg / "b2.vtk").as_posix())
        log(f"debug: saved raw cylinders under {dbg}")

    # Triangulate & clean before booleans
    log("triangulate+clean inputs…")
    parent = parent.triangulate().clean()
    b1 = b1.triangulate().clean()
    b2 = b2.triangulate().clean()

    # Boolean unions
    log("boolean union #1 (parent ∪ b1)…")
    solid = parent.boolean_union(b1, tolerance=1e-6).triangulate().clean()
    log("boolean union #2 ((…) ∪ b2)…")
    solid = solid.boolean_union(b2, tolerance=1e-6).triangulate().clean()

    if fillet:
        log("smoothing for visual quality…")
        solid = solid.smooth(n_iter=80, relaxation_factor=0.2)

    log("computing normals…")
    solid = solid.compute_normals(auto_orient_normals=True, inplace=False)

    if solid.n_points == 0 or solid.n_cells == 0:
        raise RuntimeError("empty mesh after boolean ops")

    return solid


def save_all(mesh: pv.PolyData, outstem: pathlib.Path, preview=False):
    outstem.parent.mkdir(parents=True, exist_ok=True)
    vtk_path = outstem.with_suffix(".vtk")
    stl_path = outstem.with_suffix(".stl")
    png_path = outstem.with_suffix(".png")

    # remove existing
    for p in [vtk_path, stl_path, png_path]:
        if p.exists():
            p.unlink()

    log(f"saving {vtk_path.name} and {stl_path.name}…")
    mesh.save(vtk_path.as_posix())
    mesh.save(stl_path.as_posix())

    if preview:
        log("rendering preview (off-screen)…")
        pl = pv.Plotter(off_screen=True)
        pl.add_mesh(mesh, smooth_shading=True, show_edges=False, color="lightgray")
        pl.show_bounds(grid='front', location='outer')
        pl.set_background("white")
        pl.camera_position = "iso"
        pl.show(screenshot=str(png_path))
        log(f"preview saved: {png_path.name}")

    log("done.")


def main():
    ap = argparse.ArgumentParser(description="Generate a robust 3D Y-junction airway.")
    ap.add_argument("--angle", type=float, default=60.0)
    ap.add_argument("--rp", type=float, default=2.5)
    ap.add_argument("--rd", type=float, default=2.0)
    ap.add_argument("--len_parent", type=float, default=30.0)
    ap.add_argument("--len_daughter", type=float, default=20.0)
    ap.add_argument("--name", type=str, default=None)
    ap.add_argument("--no-fillet", action="store_true")
    ap.add_argument("--preview", action="store_true")
    ap.add_argument("--debug", action="store_true", help="save raw cylinders to geom/_debug")
    args = ap.parse_args()

    outname = args.name or f"y{int(round(args.angle))}_rp{str(args.rp).replace('.','p')}_rd{str(args.rd).replace('.','p')}"
    outstem = pathlib.Path("geom") / outname

    try:
        mesh = make_y(
            angle_deg=args.angle,
            rp=args.rp,
            rd=args.rd,
            len_parent=args.len_parent,
            len_daughter=args.len_daughter,
            fillet=not args.no_fillet,
            debug=args.debug,
        )
        save_all(mesh, outstem, preview=args.preview)
    except Exception as e:
        log(f"ERROR: {e}")
        traceback.print_exc()
        # auto-clean partial outputs
        for ext in (".vtk", ".stl", ".png"):
            p = outstem.with_suffix(ext)
            if p.exists():
                try:
                    p.unlink()
                    log(f"cleaned partial: {p.name}")
                except Exception:
                    pass
        sys.exit(1)


if __name__ == "__main__":
    main()
