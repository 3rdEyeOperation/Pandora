#!/usr/bin/env python3
"""
triangulator.py

Author: 3rd Eye Operation
Copyright (c) 2025 3rd Eye Operation. All rights reserved.

DISCLAIMER / COPYRIGHT / USAGE NOTICE:
- This code is provided for EDUCATIONAL / RESEARCH purposes only.
- Do NOT use this script to carry out unlawful surveillance, tracking of individuals,
  or any activity that violates privacy or local laws.
- The authors and distributors (3rd Eye Operation) accept NO LIABILITY for misuse.
- By using this script you agree to comply with applicable laws and ethical guidelines.

Purpose:
- Given GPS lat/lon and azimuth degrees from (at least) 2 or 3 stations, estimate
  the intersection point using a local equirectangular projection + least-squares.
- Intended for classroom/lab simulation and testing with synthetic data ONLY.

CLI Usage:
  - Run with built-in example:
      python3 triangulator.py

  - Provide three station triples (lat lon az) on the command line:
      python3 triangulator.py --station 13.736717 100.523186 45 \
                             --station 13.743000 100.534000 135 \
                             --station 13.730000 100.540000 -90

  - Provide a CSV file (header: lat,lon,az) with --csv:
      python3 triangulator.py --csv stations.csv

  - For help:
      python3 triangulator.py --help

Notes:
- Azimuth expected in degrees, clockwise from true north (0 = north, 90 = east).
- This implementation uses a simple local projection (equirectangular) and is
  appropriate for short distances (tens of kilometers). For high-precision /
  long-range work, use geodesic/ellipsoidal methods (GeographicLib/pyproj).

"""
import sys
import math
import argparse
import csv
import numpy as np

R_EARTH = 6371000.0  # meters

def deg2rad(d): return d * math.pi / 180.0
def rad2deg(r): return r * 180.0 / math.pi

def project_equirectangular(lat_deg, lon_deg, lat0_deg, lon0_deg):
    lat = deg2rad(lat_deg)
    lon = deg2rad(lon_deg)
    lat0 = deg2rad(lat0_deg)
    lon0 = deg2rad(lon0_deg)
    x = R_EARTH * (lon - lon0) * math.cos(lat0)
    y = R_EARTH * (lat - lat0)
    return x, y

def inverse_project_equirectangular(x, y, lat0_deg, lon0_deg):
    lat0 = deg2rad(lat0_deg)
    lon0 = deg2rad(lon0_deg)
    lat = y / R_EARTH + lat0
    lon = x / (R_EARTH * math.cos(lat0)) + lon0
    return rad2deg(lat), rad2deg(lon)

def triangulate_from_bearings(stations):
    """
    stations: list of tuples (lat_deg, lon_deg, azimuth_deg)
    azimuth_deg is clockwise from true north (0..360 or negative allowed)
    returns: (lat_deg, lon_deg, info_str)
    """
    if len(stations) < 2:
        return None, "Need at least two stations."

    # center for local projection: average lat/lon
    lat0 = sum(s[0] for s in stations) / len(stations)
    lon0 = sum(s[1] for s in stations) / len(stations)

    pts = []
    thetas = []
    for lat, lon, az in stations:
        x, y = project_equirectangular(lat, lon, lat0, lon0)
        # convert azimuth (clockwise from north) to mathematical angle theta
        # theta measured from +x (east) counter-clockwise:
        # az=0 (north) -> theta = 90 deg, az=90 (east) -> theta=0 deg
        theta_deg = 90.0 - az
        theta = deg2rad(theta_deg)
        pts.append((x, y))
        thetas.append(theta)

    # Build linear system: sin(theta)*x - cos(theta)*y = sin(theta)*x_i - cos(theta)*y_i
    A = []
    b = []
    for (x_i, y_i), theta in zip(pts, thetas):
        s = math.sin(theta)
        c = math.cos(theta)
        A.append([s, -c])
        b.append(s * x_i - c * y_i)

    A = np.array(A)
    b = np.array(b)

    try:
        sol, residues, rank, svals = np.linalg.lstsq(A, b, rcond=None)
    except Exception as e:
        return None, f"Linear solve failed: {e}"

    x_est, y_est = float(sol[0]), float(sol[1])
    lat_est, lon_est = inverse_project_equirectangular(x_est, y_est, lat0, lon0)
    info = f"ok (residuals: {residues.tolist() if hasattr(residues,'tolist') else residues}, rank: {rank})"
    return (lat_est, lon_est), info

def read_csv_stations(path):
    stations = []
    with open(path, newline='') as fh:
        rdr = csv.DictReader(fh)
        for row in rdr:
            try:
                lat = float(row.get('lat') or row.get('latitude') or row.get('Lat'))
                lon = float(row.get('lon') or row.get('longitude') or row.get('Lon'))
                az  = float(row.get('az') or row.get('azimuth') or row.get('Az'))
            except Exception:
                raise ValueError("CSV must contain numeric columns lat, lon, az (or latitude,longitude,azimuth).")
            stations.append((lat, lon, az))
    return stations

def main():
    parser = argparse.ArgumentParser(description="Triangulate intersection from station lat,lon,azimuth. Educational use only.")
    parser.add_argument('--station', '-s', action='append', nargs=3, metavar=('LAT','LON','AZ'),
                        help="Add a station triple: LAT LON AZ (azimuth degrees clockwise from true north). Can repeat 2..N times.")
    parser.add_argument('--csv', '-c', metavar='FILE', help="CSV file with header lat,lon,az (or latitude,longitude,azimuth)")
    parser.add_argument('--example', action='store_true', help="Run built-in example dataset")
    parser.add_argument('--version', action='store_true', help="Show version / author")
    args = parser.parse_args()

    if args.version:
        print("triangulator.py — Author: 3rd Eye Operation — Educational example (2025)")
        sys.exit(0)

    stations = []
    if args.csv:
        try:
            stations = read_csv_stations(args.csv)
        except Exception as e:
            print("Failed to read CSV:", e)
            sys.exit(2)

    if args.station:
        try:
            for triple in args.station:
                lat = float(triple[0]); lon = float(triple[1]); az = float(triple[2])
                stations.append((lat, lon, az))
        except Exception as e:
            print("Invalid station triple:", e)
            sys.exit(2)

    if args.example or (not stations):
        # built-in example (synthetic)
        stations = [
            (13.736717, 100.523186, 45.0),
            (13.743000, 100.534000, 135.0),
            (13.730000, 100.540000, -90.0)
        ]
        print("Using built-in example stations (synthetic).")

    print("Stations (lat, lon, az):")
    for s in stations:
        print("  ", s)

    result, info = triangulate_from_bearings(stations)
    if result:
        lat_out, lon_out = result
        print(f"\nEstimated position: {lat_out:.6f}, {lon_out:.6f}")
    else:
        print("\nEstimation failed.")
    print("Info:", info)

if __name__ == "__main__":
    main()
