#!/usr/bin/env python3
"""Perception node.
5 lidars + 2 LD2450 radars + Mega ultrasonics
  -> fusion -> PointNet filter -> 2D SLAM -> costmap -> A* -> lidar map PNG

Usage:
  python3 node.py                    # mapping only
  python3 node.py --goal 2.0 1.0     # mapping + planning to a goal (metres, map frame)
"""
import argparse
import os
import time

import numpy as np
import yaml

from ML.sensor.lidar import LidarArray
from ML.sensor.radar import LD2450
from ML.sensor.mega import MegaUART
from ML.fusion import fuse
from perception.pointnet_infer import PointNetClassifier
from mapping.slam_2d import OccupancyGrid, Slam2D, transform
from mapping.costmap import build_costmap
from planning.astar import astar
from viz.lidar_map import save_map_png


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="config/perception.yaml")
    ap.add_argument("--goal", type=float, nargs=2, metavar=("X", "Y"))
    ap.add_argument("--png-every", type=int, default=50,
                    help="cycles between map PNG saves")
    args = ap.parse_args()

    with open(args.config) as f:
        cfg = yaml.safe_load(f)

    lidars = LidarArray(cfg["lidars"])
    radars = [LD2450(c) for c in cfg["radars"]]
    mega = MegaUART(cfg["mega"])
    pn_cfg = cfg["pointnet"]
    pn = PointNetClassifier(pn_cfg["model_path"], pn_cfg["num_points"], pn_cfg["dyn_thr"])

    g = cfg["grid"]
    grid = OccupancyGrid(g["resolution_m"], g["size_m"])
    slam = Slam2D(grid)
    pl = cfg["planner"]
    goal = tuple(args.goal) if args.goal else None

    os.makedirs("logs", exist_ok=True)
    lidars.start()
    for r in radars:
        r.start()
    mega.start()

    period = 1.0 / cfg["loop_hz"]
    path, cost = None, None
    dyn_w = np.empty((0, 2))
    step = 0

    print("perception node running. Ctrl+C to stop.")
    try:
        while True:
            t0 = time.monotonic()

            lid = lidars.merged()
            rad = np.vstack([r.targets() for r in radars])
            son = mega.points()

            xy, moving = fuse(lid, rad, son)
            labels = pn.labels(xy) if len(xy) else np.zeros(0, dtype=int)
            labels[moving] = 1                       # radar-confirmed motion
            static = xy[labels == 0]
            dyn_local = xy[labels == 1]

            pose = slam.update(static)
            dyn_w = transform(dyn_local, pose) if len(dyn_local) else np.empty((0, 2))
            cost = build_costmap(grid, dyn_w, pl["robot_radius_m"], pl["inflation_m"])

            if goal and (path is None or step % 10 == 0):
                path = astar(cost, grid, pose[:2], goal)

            near = mega.min_range()
            stop = near is not None and near < pl["safety_stop_m"]

            dt = time.monotonic() - t0
            if step % 10 == 0:
                print(f"pose=({pose[0]:+.2f}, {pose[1]:+.2f}, "
                      f"{np.degrees(pose[2]):+.0f}deg) pts={len(xy)} "
                      f"dyn={len(dyn_local)} loop={dt * 1000:.0f}ms "
                      f"path={'yes' if path else 'no'} STOP={stop}")
            if step % args.png_every == 0:
                save_map_png(grid, cost, slam.trajectory, path, goal, dyn_w,
                             "logs/lidar_map.png")

            step += 1
            time.sleep(max(0.0, period - dt))

    except KeyboardInterrupt:
        print("\nstopping...")
    finally:
        save_map_png(grid, cost, slam.trajectory, path, goal, dyn_w,
                     "logs/lidar_map.png")
        np.save("logs/logodds.npy", grid.logodds)
        lidars.stop()
        for r in radars:
            r.stop()
        mega.stop()
        print("saved logs/lidar_map.png and logs/logodds.npy")


if __name__ == "__main__":
    main()