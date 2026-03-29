import os
os.environ["CUDA_VISIBLE_DEVICES"] = "-1" 

import matplotlib
matplotlib.use('TkAgg') 
import matplotlib.pyplot as plt
import queue
import multiprocessing as mp
import numpy as np
import time

from sb3_contrib import MaskablePPO
from sb3_contrib.common.maskable.utils import get_action_masks
from game_env import RobotMatchEnv
from robot_visualizer import RobotVisualizer

# --- sub-process: for rendering ---
def render_worker(data_queue, config):
    viz = RobotVisualizer(config)
    viz.init_plot()
    
    while True:
        try:
            data = data_queue.get() # get latest game data from queue
            if data == "STOP": break
            viz.update(data)
        except Exception:
            break

# --- main process: run PPO model ---
def main():
    model_path = "../models/robot_strategy_y_v4-3_800K.zip"
    model = MaskablePPO.load(model_path, device="cpu")
    
    env = RobotMatchEnv(render_mode=None, my_color="yellow") # init model env
    # env.load_enemy_model("../models/robot_strategy_b_v4_800K.zip")
    obs, _ = env.reset()

    # init config data of game
    config = {
        'pantry_rects_min': env.pantry_rects_min,
        'collet_rects_min': env.collet_rects_min,
        'collect_sizes': env.collect_sizes,
        'robot_radius': env.robot_radius,
        'max_collect_capacity': env.max_collect_capacity,
        'max_pantry_capacity': env.max_pantry_capacity,
        'max_dir_capacity': env.max_dir_capacity,
        'pantry_size': env.pantry_size
    }
    # communication queue for passing latest game data to render process
    # maxsize=1: ensure render process get latest one frame of data
    data_queue = mp.Queue(maxsize=1)
    # start render sub-process
    render_proc = mp.Process(target=render_worker, args=(data_queue, config))
    render_proc.daemon = True # close sub-process when main process stop
    render_proc.start()

    print("Model loop running at full speed...")
    # --- set demo fps ---
    target_fps = 30
    frame_duration = 1.0 / target_fps
    try:
        # main loop for game
        for step in range(3000):
            loop_start = time.time()  # record start time of game
            # --- model predict ---
            action_masks = get_action_masks(env)
            action, _ = model.predict(obs, action_masks=action_masks, deterministic=True)
            obs, reward, terminated, truncated, info = env.step(action)
            # update latest game data
            render_data = {
                'my_stage': env.my_robot.stage,
                'my_act': action,
                'reward': reward,
                'en_stage': env.enemy_robot.stage,
                'en_act': env.enemy_robot.action,
                'collect_counts': env.collect_counts.copy(),
                'pantry_y': env.pantry_yellow_counts.copy(),
                'pantry_b': env.pantry_blue_counts.copy(),
                'my_pos': env.my_robot.pos.copy(),
                'en_pos': env.enemy_robot.pos.copy(),
                'my_held': env.my_robot.held_count.copy(),
                'en_held': env.enemy_robot.held_count.copy()
            }
            # --- push game data to queue ---
            try:
                data_queue.put_nowait(render_data)
            except queue.Full: # last frame of data didn't take by render (render is busy)
                pass           # continue

            print(f"Step {step}: {info}")
            if terminated or truncated:
                print(f"round stop, steps: {step}")
                break

            # --- ensure demo run with 30 Hz ---
            elapsed = time.time() - loop_start
            sleep_time = frame_duration - elapsed
            if sleep_time > 0:
                time.sleep(sleep_time)

        data_queue.put("STOP")
        render_proc.join()
                
    except KeyboardInterrupt:
        print("Interrupted")
    finally:
        render_proc.terminate()
        print("Test complete")

if __name__ == "__main__":
    main()