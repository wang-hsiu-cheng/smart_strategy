import time
import queue
import multiprocessing as mp
from sb3_contrib import MaskablePPO

# 引入環境與配置檔 (位於同層 train/src/)
import config
from game_env import RobotMatchEnv
from robot_visualizer import RobotVisualizer  # 請依你的視覺化模組檔名調整引用


def render_worker(data_queue, viz_config):
    """
    獨立進程：專門處理 Matplotlib 繪圖渲染，完全不干擾模型推論頻率
    """
    viz = RobotVisualizer(viz_config)
    viz.init_plot()
    
    while True:
        try:
            data = data_queue.get()
            if data == "STOP":
                break
            viz.update(data)
        except Exception:
            break


def main():
    # 1. 從 config (.env) 讀取模型與環境設定
    print(f"[*] Loading model from: {config.INF_MODEL_PATH}")
    print(f"[*] Running on device: {config.INF_DEVICE} | Robot color: {config.INF_MY_COLOR}")
    
    model = MaskablePPO.load(config.INF_MODEL_PATH, device=config.INF_DEVICE)
    
    # 2. 初始化測試環境
    env = RobotMatchEnv(
        render_mode=config.INF_RENDER_MODE, 
        my_color=config.INF_MY_COLOR
    )
    
    # 若 .env 有設定敵方對手模型，則動態載入
    if config.INF_ENEMY_MODEL_PATH:
        print(f"[*] Loading enemy opponent: {config.INF_ENEMY_MODEL_PATH}")
        env.load_enemy_model(config.INF_ENEMY_MODEL_PATH)
    else:
        print("[*] No enemy model specified, running without dynamic opponent.")

    obs, _ = env.reset()

    # 3. 初始化視覺化常數配置 (更名為 viz_config 避免覆蓋 config 模組)
    viz_config = {
        'pantry_rects_min': env.pantry_rects_min,
        'collet_rects_min': env.collet_rects_min,
        'collect_sizes': env.collect_sizes,
        'robot_radius': env.robot_radius,
        'max_collect_capacity': env.max_collect_capacity,
        'max_pantry_capacity': env.max_pantry_capacity,
        'max_dir_capacity': env.max_dir_capacity,
        'pantry_size': env.pantry_size
    }

    # 4. 建立通訊佇列與繪圖子進程 (maxsize=1 確保永遠取最新幀)
    data_queue = mp.Queue(maxsize=1)
    render_proc = mp.Process(target=render_worker, args=(data_queue, viz_config))
    render_proc.daemon = True
    render_proc.start()

    print("[*] Model loop running at full speed (Target: 30 FPS)...")
    target_fps = 30
    frame_duration = 1.0 / target_fps

    try:
        for step in range(3000):
            loop_start = time.time()

            # 5. 獲取 Action Mask 並預測動作
            # 優先採用 env.action_masks()，若無則降級呼叫自定義函數
            if hasattr(env, "action_masks"):
                masks = env.action_masks()
            else:
                masks = env._get_action_masks(env.my_robot)
                
            action, _ = model.predict(obs, action_masks=masks, deterministic=True)
            obs, reward, terminated, truncated, info = env.step(action)

            # 6. 打包當幀資料送至繪圖進程 (使用 .copy() 防止記憶體引用污染)
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

            try:
                data_queue.put_nowait(render_data)
            except queue.Full:
                pass  # 若繪圖進程忙碌，直接捨棄舊幀以保證推論不掉速

            if step % 30 == 0:
                print(f"Step {step} | Reward: {reward:.3f} | Info: {info}")

            if terminated or truncated:
                print(f"[+] Episode finished at step {step}")
                break

            # 7. 控制幀率為 30 Hz
            elapsed = time.time() - loop_start
            sleep_time = frame_duration - elapsed
            if sleep_time > 0:
                time.sleep(sleep_time)

        # 正常結束：清空舊幀後傳入停止訊號
        try:
            data_queue.get_nowait()
        except queue.Empty:
            pass
        data_queue.put("STOP")
        render_proc.join(timeout=2.0)
                
    except KeyboardInterrupt:
        print("\n[!] Interrupted by user")
    finally:
        if render_proc.is_alive():
            render_proc.terminate()
            render_proc.join()
        print("[+] Test completed cleanly")


if __name__ == "__main__":
    # 確保 Windows / WSL 2 下多程序啟動模式一致
    mp.freeze_support()
    main()