# Smart Main Program
### 專案架構
```bash=
|-- README.md
|-- deploy
|   `-- src
|       `-- main_client
|           |-- launch
|           |   `-- launch.py
|           |-- main_client
|           |   |-- __init__.py
|           |   |-- demo_node.py
|           |   |-- main_client.py
|           |   |-- sensor_manager.py
|           |   `-- fake_firmware.py
|           |-- package.xml
|           |-- params
|           |   `map_point.yaml
|           |-- resource
|           |-- setup.cfg
|           |-- setup.py
|           `-- test
|-- docker
|   |-- Dockerfile
|   |-- Dockerfile_light
|   |-- compose.yaml
|   `-- compose_light.yaml
`-- train
    |-- logs
    |   |-- PPO_17
    |   `-- PPO_18
    |-- models
    |   |-- robot_strategy_v1_500K.zip
    |   `-- robot_strategy_v1_800K.zip
    `-- src
        |-- __pycache__
        |-- game_env.py
        |-- test_and_plot.py
        `-- train.py
```

### GUI 顯示除錯
- 手動將 tkinter 的位置連結到安裝 python 工具的虛擬環境
   ```bash=
   ln -s /usr/lib/python3.10/lib-dynload/_tkinter.cpython-310-x86_64-linux-gnu.so /venv/lib/python3.10/site-packages/
   ```

### 使用 tensorboard
- 在本地安裝並執行 tensorboard
   ```bash=
   python -m tensorboard.main --logdir=./logs
   ```
- 在瀏覽器開啟 localhost/6006

### 還未更新的 docker 環境變化
- 安裝 sb3-contrib
   ```bash=
   pip install sb3-contrib
   ```