```shell=
pip install tensorboard
pip install shimmy
apt-get update && apt-get install -y python3-tk
# 找出系統 tkinter 的位置並連結到虛擬環境 (路徑可能隨版本微調)
ln -s /usr/lib/python3.10/lib-dynload/_tkinter.cpython-310-x86_64-linux-gnu.so /venv/lib/python3.10/site-packages/
```