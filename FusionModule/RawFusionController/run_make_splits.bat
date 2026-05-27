@echo off
REM 修改 D:\ 为你的数据根目录（里面应有 scene_1 ... scene_9）
python tools\make_splits.py --data_root D:\ --out_dir data_catalog
pause
