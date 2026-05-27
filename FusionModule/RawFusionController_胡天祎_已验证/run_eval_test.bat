@echo off
python scripts\eval.py --ckpt checkpoints\full_motion_aware\best.pt --list data_catalog\test.txt --out_json outputs\metrics\test_eval.json
pause
