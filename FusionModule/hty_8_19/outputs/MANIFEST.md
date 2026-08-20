# 交付文件完整性清单

生成时间：2026-08-18。哈希算法：SHA-256。RAW 文件均为连续小端 `uint16` 流；gate 文件为 packed 半分辨率 `uint8` 流。

| 文件 | 字节数 | SHA-256 |
|---|---:|---|
| `README.md` | 5890 | `227dc23a98c897dcac6d0a3cab4e2ec9ce4193e306ea34af30a1a2dd4cada081` |
| `configs/company.yaml` | 2007 | `ff28bf0e4428bb76b8997f1b0fc0a419d170d1d4de3934178de5427fc64148f5` |
| `docs/hty8.17_original.md` | 13919 | `7c413175f936ee9de86dfd6d5a8d3fc32fb09501cee132bdc6c907e81bee84bb` |
| `docs/方案说明_修订版_20260818.md` | 7194 | `dc07635b2679e8dab7e1b5cd9c9bbff07bb6003ba904721f3bb1fd0ce9ed1471` |
| `reports/公司数据融合结果_20260818.md` | 4101 | `ec5db8511cda350511adc2b8b82e6ac9f4e2079476e4401b7b62401b87b7fe8f` |
| `outputs/checkpoints/joint_v2/best.pt` | 2451207 | `b4bcde8babd12a4e9c6ba040ce6ddc7e6833953ca924216be54c6a7d389b5423` |
| `outputs/raw/645x_learned_fusion.raw` | 829440000 | `883524b712b50d866b3932359cfd7516efa0b702974d0c8ed57986e537de0fa9` |
| `outputs/raw/128x_learned_fusion.raw` | 829440000 | `9899db6383d9872f9ddb5d93c61385f7eebc9fdaab19c8b3d0d7773ea71b0de2` |
| `outputs/gates/645x_gate_u8.raw` | 103680000 | `078c9c596a2763fc3328c2daf1e9c1e29ccf00fb9349371545fb62e0c01d09f7` |
| `outputs/gates/128x_gate_u8.raw` | 103680000 | `438b5126eccbe2926b36082e635f660b5eb8b5187c281ab96dc1bd4f962c2ca7` |
| `outputs/videos/645x_comparison.mp4` | 11149808 | `ee8d8e4c72d1b0bc2d9f0a365616c52b374eee88045943f8cfcabde8d54b6b4d` |
| `outputs/videos/128x_comparison.mp4` | 8112400 | `2e2d5e40e2de632827a1317fcdb812df22c01a0162e8e09e4636dc0e786e91a8` |
| `outputs/metrics/645x_evaluation.json` | 1088 | `a9006de2c419beaac0676fa402b1e50f43b8f80e5a6e5d3b59afb48b64d884f9` |
| `outputs/metrics/128x_evaluation.json` | 1087 | `2372f4cc39cd24e804ec4d4f8feefad5c0a434ae281e8c42a76e0847782e2073` |
| `outputs/metrics/645x_stability.json` | 1026 | `9541cb341d5c83690e890c7d77eecc40963ecdd7d547d4898363137e8e79b0a8` |
| `outputs/metrics/128x_stability.json` | 1026 | `dd3ac9aca23afe257eb583bc030eca1bc2751f7ea8a697e6cef3b5940f9bd18f` |

视频临时 ASCII 路径副本与交付目录副本已逐字节核对；两段视频均可由 OpenCV 读取 200 帧、1920x720、20 fps。
