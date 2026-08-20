# Final v3 Artifact Manifest

本清单对应 `outputs/checkpoints/joint_v3/best.pt` 生成的最终融合结果。哈希算法为 SHA-256；大小为字节。

| Artifact | Bytes | SHA-256 |
|---|---:|---|
| `checkpoints/joint_v3/best.pt` | 2451207 | `b2f7520e2cca68ac5c635b7294ed0f917bca13b23805a0517e8fe8d080f92f04` |
| `checkpoints/joint_v3/summary.json` | 439 | `dedb5df2df10ce5dca880d526664d2ca96ee51e3b7658c9a294be177374f9ee6` |
| `raw/645x_learned_fusion.raw` | 829440000 | `b8885dac67608be9069ab76c55a1171ecf07d8828824ce36fc11d76b59ec1f7d` |
| `raw/128x_learned_fusion.raw` | 829440000 | `9b940550d96b1c4d65a1da27c677f35acf9eb89f2e7e4f3415bbc822e0c46b80` |
| `gates/645x_gate_u8.raw` | 103680000 | `777edf3289787f9da279d6d8f3471363fc72686930beeb7e37414b555e57aa1e` |
| `gates/128x_gate_u8.raw` | 103680000 | `aadf4b074325354d5533a0349705880bcd951cf37d6b0c7fa7aebf0953d49ffd` |
| `videos/645x_comparison.mp4` | 11149257 | `1e867f92e61c147fe3497d82e4d0dc18815cf569029af2627a4dcc9f37eb7d7e` |
| `videos/645x_comparison.json` | 560 | `3251d6a75eec25410a3c7a2557b9e1e856e56c57335e0694dc8fc5a45167f30b` |
| `videos/128x_comparison.mp4` | 8111234 | `f45e5d30db58f4d833ac28800c58b3a1bd67d9109c7d5df8f5f866b1c056f349` |
| `videos/128x_comparison.json` | 564 | `ebb0a9e673d3240951ced9089a34eca62ac9228dec151749708a729191bd0724` |
| `images/645x_comparison_frame_0050.png` | 2348918 | `f3695251e1653935a585c904ceae6549f4861a8cf33642df8835deb189a57981` |
| `images/128x_comparison_frame_0050.png` | 2140889 | `c888c1be3061a3f449138b2a8959af1be7d7d9aa974b39cf808b43736ae658e6` |
| `metrics/645x_evaluation.json` | 1087 | `d3e97f2ad3685c979b88b6e6e5795f0a4e4d77502c5aa4bd83268c1ee60aa792` |
| `metrics/128x_evaluation.json` | 1086 | `fd8f32008bc2b01f38f5ec4a4d5fb4f9ebf95a1d6163013fe383a3662e4cce76` |
| `metrics/645x_stability.json` | 1027 | `c42a6abf82a00d62bce7069d05daf5ea6d81848a83f9d65732730dc42ad1509e` |
| `metrics/128x_stability.json` | 1008 | `87cd3baea04497b2c5326cc2617959050600db48e059041135efb25ed7d2e396` |

视频元数据：200 帧、20 fps、10 秒、1920x720。RAW 元数据：200 帧、1920x1080、`uint16` little-endian；融合流为候选有效 DN，gate 流为 `uint8`（除以 255 得到 `[0,1]`）。

校验命令示例：

```powershell
Get-FileHash .\outputs\checkpoints\joint_v3\best.pt -Algorithm SHA256
Get-FileHash .\outputs\videos\645x_comparison.mp4 -Algorithm SHA256
```

旧的 `outputs/MANIFEST.md` 是前序 v1/v2 汇总，保留用于追溯；发布和验收请使用本文件。
