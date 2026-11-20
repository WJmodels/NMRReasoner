# 全量测试流水线

使用 GPU 3，正向 NCSP 与逆向 SE 各运行完整的 54,615 条测试数据；原始完整权重、vLLM、贪心解码、最多新增 5000 tokens。

自动顺序：NCSP 推理与自动评估 → NCSP 独立复算 → SE 推理与自动评估 → SE 独立复算 → 汇总报告。任一阶段失败则停止后续阶段，并记录失败状态。

- `status.json`：running / completed / failed。仅在状态变化时更新，不轮询。
- `ncsp_infer.log`、`se_infer.log`：实时推理进度。
- `progress.log`、`commands.json`：阶段完成状态、命令和退出码。
- `ncsp/`、`se/`：完整预测、自动指标和独立复算指标。
- `REPORT.md`、`summary.json`：全部完成并核对行数、复算一致性后生成。
- `finalize.log`：汇总步骤日志。

运行中的主流水线 PID：2578914。后台管理进程已脱离终端（PPID=1、无控制终端、忽略 SIGHUP），通过 Linux pidfd 阻塞等待主流水线退出，无定时检测。如果原进程随终端关闭而提前退出，后台管理进程会自动接管未完成阶段；已完成阶段保留，尚未完成的整项推理重新运行。完成后自动生成汇总报告；运行错误记为 failed，不伪报成功。服务器需保持开机。
