"""User-facing CLI; heavy ML dependencies load only when a command runs."""

import argparse
import json
from pathlib import Path
from .config import DEFAULT_CONFIG, load_config, resolve_path


def positive_int(value):
    number = int(value)
    if number < 1:
        raise argparse.ArgumentTypeError("must be a positive integer")
    return number


def build_parser():
    parser = argparse.ArgumentParser(
        description="NMRReasoner · NMR 结构推断与化学位移预测",
        epilog="快速开始：python test.py --data /path/to/test_arrow --model /path/to/model --output runs/demo",
    )
    parser.add_argument("--debug", action="store_true", help="显示完整异常堆栈")
    commands = parser.add_subparsers(dest="command", required=True)
    for command, help_text in [
        ("inspect", "检查数据、完整权重和配置"),
        ("infer", "生成候选并自动评估"),
    ]:
        p = commands.add_parser(command, help=help_text, description=help_text)
        p.add_argument(
            "--debug",
            action="store_true",
            default=argparse.SUPPRESS,
            help="显示完整异常堆栈",
        )
        p.add_argument(
            "--config",
            default=str(DEFAULT_CONFIG),
            help="JSON 配置；其中数据/权重相对路径以配置文件目录为基准",
        )
        p.add_argument("--data", help="Arrow 数据目录（相对当前工作目录）")
        p.add_argument("--model", help="完整权重目录或只包含一个 checkpoint 的父目录")
        p.add_argument("--output", help="结果目录（相对当前工作目录）")
        p.add_argument(
            "--backend",
            choices=["transformers", "vllm"],
            help="推理后端，默认 vllm",
        )
        p.add_argument("--device", help="Transformers 设备，默认 cuda:0；可用 cpu")
        p.add_argument("--limit", type=int, help="前 N 条样本，默认 3；0 表示全量")
        p.add_argument(
            "--max-tokens", type=positive_int, help="每条最多新增 token 数，默认 5000"
        )
        p.add_argument("--beam-width", type=positive_int, help="束搜索候选数，默认 1")
        for name, default in [("temperature", 0), ("top-p", 0.95)]:
            p.add_argument("--" + name, type=float, help=f"采样参数，默认 {default}")
        p.add_argument("--top-k", type=int, help="采样 top-k，默认 20")
        p.add_argument(
            "--adjust-decimal-places",
            action=argparse.BooleanOptionalAction,
            default=None,
            help="统一输入位移小数位，默认关闭",
        )
        p.add_argument("--c-shift-decimal-place", type=int, help="碳位移小数位，默认 1")
        p.add_argument("--h-shift-decimal-place", type=int, help="氢位移小数位，默认 2")
        p.add_argument(
            "--tensor-parallel-size", type=positive_int, help="vLLM GPU 并行数，默认 1"
        )
        p.add_argument(
            "--gpu-memory-utilization", type=float, help="vLLM 显存比例，默认 0.7"
        )
    for command, help_text in [
        ("evaluate", "重新评估 TXT 预测"),
        ("vote", "聚合 JSONL 候选并计算 Top-K"),
        ("train", "使用 Unsloth 微调"),
    ]:
        p = commands.add_parser(command, help=help_text, description=help_text)
        p.add_argument(
            "--debug",
            action="store_true",
            default=argparse.SUPPRESS,
            help="显示完整异常堆栈",
        )
        p.add_argument(
            "--data", required=True, help="Arrow 数据目录，包含 question/cot/answer"
        )
        p.add_argument(
            "--output", required=True, help="评估 JSON 文件，或投票/训练输出目录"
        )
        if command == "train":
            p.add_argument("--model", required=True, help="本地基础模型目录")
            p.add_argument(
                "--epochs", type=positive_int, default=20, help="训练轮数，默认 20"
            )
            p.add_argument(
                "--batch-size",
                type=positive_int,
                default=64,
                help="每设备 batch size，默认 64",
            )
        else:
            p.add_argument(
                "--predictions",
                required=True,
                **({"nargs": "+"} if command == "vote" else {}),
                help=(
                    "JSONL 文件，优先级从高到低"
                    if command == "vote"
                    else "每行一个预测的 TXT 文件"
                ),
            )
        if command == "vote":
            p.add_argument(
                "--topk",
                type=positive_int,
                default=10,
                help="评估 Top-1 至 Top-K，默认 10",
            )
            p.add_argument(
                "--workers", type=positive_int, default=1, help="投票进程数，默认 1"
            )
    return parser


def dispatch(args):
    if args.command in ("inspect", "infer"):
        config = load_config(args.config, vars(args))
        if args.command == "infer":
            from .inference import infer

            return infer(config)
        from .assets import load_data, verify_task, checkpoint

        data = load_data(config["data"])
        result = dict(
            rows=len(data),
            columns=data.column_names,
            task=verify_task(data),
            checkpoint=str(checkpoint(config["model"])),
            config=config,
            first_question=data[0]["question"],
        )
        print(json.dumps(result, indent=2, ensure_ascii=False))
        return result
    # Match inference path expansion for evaluate, vote and train as well.
    for key in ("data", "model", "output"):
        if hasattr(args, key):
            setattr(args, key, resolve_path(getattr(args, key), Path.cwd()))
    if hasattr(args, "predictions"):
        if isinstance(args.predictions, list):
            args.predictions = [resolve_path(p, Path.cwd()) for p in args.predictions]
        else:
            args.predictions = resolve_path(args.predictions, Path.cwd())
    if args.command == "evaluate":
        from .evaluation import evaluate

        return evaluate(args.data, args.predictions, args.output)
    if args.command == "vote":
        from .evaluation import vote

        return vote(args.data, args.predictions, args.output, args.topk, args.workers)
    from .training import train

    return train(args.data, args.model, args.output, args.epochs, args.batch_size)


def main(argv=None):
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        return dispatch(args)
    except (OSError, ValueError, ImportError, RuntimeError) as exc:
        if args.debug:
            raise
        message = str(exc)
        if isinstance(exc, ImportError):
            message += "\n请激活项目环境并安装 requirements.txt；训练/vLLM 依赖见 runtime.yml。"
        parser.exit(
            2, f"错误：{message}\n使用 --help 查看参数，或 --debug 查看完整堆栈。\n"
        )
