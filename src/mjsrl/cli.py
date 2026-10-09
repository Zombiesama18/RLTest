"""One entry point for simulation, authorized replay data and offline research."""

from __future__ import annotations

import argparse
import importlib.metadata
import json
import platform
from dataclasses import asdict
from pathlib import Path

import torch

from .arena import arena
from .checkpoint import load_checkpoint
from .features import collate
from .ingest import ingest
from .model import ModelConfig
from .rollout import HeuristicAgent, ModelAgent, RandomAgent, play_game
from .schema import MahjongState
from .storage import read_transitions, write_transitions
from .train import TrainConfig, train


def json_file(path):
    return json.loads(Path(path).read_text(encoding="utf-8-sig"))


def generate(output, games=10, seed=42, mode="4p-red-half", agent="shanten", gamma=0.997):
    if games < 1:
        raise ValueError("At least one game is required")
    agent_class = HeuristicAgent if agent == "shanten" else RandomAgent

    def records():
        for i in range(games):
            agents = {seat: agent_class(seed + i * 4 + seat) for seat in range(4)}
            game = play_game(agents, seed + i, mode, gamma)
            print(
                json.dumps(
                    {"game": i + 1, "scores": game.scores, "decisions": len(game.transitions)}
                ),
                flush=True,
            )
            yield from game.transitions

    return write_transitions(output, records())


@torch.no_grad()
def infer(checkpoint, state_file, device="cpu"):
    state = MahjongState.from_dict(json_file(state_file))
    model, payload = load_checkpoint(checkpoint, device)
    model.eval()
    output = model(collate([state], device, model.config.history_length))
    probabilities = output["logits"][0].softmax(-1).cpu().tolist()
    selected, _ = ModelAgent(model, device, stochastic=False).act(state)
    return {
        "chosen_action": asdict(state.legal_actions[selected]),
        "candidates": [
            {"action": asdict(action), "probability": prob}
            for action, prob in zip(state.legal_actions, probabilities, strict=True)
        ],
        "state_value": float(output["value"][0]),
        "training_step": payload["step"],
        "context": "offline_replay",
    }


def smoke(output: Path, steps=10, device="cpu"):
    dataset = output / "data.parquet"
    generate(dataset, games=4, mode="4p-red-single")
    for algorithm in ("bc", "iql"):
        init = output / "bc" / "latest.pt" if algorithm == "iql" else None
        train(
            TrainConfig(
                algorithm=algorithm,
                steps=steps,
                device=device,
                precision="bf16" if device.startswith("cuda") else "fp32",
            ),
            output / algorithm,
            dataset,
            init=init,
            split=None,
        )
    train(
        TrainConfig(
            algorithm="ppo",
            steps=1,
            mode="4p-red-single",
            device=device,
            precision="bf16" if device.startswith("cuda") else "fp32",
        ),
        output / "ppo",
        init=output / "iql" / "latest.pt",
        split=None,
    )
    records = read_transitions(dataset)
    state_path = output / "example_state.json"
    state_path.write_text(json.dumps(records[0].state.to_dict(), indent=2), encoding="utf-8")
    result = infer(output / "ppo" / "latest.pt", state_path, device)
    (output / "inference.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    report = {
        "status": "passed",
        "bc_steps": steps,
        "iql_steps": steps,
        "ppo_updates": 1,
        "transitions": len(records),
        "inference_action": result["chosen_action"],
        "strength_verified": False,
        "device": device,
    }
    (output / "smoke.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    return report


def main():
    parser = argparse.ArgumentParser(description="雀魂四人麻将离线强化学习研究")
    sub = parser.add_subparsers(dest="command", required=True)
    sub.add_parser("doctor", help="检查依赖与 GPU")
    data = sub.add_parser("generate", help="模拟器生成样本（用于流程验证）")
    data.add_argument("--output", type=Path, required=True)
    data.add_argument("--games", type=int, default=10)
    data.add_argument("--seed", type=int, default=42)
    data.add_argument("--mode", default="4p-red-half")
    data.add_argument("--agent", choices=["shanten", "random"], default="shanten")
    data.add_argument("--gamma", type=float, default=0.997)
    replay = sub.add_parser("ingest", help="授权的已解码雀魂牌谱/MJAI → Parquet")
    replay.add_argument("--input", type=Path, required=True)
    replay.add_argument("--output", type=Path, required=True)
    replay.add_argument("--manifest", type=Path, required=True)
    replay.add_argument("--format", choices=["majsoul", "mjai"], default="majsoul")
    replay.add_argument("--gamma", type=float, default=0.997)
    validation = sub.add_parser("validate", help="逐条验证数据协议与合法动作标签")
    validation.add_argument("--data", type=Path, required=True)
    training = sub.add_parser("train", help="BC / IQL / PPO")
    training.add_argument("--config", type=Path, required=True)
    training.add_argument("--data", type=Path)
    training.add_argument("--validation", type=Path)
    training.add_argument("--output", type=Path, required=True)
    training.add_argument("--init", type=Path)
    training.add_argument("--resume", type=Path)
    training.add_argument("--steps", type=int)
    training.add_argument("--device")
    training.add_argument(
        "--split", choices=["train", "validation", "test", "all"], default="train"
    )
    inference = sub.add_parser("infer", help="已结束对局的结构化状态 → 策略")
    inference.add_argument("--checkpoint", type=Path, required=True)
    inference.add_argument("--state", type=Path, required=True)
    inference.add_argument("--device", default="cpu")
    evaluation = sub.add_parser("arena", help="匹配种子、轮换座位的模拟器评估")
    evaluation.add_argument("--candidate", type=Path, required=True)
    evaluation.add_argument("--baseline", type=Path)
    evaluation.add_argument("--output", type=Path, required=True)
    evaluation.add_argument("--seeds", type=int, default=10)
    evaluation.add_argument("--first-seed", type=int, default=10000)
    evaluation.add_argument("--mode", default="4p-red-half")
    evaluation.add_argument("--device", default="cpu")
    visual = sub.add_parser("vision", help="离线图片、标定 ROI、牌模板 → 状态估计")
    visual.add_argument("--frames", type=Path, nargs="+", required=True)
    visual.add_argument("--templates", type=Path, required=True)
    visual.add_argument("--profile", type=Path, required=True)
    visual.add_argument("--state", type=Path, required=True)
    visual.add_argument("--output", type=Path, required=True)
    demo = sub.add_parser("smoke", help="一条命令验证采样 → BC → IQL → PPO → 推理")
    demo.add_argument("--output", type=Path, default=Path("runs/smoke"))
    demo.add_argument("--steps", type=int, default=10)
    demo.add_argument("--device", default="cpu")
    args = parser.parse_args()
    if args.command == "doctor":
        result = {
            "python": platform.python_version(),
            "torch": torch.__version__,
            "riichienv": importlib.metadata.version("riichienv"),
            "cuda": torch.cuda.is_available(),
            "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
        }
    elif args.command == "generate":
        result = {
            "transitions": generate(
                args.output, args.games, args.seed, args.mode, args.agent, args.gamma
            )
        }
    elif args.command == "ingest":
        result = ingest(args.input, args.output, args.manifest, args.format, args.gamma)
    elif args.command == "validate":
        records = read_transitions(args.data)
        result = {
            "transitions": len(records),
            "episodes": len({r.episode_id for r in records}),
            "status": "valid",
        }
    elif args.command == "train":
        value = json_file(args.config)
        model_config = ModelConfig(**value.pop("model", {}))
        if args.steps is not None:
            value["steps"] = args.steps
        if args.device is not None:
            value["device"] = args.device
        train(
            TrainConfig(**value),
            args.output,
            args.data,
            args.init,
            args.resume,
            model_config,
            None if args.split == "all" else args.split,
            args.validation,
        )
        result = {"checkpoint": str(args.output / "latest.pt")}
    elif args.command == "infer":
        torch.set_num_threads(2)
        result = infer(args.checkpoint, args.state, args.device)
    elif args.command == "arena":
        result = arena(
            args.candidate,
            args.baseline,
            args.output,
            args.seeds,
            args.first_seed,
            args.mode,
            args.device,
        )
        result = {k: v for k, v in result.items() if k != "rows"}
    elif args.command == "vision":
        from .vision import OfflineVision

        recognizer = OfflineVision(args.templates)
        estimate = None
        for frame in args.frames:
            estimate = recognizer.recognize(frame, json_file(args.profile), json_file(args.state))
        result = asdict(estimate)
        if estimate.state is not None:
            result["state"] = estimate.state.to_dict()
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(result, indent=2), encoding="utf-8")
    else:
        result = smoke(args.output, args.steps, args.device)
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
