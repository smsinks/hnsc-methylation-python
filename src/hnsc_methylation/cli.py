"""Command-line entry points using the same functions as the notebook."""
from pathlib import Path
import argparse
from .pipeline import MethylationPipeline, PipelineConfig, _data_tools


def main():
    parser = argparse.ArgumentParser(description="Run the HNSC methylation manuscript pipeline.")
    commands = parser.add_subparsers(dest="command", required=True)
    for command in ["run", "verify"]:
        child = commands.add_parser(command)
        child.add_argument("--repo-root", type=Path, default=Path.cwd())
        if command == "run":
            child.add_argument("--output", type=Path, default=Path("results"))
            child.add_argument("--permutations", type=int, default=100)
    args = parser.parse_args()
    root = args.repo_root.resolve()
    if args.command == "verify":
        tools = _data_tools(root)
        integrity = tools.verify_package(root)
        inputs = tools.materialize_inputs(root)
        print(f"Verified {integrity['stored_files_checked']} bundled files "
              f"and prepared {len(inputs)} source inputs.")
    else:
        pipeline = MethylationPipeline(PipelineConfig(root, args.output, args.permutations))
        report = pipeline.run()
        print(report.to_string(index=False))


if __name__ == "__main__":
    main()
