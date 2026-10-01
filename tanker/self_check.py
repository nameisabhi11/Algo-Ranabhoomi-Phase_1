import _paths  # noqa: F401
import sys

from benchkit.cli import self_check_main

if __name__ == "__main__":
    import metrics
    args = sys.argv[1:]
    if not any(arg == "--adapter" or arg.startswith("--adapter=") for arg in args):
        args = ["--adapter", "adapters.mine:MySolver"] + args
    raise SystemExit(self_check_main(metrics.BENCH, args))
