"""Console entry point: `ovinfer` with the `server` subcommand.

Subcommand structure leaves room for `ovinfer models ...` in v0.2.
CLI arguments override OVINFER_* env vars (None means "not given here,
fall through to env/defaults" in config.load_settings).
"""

import argparse

from . import __version__


def _parse_listen(raw: str) -> tuple[str, int]:
    host, sep, port = raw.rpartition(":")
    if not sep or not host:
        raise argparse.ArgumentTypeError(f"expected HOST:PORT, got {raw!r}")
    try:
        return host, int(port)
    except ValueError:
        raise argparse.ArgumentTypeError(
            f"port must be an integer, got {port!r}"
        ) from None


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(
        prog="ovinfer",
        description="OpenVINO GenAI inference server with an OpenAI-compatible API",
    )
    parser.add_argument(
        "--version", action="version", version=f"ovinfer {__version__}"
    )
    sub = parser.add_subparsers(dest="command", required=True)

    server = sub.add_parser("server", help="run the OpenAI-compatible HTTP server")
    server.add_argument(
        "--listen",
        type=_parse_listen,
        default=None,
        metavar="HOST:PORT",
        help="bind address (default 127.0.0.1:8000)",
    )
    server.add_argument(
        "--device", default=None, metavar="DEV",
        help="default inference device (CPU/GPU/NPU)",
    )
    server.add_argument(
        "--device-map", default=None, metavar="MODEL=DEV[,...]",
        help="per-model device overrides",
    )
    server.add_argument(
        "--preload", default=None, metavar="MODEL[:KIND][,...]",
        help="models to warm up at startup (kind: llm|embedding|rerank, default llm)",
    )
    server.add_argument(
        "--models-root", default=None, metavar="PATH",
        help="models directory (org/model layout)",
    )

    args = parser.parse_args(argv)
    if args.command == "server":
        _run_server(args)


def _run_server(args: argparse.Namespace) -> None:
    import uvicorn

    from .backend import Backend
    from .config import parse_device_map, parse_preload
    from .errors import OvinferError
    from .server import create_app

    try:
        backend = Backend(
            models_root=args.models_root,
            default_device=args.device,
            device_map=(
                parse_device_map(args.device_map)
                if args.device_map is not None
                else None
            ),
            preload=(
                parse_preload(args.preload) if args.preload is not None else None
            ),
        )
        backend.preload_models()
    except OvinferError as exc:
        # ConfigError, ModelNotFound during preload, ... — clean message, no traceback.
        raise SystemExit(f"ovinfer: {exc}") from None
    host, port = args.listen or ("127.0.0.1", 8000)
    uvicorn.run(create_app(backend), host=host, port=port)


if __name__ == "__main__":
    main()
