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
        "--models-root", default=None, metavar="PATH",
        help="models directory (org/model layout)",
    )
    server.add_argument(
        "--config", default=None, metavar="PATH",
        help="YAML config file (settings baseline; env and CLI override it)",
    )
    server.add_argument(
        "--autoload",
        action=argparse.BooleanOptionalAction,
        default=None,
        help="allow loading any on-disk model on request "
        "(--no-autoload restricts to models: in the config file)",
    )

    args = parser.parse_args(argv)
    if args.command == "server":
        _run_server(args)


def _run_server(args: argparse.Namespace) -> None:
    import uvicorn

    from .backend import Backend
    from .errors import OvinferError
    from .server import create_app

    try:
        backend = Backend(
            models_root=args.models_root,
            default_device=args.device,
            autoload=args.autoload,
            config_file=args.config,
        )
        backend.preload_models()
    except OvinferError as exc:
        # ConfigError, ModelNotFound during preload, ... — clean message, no traceback.
        raise SystemExit(f"ovinfer: {exc}") from None
    host, port = args.listen or ("127.0.0.1", 8000)
    uvicorn.run(create_app(backend), host=host, port=port)


if __name__ == "__main__":
    main()
