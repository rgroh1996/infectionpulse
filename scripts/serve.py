"""Run cached InfectionPulse REST/OpenAPI and MCP tools without model inference."""

import argparse
import os


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--transport", choices=["http", "stdio"], default="http")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument("--bundle", help="Path to versioned forecast_bundle.json")
    parser.add_argument("--locations", help="Optional location catalog JSON or CSV")
    args = parser.parse_args()
    from infectionpulse.service import ForecastService, create_app

    service = ForecastService(args.bundle, args.locations)
    if args.transport == "stdio":
        from infectionpulse.mcp_server import create_mcp

        create_mcp(service).run(transport="stdio")
    else:
        if args.host not in ("127.0.0.1", "localhost", "::1") and not os.getenv(
            "INFECTIONPULSE_ACCESS_TOKEN"
        ):
            parser.error(
                "Remote binding requires INFECTIONPULSE_ACCESS_TOKEN (separate from TABPFN_TOKEN)"
            )
        import uvicorn

        uvicorn.run(create_app(service), host=args.host, port=args.port)


if __name__ == "__main__":
    main()
