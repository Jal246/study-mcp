from study_mcp.server import mcp


def main() -> None:
    # stdio = talk to the AI app through standard input/output (it launches us as a subprocess).
    mcp.run("stdio")
