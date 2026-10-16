# obsidian-semantic-mcp

## Project

- Native Python 3.14 MCP server for Obsidian; use `uv` for development.
- `osm` is the project CLI. The version is defined in `pyproject.toml`.
- Preserve Obsidian notes and existing database data. Do not mix vectors from different embedding models.
- Do not commit or push without the owner's request.
- Read `project-guide/ARCHITECTURE.md` before changing runtime flows or storage behavior.

## Versioning

- Keep the project version in `pyproject.toml` as `MAJOR.MINOR.PATCH`.
- For small fixes and documentation changes, increment `PATCH` by 1 (for example, `0.20.0` → `0.20.1`).
- For substantial features or changes in behavior, increment `MINOR` by 1 and reset `PATCH` to 0 (for example, `0.20.3` → `0.21.0`).
- Do not change `MAJOR` without an explicit decision from the owner.

## Installed command

An installed command must not depend on the source checkout for execution, configuration, or data. Use paths relative to the installation root or standard user configuration paths such as `~/.config/osm/`.

## Code style

- Run `uv run ruff check osm_cli.py src` and `uv run ruff format --check osm_cli.py src` after editing runtime Python files.
- Run `uv run ty check osm_cli.py src` after changing Python types or interfaces.
