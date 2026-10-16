# Obsidian Semantic MCP

OSM adds semantic, keyword, and hybrid search for Obsidian notes to local MCP clients. It reads your Markdown vaults, uses local Ollama for embeddings, and stores the index in either **Qdrant** or **PostgreSQL + pgvector**. Your notes remain in Obsidian. The command is `osm`; the default embedding model is `qwen3-embedding:0.6b`.

## Install and configure

On macOS, install OSM from the Homebrew tap. Homebrew installs the required Python and build tools:

```sh
brew install joynik/tap/osm
```

Install only the services you do not already have:

```sh
brew install ollama
brew services start ollama
ollama pull qwen3-embedding:0.6b
```

Choose **one** database:

| Qdrant | PostgreSQL + pgvector |
| --- | --- |
| `brew install joynik/tap/qdrant` | `brew install postgresql@18 pgvector` |
| `brew services start qdrant` | `brew services start postgresql@18` |
| OSM creates its collection during `osm init`. | Create a database and enable the vector extension as shown below. |

For PostgreSQL, use your installed PostgreSQL major version if it differs from 18:

```sh
createdb obsidian_brain
psql -d obsidian_brain -c "CREATE EXTENSION IF NOT EXISTS vector;"
```

A local database can use `postgresql://localhost/obsidian_brain` when your PostgreSQL account permits it. Otherwise, enter a URL with your database user and password during `osm init`. Do not recreate an existing database. Check running Homebrew services with `brew services list`.

After Ollama and the database are ready, configure OSM:

```sh
osm init
osm status
```

`osm init` asks for vault paths, database, Ollama URL, model, and other settings. You can add more vaults by running it again. It saves installed settings in `~/.config/osm/runtime.json`. Existing settings are copied there automatically on first use. To update OSM after a new tap release, run `brew update && brew upgrade joynik/tap/osm`.

For development from this checkout, install [uv](https://docs.astral.sh/uv/), run `uv sync --python 3.14`, copy `.env.example` to `.env`, fill in your paths and URLs, then use `uv run osm ...`. Saved `osm init` settings take precedence over the checkout `.env`.

The wizard asks separately before connecting each MCP client and before granting that client write access. Both answers default to **No**. Files in folders named `archive` are excluded from indexing by default; `osm init` can include them.

## Run and check

An MCP client starts `osm run` when it connects. For a manual launch, keep `osm run` running in its terminal. Ollama and the chosen database must be available whenever OSM indexes or searches; `brew services` can start them after login. The dashboard runs only when requested.

```sh
osm                 # Choose a command with the arrow keys
osm status          # Check Ollama, database, and vaults
osm vaults          # Show configured vaults
osm logs --follow   # Watch server activity
osm dashboard       # Open the optional local dashboard
osm dashboard stop
osm stop            # Stop local OSM server processes
```

If Ollama is unavailable, a running OSM server keeps retrying; on macOS it notifies you after two minutes. `osm stop` does not stop Ollama or the database. A connected MCP client can start OSM again.

## Connect clients manually

You can connect clients during `osm init` or use the entries below. Restart the client after connecting. `OSM_ALLOW_WRITE=0` keeps note-writing tools disabled for that client; change it to `1` only when you want that client to edit notes.

| Client | Command |
| --- | --- |
| Codex | `codex mcp add obsidian-semantic --env OSM_ALLOW_WRITE=0 -- osm run` |
| Claude Code | `claude mcp add --scope user obsidian-semantic -e OSM_ALLOW_WRITE=0 -- osm run` |
| Pi | `pi mcp add obsidian-semantic --env OSM_ALLOW_WRITE=0 -- osm run` |

For OpenCode, add this entry inside `mcp` in `~/.config/opencode/opencode.json`, keeping your other entries:

```json
{
  "obsidian-semantic": {
    "type": "local",
    "command": ["osm", "run"],
    "enabled": true,
    "environment": {"OSM_ALLOW_WRITE": "0"}
  }
}
```

For Claude Desktop on macOS, add an `obsidian-semantic` entry under `mcpServers` in `~/Library/Application Support/Claude/claude_desktop_config.json` with `"command": "osm"`, `"args": ["run"]`, and `"env": {"OSM_ALLOW_WRITE": "0"}`. Run `osm disconnect` to remove OSM from supported clients; it does not remove notes, settings, or indexed data.

## Search and maintenance

OSM updates the index after note changes and searches long notes in passages. Semantic search uses Ollama; keyword search uses the database index. PostgreSQL text search defaults to Russian; choose English or simple during `osm init`. Qdrant uses Russian BM25 for keyword search.

To change the embedding model, use migration rather than editing `EMBEDDING_MODEL` on an existing index:

```sh
ollama pull MODEL
osm migrate --embedding-model MODEL --dry-run
osm migrate --embedding-model MODEL
```

Migration builds new vectors before switching the active index. The dashboard's **Safe Rebuild** similarly keeps the previous index until a new one is ready. If you stop indexing `archive` folders after they were already indexed, use Safe Rebuild to remove those notes from active search. Neither operation deletes Obsidian files.

Other commands: `osm open --vault PATH NOTE.md` opens a note in Obsidian; `osm version`, `osm -v`, and `osm -V` check the installed version; `osm update` shows the Homebrew update command; `osm help` lists all commands.

## Tests

Run the full suite from the project directory with PostgreSQL, Qdrant, and Ollama running. `osm_codex_test` must be a separate test database with the `vector` extension; the PostgreSQL tests change its tables. The Qdrant test creates and removes its own collection.

```sh
PYTEST_DATABASE_URL=postgresql://localhost/osm_test \
PYTEST_QDRANT_URL=http://127.0.0.1:6333 \
PYTEST_OLLAMA_URL=http://127.0.0.1:11434 \
uv run pytest -q
```
