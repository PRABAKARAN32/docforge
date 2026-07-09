# CLI reference

The `docforge` command. Every subcommand also documents itself: `docforge --help` or
`docforge <command> --help`.

```bash
docforge sync <url> [options]     # crawl, detect changes, embed into the vector store
docforge diff <url> [options]     # preview what would change — write nothing
docforge status                   # list knowledge bases and their page counts
docforge list                     # alias for status
docforge search "<query>" [options]   # search one knowledge base or all of them
docforge remove <name>            # drop a knowledge base (or --all to wipe everything)
```

A vector store is required for `sync`/`search`/`remove` — see [`CONFIGURATION.md`](CONFIGURATION.md)
for the three ways to provide one (Docker, embedded, remote).

---

## `docforge sync <url>`

Discover a site's pages, crawl them, detect exactly what changed since the last run, and embed
only the new/changed pages into the vector store — deleting stale chunks first. Re-running with
no changes does nothing.

**Positional:** `url` — the documentation site to sync.

### Crawling flags

| Flag | Default | Meaning |
|---|---|---|
| `--bfs` | off | If the site has no sitemap, crawl it page-by-page instead of giving up. |
| `--max-pages N` | no limit | Cap on pages processed. |
| `--concurrency N` | `5` | Pages crawled in parallel. Higher is faster but heavier on the target server. |
| `--crawl-delay MIN MAX` | `0.5 1.5` | Random delay range (seconds) between requests to the *same* site — the real throughput knob on a single big domain. |
| `--no-rate-limit` | off | Remove the per-site delay entirely. Fastest, least polite; may trigger 429s. |

### Change-detection flags

| Flag | Default | Meaning |
|---|---|---|
| `--name NAME` | derived from host | Knowledge-base name — its own manifest scope + Qdrant collection. |
| `--db PATH` | `docforge.db` (or `DOCFORGE_DB`) | Manifest database file. |
| `--conditional {auto,on,off}` | `auto` | Use HTTP conditional requests (ETag/304) to skip re-crawling unchanged pages. `off` disables the pre-check entirely. |
| `--force` | off | Ignore stored validators and re-crawl every page (skips the 304 pre-check). |
| `--dry-run` | off | Show what would change without writing anything — no embedding, no manifest update. |

### Vector-store flags

| Flag | Default | Meaning |
|---|---|---|
| `--qdrant-url URL` | `http://localhost:6333` (or `QDRANT_URL`) | Qdrant server: Docker, native install, or remote. |
| `--qdrant-path DIR` | unset | Run Qdrant embedded (no Docker), vectors stored in this folder. Takes precedence over `--qdrant-url`. |
| `--qdrant-api-key KEY` | unset (or `QDRANT_API_KEY`) | API key for a remote/managed Qdrant. Prefer the env var over the flag — flags land in shell history. |
| `--qdrant-timeout SECONDS` | `60` | Request timeout. Raise it for a slow or distant cluster. |

### Embedding flags

| Flag | Default | Meaning |
|---|---|---|
| `--embed-model NAME` | `BAAI/bge-small-en-v1.5` (or `DOCFORGE_EMBED_MODEL`) | fastembed model name. Must stay the same across syncs of one knowledge base — switching models changes the vector dimension. |
| `--device {auto,cpu,cuda}` | `auto` | Embedding compute device. `auto` uses a GPU if available, else CPU; `cuda` falls back to CPU with a warning if GPU setup fails. |

### Examples

```bash
docforge sync https://docs.example.com/               # build / refresh the knowledge base
docforge sync https://nginx.org/en/docs/ --bfs         # no sitemap? crawl page-by-page
docforge sync <url> --dry-run                          # preview changes, write nothing
docforge sync <url> --max-pages 50 --concurrency 10    # cap pages; crawl 10 in parallel
docforge sync <url> --qdrant-path ./vectors             # no Docker (Qdrant embedded on disk)
docforge sync <url> --device cuda                       # use a GPU for embedding
docforge sync <url> --force                             # ignore ETag/304, re-crawl everything
```

---

## `docforge diff <url>`

Same discovery/crawl/change-detection flags as `sync` (crawling + change-detection groups
above, minus `--dry-run` — `diff` never writes by definition), but only *lists* what would
change instead of embedding or writing anything:

```bash
docforge diff https://docs.example.com/
docforge diff https://docs.example.com/ --name docker --bfs
```

---

## `docforge status` / `docforge list`

List every tracked knowledge base and its page count. Identical commands, two names.

| Flag | Default | Meaning |
|---|---|---|
| `--db PATH` | `docforge.db` (or `DOCFORGE_DB`) | Manifest database file to inspect. |

```bash
docforge status
#   1578  docs_docker_com
#     84  sitemaps_org
```

---

## `docforge search "<query>"`

Embed a query and return the most similar chunks.

**Positional:** `query` — natural-language search text.

| Flag | Default | Meaning |
|---|---|---|
| `--name NAME` | unset | Search only this knowledge base. |
| `--all` | — | Search across all knowledge bases (this is already the default when `--name` is omitted). |
| `--limit N` | `5` | Number of results. |
| `--db PATH` | `docforge.db` (or `DOCFORGE_DB`) | Manifest database (used to list KBs when searching all). |

Plus the same vector-store and embedding flags as `sync` (`--qdrant-url`, `--qdrant-path`,
`--qdrant-api-key`, `--qdrant-timeout`, `--embed-model`, `--device`).

```bash
docforge search "how to configure crawling"              # searches every knowledge base
docforge search "how to configure crawling" --name docker # scoped to one
docforge search "installation steps" --limit 10
```

---

## `docforge remove <name>`

Delete a knowledge base by name — its manifest pages and its vector-store collection. Or
`--all` to wipe *every* knowledge base **and** the manifest database file itself.

**Positional:** `name` — the knowledge base to remove (optional if `--all` is given).

| Flag | Default | Meaning |
|---|---|---|
| `--all` | off | Remove everything: all collections + the manifest DB file (and embedded vectors folder). |
| `--db PATH` | `docforge.db` (or `DOCFORGE_DB`) | Manifest database file. |

Plus the same vector-store flags as `sync` (`--qdrant-url`, `--qdrant-path`, `--qdrant-api-key`,
`--qdrant-timeout`).

```bash
docforge remove docker       # drop one knowledge base
docforge remove --all        # wipe everything -- manifest file included
```

---

## Exit codes

`0` on success (including "nothing to do" — an idempotent no-op sync is still success). `1` on
a real failure: no pages discovered, the vector store is unreachable, or an explicit usage
error (e.g. `remove` with neither a name nor `--all`).
