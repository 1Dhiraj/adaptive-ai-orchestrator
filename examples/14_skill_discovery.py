#!/usr/bin/env python3
"""Finding a capability the system does not have yet -- and asking first.

When no built-in tool, MCP server or installed skill covers a step, the
orchestrator can go looking: the official MCP registry for servers, public
repositories for procedural skills. What it will not do is install anything
on its own.

The flow is search -> review -> approve -> connect, and every stage is
refusable:

  * candidates are shown with their source URL, publisher and version, so
    "where did this come from?" is answerable before anything is trusted;
  * connecting requires the exact digest of the reviewed candidate, so
    approving one thing and installing another is not possible;
  * approval is explicit -- ``approved=True`` is the only way through, and a
    wrong or stale digest is refused;
  * npm packages install with --ignore-scripts, so a package cannot run code
    merely by being fetched;
  * only an allow-list of hosts is contactable at all.

Search results are mocked here, so this example needs no network and
installs nothing. The refusals below are real: they come from the same code
path a live search would use.
"""

# Run from anywhere without installing: put the project root on sys.path.
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import tempfile

from orchestrator.acquisition import AcquisitionError, CapabilityAcquirer


def fake_registry_results():
    """What a search of the MCP registry would return, shaped like the real thing."""
    return [
        {
            "name": "io.modelcontextprotocol/filesystem",
            "version": "1.2.0",
            "description": "Read and write files under a chosen directory",
            "publisher": "modelcontextprotocol",
            "source": "https://registry.modelcontextprotocol.io/v0.1/servers/filesystem",
            "why": "the step needs to read local files and no tool provides that",
        },
        {
            "name": "io.modelcontextprotocol/fetch",
            "version": "0.4.1",
            "description": "Fetch a URL and return its contents as text",
            "publisher": "modelcontextprotocol",
            "source": "https://registry.modelcontextprotocol.io/v0.1/servers/fetch",
            "why": "the step mentions reading a web page",
        },
    ]


def main() -> int:
    print("=" * 72)
    print("A step needs a capability nothing installed provides")
    print("=" * 72)
    print("  step  : 'Summarise the notes in ./research and cite each source'")
    print("  needs : read local files")
    print("  have  : no filesystem tool, no matching MCP server, no skill")

    print()
    print("=" * 72)
    print("1. Search trusted sources  (mocked here; no network is used)")
    print("=" * 72)
    candidates = fake_registry_results()
    for i, candidate in enumerate(candidates, 1):
        print(f"\n  [{i}] {candidate['name']}  v{candidate['version']}")
        print(f"      {candidate['description']}")
        print(f"      publisher : {candidate['publisher']}")
        print(f"      source    : {candidate['source']}")
        print(f"      why       : {candidate['why']}")

    print()
    print("=" * 72)
    print("2. Nothing is installed without an explicit, matching approval")
    print("=" * 72)

    with tempfile.TemporaryDirectory() as tmp:
        acquirer = CapabilityAcquirer(Path(tmp), "example")
        try:
            print("\n  Connecting something that was never reviewed:")
            try:
                acquirer.connect("io.modelcontextprotocol/filesystem",
                                 digest="made-up-digest", approved=True)
            except AcquisitionError as exc:
                print(f"    REFUSED: {exc}")

            print("\n  Approving with the right id but no approval flag:")
            try:
                acquirer.connect("io.modelcontextprotocol/filesystem",
                                 digest="made-up-digest", approved=False)
            except AcquisitionError as exc:
                print(f"    REFUSED: {exc}")

            print("\n  Reviewing something that was never found by a search:")
            try:
                acquirer.review("something-nobody-searched-for")
            except AcquisitionError as exc:
                print(f"    REFUSED: {exc}")

            print(f"\n  Installed extensions after all that: {acquirer.list()}")
        finally:
            acquirer.close()

    print()
    print("=" * 72)
    print("What a real approval looks like")
    print("=" * 72)
    print("  acquirer.search_tools('read local files')   -> candidates + digests")
    print("  acquirer.review(candidate_id)               -> full source + preview")
    print("  acquirer.connect(candidate_id, digest, approved=True)")
    print()
    print("  The digest binds the approval to exactly what was reviewed, so a")
    print("  candidate that changed between review and approval is refused.")
    print("  Web content is treated as data: instructions found inside a README")
    print("  or a tool description are never followed.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
