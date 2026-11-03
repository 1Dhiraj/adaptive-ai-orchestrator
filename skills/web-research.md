---
name: web-research
description: Turning browser steps into sourced findings instead of a description of pages visited
keywords: research, find out, look up, compare, investigate, sources, latest, current, news, pricing, documentation
---

# Research that someone can check

A research step produces **claims with sources**. A list of pages visited is
not a finding, and neither is a summary that no URL supports.

## Search, then read the thing you found

A search results page is an index, not a source. Reading only the search page
gives you titles and snippets written to be clicked, not facts. Follow the
result and read the page itself — which is why research steps usually want
autonomous mode:

```
TOOL_DIRECTIVE: {"arguments": {
  "goal": "find the current stable Python version and the date it was released",
  "targets": ["google.com", "python.org"],
  "autonomous": true}}
```

Put every host you may need in `targets`: the search engine *and* the sites
you expect to follow into. A result you are not allowed to open is a result
you cannot use.

## Prefer the primary source

For a product's pricing, the product's own pricing page. For a release date,
the project's own release notes. For a law, the statute. A secondary source
is worth citing when it is the only one, and worth checking when it is not.

## Record what you actually read

For each finding, keep:

- the claim, in one sentence
- the URL it came from
- when the page said it was updated, if it said

Use `find '<text>'` to confirm a specific figure is really on the page rather
than inferred from a snippet. It searches the page and returns the matching
lines, which is both cheaper and more honest than pasting the whole document
into your answer.

## Say what you could not establish

An explicit "the pricing page did not give a figure for annual billing" is a
useful result. A confident number that no page stated is not, and nobody
downstream can tell the difference without redoing the work.

Where sources disagree, report both and say which you trust and why. Do not
average them, and do not silently pick one.

## Dates and staleness

"Latest" is a claim about a moment. Say which version or figure was current
and when you read it, so a reader in three months knows what they are holding.
