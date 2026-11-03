---
name: browser-tasks
description: How to drive a real browser through computer_use so the work actually happens
keywords: browser, website, web page, url, navigate, click, search, form, login, sign in, scrape, extract, download, tab, screenshot
---

# Driving a browser

A browser step succeeds when the *goal* is reached, not when a page opens.
"Opened google.com" is not a search result, and reporting it as one is the
single most common way these runs go wrong.

## Choose the mode before you write the directive

**Scripted** — you can name every step in advance, because the page is known
and stable:

```
TOOL_DIRECTIVE: {"arguments": {
  "goal": "read the current Python release number",
  "targets": ["python.org"],
  "actions": ["open the page", "read the page"]}}
```

**Autonomous** — anything stands between you and the goal that you cannot see
yet: a search you must read before clicking, a consent banner, a login, a
multi-page form, a result whose position you cannot predict.

```
TOOL_DIRECTIVE: {"arguments": {
  "goal": "find the maintainer's name on the project's GitHub page and report it",
  "targets": ["github.com"],
  "autonomous": true}}
```

Autonomous mode reads the page after every action and chooses the next one
from what is actually there. Prefer it whenever a scripted plan would be a
guess. A guess that misses costs the whole step; a loop that misses costs one
turn.

## Targets are permissions, not itinerary

`targets` is the allow-list this step may touch. Every host you might need
belongs in it — including the one a search result leads to. It does **not**
mean "open all of these": naming `google.com` and `github.com` means you may
search and then follow a result, not that the GitHub home page gets opened
and the search results thrown away.

Anything not on the operator's allow-list is refused, whatever you put in
`targets`. If a task needs a site that is not allowed, say so in the output
instead of routing around it.

## Write actions in the tool's own words

Actions are compiled, not interpreted loosely. An unrecognised action fails
the step rather than being skipped, so use these forms:

- `click Sign in` / `double-click README.md` / `right-click the logo`
- `type 'quarterly report' into Search and submit`
- `fill the form with Name='Ada Lovelace', Email='ada@example.com'`
- `select 'India' from Country` · `check Remember me` · `clear Search`
- `scroll to the bottom` · `press Tab` · `hover Help`
- `wait for 'Results'` · `wait until 'Loading' is gone` · `wait 3 seconds`
- `find 'total cost'` — searches the page without dumping all of it
- `new tab https://example.com/pricing` · `switch to tab 2` · `close tab`
- `screenshot` · `full-page screenshot` · `read the console` · `read the network requests`

Label controls by the text a person would read on them. Never invent a CSS
selector or a `ref=` handle: the tool resolves labels itself, and a label it
cannot find uniquely is reported rather than guessed at.

## Waiting beats retrying

Content that arrives after the page does needs `wait for '<text>'`, not a
second navigation. Re-navigating throws away the state you just built —
including anything you typed.

## Finish the job

- Reading a page is not producing a deliverable. If the task asks for a file,
  a summary or a PDF, a later step must actually write it.
- Report what the page said, with the URL you read it from. An answer with no
  source is indistinguishable from an invented one.
- If the goal could not be reached, say which action failed and what the page
  showed instead. A truthful failure is worth more than a confident summary
  of a page you never got to.

## Never type a credential

Passwords, card numbers, API keys and one-time codes are refused before they
are typed — and before they are written to the action log. If a step needs a
sign-in, stop and ask the person to complete it themselves.
