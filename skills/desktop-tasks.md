---
name: desktop-tasks
description: How to drive a real Windows application through computer_use without acting on the wrong window
keywords: desktop, notepad, wordpad, window, application, keyboard, type into, screenshot, clipboard, save file, open app, launch
---

# Driving a desktop application

Desktop control is the sharpest tool here: keystrokes and clicks land wherever
the focus is, on the person's own machine, while they may still be using it.
Everything below exists because of that.

## Always name the window

Every action carries the window it is meant for. The tool raises that window
and re-checks what is actually in front before it types or clicks, so a step
allowed to drive Notepad cannot type into whatever stole focus a moment
earlier.

```
TOOL_DIRECTIVE: {"arguments": {
  "goal": "write the summary into a new Notepad document and save it",
  "targets": ["notepad"],
  "actions": ["launch notepad", "type 'Quarterly summary'", "save file"]}}
```

Omitting the window does not make the step more flexible. It makes it refuse,
because the tool will not act on an unknown foreground.

## Prefer description over coordinates

`click the Save button` asks the tool to look at the screen and find it.
`click at (742,318)` is a guess that silently clicks the wrong thing when the
window moves, the resolution differs, or the layout changes. Use coordinates
only when you were given them, or when a `locate` in this same run returned
them.

## The vocabulary

- `launch notepad` · `focus Notepad` · `list the windows`
- `screenshot` · `locate the Save button` · `click the Save button`
- `double-click the file` · `right-click the logo` · `click at (400,300)`
- `type 'hello'` · `press ctrl+s` · `select all` · `copy` · `paste`
- `read the window` — select-all + copy, so you can check what is actually
  there; the person's clipboard is put back afterwards
- `read the clipboard` · `put 'text' on the clipboard`
- `scroll down` · `move to (x,y)` · `drag (x,y) to (x,y)`
- `maximise the window` · `minimise the window` · `close the window`
- `wait 2 seconds`

An unrecognised action fails the step rather than being skipped.

## Use autonomous mode when the app will surprise you

Dialogs, "do you want to save changes?", first-run wizards and modal prompts
cannot be scripted in advance. Give the goal and set `"autonomous": true`; the
tool then lists the windows after each action and picks the next step from
what is really on screen.

## Verify before you claim

Typing is asynchronous — the application consumes injected keystrokes at its
own pace. Before saving or closing, `read the window` and confirm the text
arrived. "Typed 240 characters" means the events were accepted by Windows,
not that the document contains them.

## Prefer the shell when there is one

If the work can be done with a command — creating a file, converting a
document, counting things — use `computer_task` instead. It needs no window
and no focus, so it runs while the person carries on working, and it does not
depend on where a button happens to sit this week. Reach for desktop control
when the software has no interface but its GUI.

## Never type a credential

Passwords, card numbers, API keys and one-time codes are refused before they
are typed. If an application needs a sign-in, stop and ask the person.
