#!/usr/bin/env sh
#
# skill.sh -- work with the orchestrator's skill library from a shell.
#
# Skills are markdown files in skills/ that get injected into an agent's
# prompt when they match the step it is running. Because matching is implicit
# -- by role and by keyword, never wired up per step -- the two questions that
# actually come up are "is this file valid?" and "which steps will this fire
# on?". Both are answered here, so neither needs a Python session.
#
#   ./skill.sh list                      every skill, with what it matches on
#   ./skill.sh show <name>               one skill in full
#   ./skill.sh validate [dir]            parse every file; non-zero if any fail
#   ./skill.sh match "<description>"     which skills a step would receive
#   ./skill.sh render "<description>"    the exact prompt block it would get
#   ./skill.sh new <name>                scaffold a new skill file
#   ./skill.sh test                      run the skill test suite
#
# Runs under Git Bash, WSL, macOS and Linux. Windows users without a shell can
# call the same library directly: python -c "from orchestrator.skills import ..."

set -eu

here=$(CDPATH= cd -- "$(dirname -- "$0")" && pwd)
cd "$here"

SKILLS_DIR=${SKILLS_DIR:-skills}
PYTHON=${PYTHON:-python}

if ! command -v "$PYTHON" >/dev/null 2>&1; then
    if command -v python3 >/dev/null 2>&1; then
        PYTHON=python3
    else
        echo "skill.sh: no python on PATH (set PYTHON=/path/to/python)" >&2
        exit 127
    fi
fi

usage() {
    sed -n '3,22p' "$0" | sed 's/^# \{0,1\}//'
}

# One helper for every subcommand that needs the library: keeping the Python
# in a single place means the shell never has to know how a skill is parsed.
run_py() {
    SKILL_DIR="$SKILLS_DIR" SKILL_ARG="${1:-}" "$PYTHON" - "$2" <<'PYTHON'
import os
import sys
from pathlib import Path

sys.path.insert(0, os.getcwd())

from orchestrator.models import Step
from orchestrator.skills import SkillError, SkillLibrary, load_skill_file

mode = sys.argv[1]
argument = os.environ.get("SKILL_ARG", "")
directory = Path(os.environ.get("SKILL_DIR", "skills"))

if not directory.is_dir():
    print(f"no skills directory at {directory}/", file=sys.stderr)
    raise SystemExit(1)


def probe(description: str) -> Step:
    """A step that exists only to be matched against."""
    return Step(id="probe", name=description[:60], description=description,
                agent_role=os.environ.get("SKILL_ROLE", "") or "generalist")


if mode == "list":
    library = SkillLibrary()
    library.load_dir(str(directory))
    if not len(library):
        print(f"no skills in {directory}/")
        raise SystemExit(0)
    for skill in sorted(library, key=lambda s: s.name):
        matches = []
        if skill.roles:
            matches.append("roles: " + ", ".join(skill.roles))
        if skill.keywords:
            matches.append("keywords: " + ", ".join(skill.keywords[:6])
                           + (" ..." if len(skill.keywords) > 6 else ""))
        print(f"{skill.name}  ({len(skill.content)} chars)")
        if skill.description:
            print(f"    {skill.description}")
        for line in matches:
            print(f"    {line}")
        if not matches:
            print("    applies to every step (no roles or keywords declared)")

elif mode == "show":
    path = directory / f"{argument}.md"
    if not path.is_file():
        matched = [p for p in sorted(directory.glob("*.md")) if argument in p.stem]
        if len(matched) != 1:
            print(f"no skill named {argument!r} in {directory}/", file=sys.stderr)
            raise SystemExit(1)
        path = matched[0]
    skill = load_skill_file(path)
    print(f"# {skill.name}  <- {skill.source}")
    if skill.description:
        print(f"# {skill.description}")
    print()
    print(skill.content)

elif mode == "validate":
    failures = 0
    files = sorted(directory.glob("*.md"))
    if not files:
        print(f"no skills in {directory}/")
    for path in files:
        try:
            skill = load_skill_file(path)
        except (SkillError, OSError) as exc:
            print(f"FAIL {path}: {exc}")
            failures += 1
            continue
        problems = []
        if not skill.description:
            problems.append("no description (the library lists it blank)")
        if not skill.roles and not skill.keywords:
            problems.append("no roles or keywords: it will apply to EVERY step")
        if len(skill.content) > 4000:
            problems.append(
                f"{len(skill.content)} chars: truncated to 4000 when injected")
        if problems:
            print(f"WARN {path.name}: " + "; ".join(problems))
        else:
            print(f"ok   {path.name}")
    if failures:
        print(f"\n{failures} skill(s) could not be parsed", file=sys.stderr)
        raise SystemExit(1)

elif mode in {"match", "render"}:
    if not argument:
        print("give a step description to match against", file=sys.stderr)
        raise SystemExit(2)
    library = SkillLibrary()
    library.load_dir(str(directory))
    step = probe(argument)
    matched = library.match(step)
    if mode == "match":
        if not matched:
            print(f"no skill matches: {argument}")
        for skill in matched:
            why = "role" if skill.roles and not skill.keywords else "keyword"
            print(f"{skill.name}  (matched by {why})")
    else:
        block = library.render_for(step)
        print(block or f"(no skills would be injected for: {argument})")

elif mode == "new":
    if not argument:
        print("give the new skill a name", file=sys.stderr)
        raise SystemExit(2)
    path = directory / f"{argument}.md"
    if path.exists():
        print(f"{path} already exists", file=sys.stderr)
        raise SystemExit(1)
    path.write_text(
        f"---\nname: {argument}\ndescription: one line on what this teaches\n"
        "roles: \nkeywords: \n---\n\n"
        f"# {argument.replace('-', ' ').title()}\n\n"
        "Write the procedure here. Declare roles or keywords above, or this\n"
        "will be injected into every step.\n",
        encoding="utf-8")
    print(f"created {path}")

else:
    print(f"unknown mode {mode!r}", file=sys.stderr)
    raise SystemExit(2)
PYTHON
}

command=${1:-help}
[ $# -gt 0 ] && shift

case "$command" in
    list)      run_py "" list ;;
    show)      run_py "${1:-}" show ;;
    validate)  [ $# -gt 0 ] && SKILLS_DIR=$1; run_py "" validate ;;
    match)     run_py "${1:-}" match ;;
    render)    run_py "${1:-}" render ;;
    new)       run_py "${1:-}" new ;;
    test)      exec "$PYTHON" -m pytest tests/test_skills.py -q ;;
    help|-h|--help) usage ;;
    *)
        echo "skill.sh: unknown command '$command'" >&2
        echo >&2
        usage >&2
        exit 2
        ;;
esac
