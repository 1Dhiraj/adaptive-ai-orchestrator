"""Skills: procedures, not capabilities.

Tools, MCP servers and API connections give agents new *things they can do*.
A skill gives an agent already capable of the task a better way to do it --
your team's incident postmortem format, your API design conventions, your
house style for release notes. It is reference material injected into the
prompt, the same idea as a Claude Code ``SKILL.md``.

A skill is a markdown file with a small frontmatter block:

    ---
    name: incident-postmortem
    description: Our blameless postmortem format
    roles: devops, reviewer
    keywords: incident, outage, postmortem
    ---

    # Postmortem format
    1. Timeline (UTC timestamps)
    2. Impact ...

Skills apply themselves: a skill matches a step when the step's role is
listed, or one of its keywords appears in the step's description. No wiring
per step is needed -- drop a file in ``skills/`` and it starts applying
wherever it's relevant.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, TYPE_CHECKING

if TYPE_CHECKING:  # pragma: no cover
    from .models import Step

DEFAULT_SKILLS_DIR = "skills"

_FRONTMATTER_RE = re.compile(r"\A---\s*\n(.*?)\n---\s*\n?", re.DOTALL)


class SkillError(ValueError):
    """A skill file could not be parsed."""


@dataclass
class Skill:
    """One piece of reference material an agent can be handed."""

    name: str
    description: str = ""
    #: Agent roles this applies to. Empty means "match by keyword only".
    roles: List[str] = field(default_factory=list)
    #: Substrings matched (case-insensitively) against a step's description.
    keywords: List[str] = field(default_factory=list)
    #: The instructions themselves -- what gets injected into the prompt.
    content: str = ""
    #: Where this was loaded from, for diagnostics.
    source: Optional[str] = None

    def applies_to(self, step: "Step") -> bool:
        from .agents import _slug_role

        if self.roles:
            # Exact slug comparison, deliberately NOT normalise_role: that
            # does fuzzy substring matching meant for picking a fallback
            # agent, and it would match 'medical_writer' to the built-in
            # 'writer' -- exactly the collision a skill author writing
            # `roles: medical_writer` means to avoid.
            step_role = _slug_role(step.agent_role)
            declared_roles = {_slug_role(r) for r in self.roles}
            if step_role in declared_roles:
                return True
        if self.keywords:
            haystack = f"{step.description} {step.name}".lower()
            if any(kw.lower() in haystack for kw in self.keywords):
                return True
        return not self.roles and not self.keywords  # untargeted -> always applies

    def render(self, max_chars: int = 4000) -> str:
        """The block injected into an agent's prompt."""
        body = self.content.strip()
        if len(body) > max_chars:
            body = body[:max_chars].rstrip() + "\n...[truncated]"
        return f"### {self.name}\n{body}"

    def to_dict(self) -> Dict[str, Any]:
        return {
            "name": self.name, "description": self.description,
            "roles": self.roles, "keywords": self.keywords,
            "source": self.source, "length": len(self.content),
        }


def _parse_frontmatter(text: str) -> tuple:
    """Split ``---\\nkey: value\\n---\\nbody`` into (fields, body).

    A deliberately small parser -- no PyYAML dependency -- handling only
    what a skill needs: scalar strings and comma-separated lists.
    """
    match = _FRONTMATTER_RE.match(text)
    if not match:
        return {}, text.strip()

    fields: Dict[str, Any] = {}
    for line in match.group(1).splitlines():
        line = line.strip()
        if not line or line.startswith("#") or ":" not in line:
            continue
        key, _, value = line.partition(":")
        key, value = key.strip().lower(), value.strip()
        if key in {"roles", "keywords"}:
            fields[key] = [v.strip() for v in value.split(",") if v.strip()]
        else:
            fields[key] = value.strip("'\"")

    body = text[match.end():].strip()
    return fields, body


def parse_skill(text: str, default_name: str = "skill", source: Optional[str] = None) -> Skill:
    fields, body = _parse_frontmatter(text)
    if not body:
        raise SkillError(f"{source or default_name}: skill has no content")
    return Skill(
        name=str(fields.get("name") or default_name),
        description=str(fields.get("description", "")),
        roles=list(fields.get("roles", [])),
        keywords=list(fields.get("keywords", [])),
        content=body,
        source=source,
    )


def load_skill_file(path: Path) -> Skill:
    text = path.read_text(encoding="utf-8")
    return parse_skill(text, default_name=path.stem, source=str(path))


# ---------------------------------------------------------------------------
# Library
# ---------------------------------------------------------------------------


class SkillLibrary:
    """A collection of skills, matched against steps as they run."""

    def __init__(self, skills: Optional[List[Skill]] = None):
        self._skills: Dict[str, Skill] = {}
        for skill in skills or []:
            self.add(skill)

    def add(self, skill: Skill) -> "SkillLibrary":
        self._skills[skill.name] = skill
        return self

    def remove(self, name: str) -> None:
        self._skills.pop(name, None)

    def get(self, name: str) -> Optional[Skill]:
        return self._skills.get(name)

    def __len__(self) -> int:
        return len(self._skills)

    def __contains__(self, name: object) -> bool:
        return name in self._skills

    def __iter__(self):
        return iter(self._skills.values())

    @property
    def names(self) -> List[str]:
        return sorted(self._skills)

    def load_dir(self, path: str = DEFAULT_SKILLS_DIR) -> List[str]:
        """Load every ``*.md`` file in a directory. Missing dir -> no-op.

        A malformed file is skipped with a printed warning rather than
        aborting the whole load, matching the project's degrade-not-crash
        posture for optional configuration.
        """
        directory = Path(path)
        if not directory.is_dir():
            return []

        loaded: List[str] = []
        for file_path in sorted(directory.glob("*.md")):
            try:
                skill = load_skill_file(file_path)
            except (SkillError, OSError) as exc:
                print(f"[skills] skipping '{file_path}': {exc}")
                continue
            self.add(skill)
            loaded.append(skill.name)
        return loaded

    def match(self, step: "Step") -> List[Skill]:
        """Every skill relevant to this step, most specific first."""
        matched = [s for s in self._skills.values() if s.applies_to(step)]
        # Skills that explicitly target this step's role/keywords are more
        # relevant than untargeted ones that match everything by default.
        matched.sort(key=lambda s: 0 if (s.roles or s.keywords) else 1)
        return matched

    def render_for(self, step: "Step", max_skills: int = 3, max_chars: int = 4000) -> str:
        """The reference-material block to inject into this step's prompt."""
        matched = self.match(step)[:max_skills]
        if not matched:
            return ""
        return "\n\n".join(skill.render(max_chars) for skill in matched)

    def to_dict(self) -> List[Dict[str, Any]]:
        return [s.to_dict() for s in sorted(self._skills.values(), key=lambda s: s.name)]
