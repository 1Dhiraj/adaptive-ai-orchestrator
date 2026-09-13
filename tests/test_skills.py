"""Skills: reference material injected into an agent's prompt where relevant."""

from __future__ import annotations

import pytest

from orchestrator.agents import Agent
from orchestrator.models import Step
from orchestrator.skills import (
    DEFAULT_SKILLS_DIR,
    Skill,
    SkillError,
    SkillLibrary,
    load_skill_file,
    parse_skill,
)


def step(role="backend", description="Do the thing", **kwargs) -> Step:
    return Step(id="s", description=description, agent_role=role, **kwargs)


class TestParsing:
    def test_frontmatter_and_body_are_split(self):
        skill = parse_skill(
            "---\nname: x\ndescription: d\nroles: backend, devops\n"
            "keywords: a, b\n---\n\n# Body\ncontent here")
        assert skill.name == "x" and skill.description == "d"
        assert skill.roles == ["backend", "devops"]
        assert skill.keywords == ["a", "b"]
        assert skill.content == "# Body\ncontent here"

    def test_missing_frontmatter_is_treated_as_pure_body(self):
        skill = parse_skill("just some content, no frontmatter", default_name="fallback")
        assert skill.name == "fallback"
        assert skill.content == "just some content, no frontmatter"

    def test_empty_body_is_an_error(self):
        with pytest.raises(SkillError, match="no content"):
            parse_skill("---\nname: x\n---\n")

    def test_default_name_used_when_frontmatter_omits_it(self):
        skill = parse_skill("---\ndescription: d\n---\nbody", default_name="from-filename")
        assert skill.name == "from-filename"

    def test_quoted_scalar_values_are_unquoted(self):
        skill = parse_skill('---\nname: "quoted"\n---\nbody')
        assert skill.name == "quoted"

    def test_blank_and_comment_lines_in_frontmatter_are_ignored(self):
        skill = parse_skill("---\n# a comment\n\nname: x\n---\nbody")
        assert skill.name == "x"

    def test_load_skill_file_uses_the_stem_as_default_name(self, tmp_path):
        path = tmp_path / "my-skill.md"
        path.write_text("no frontmatter here", encoding="utf-8")
        skill = load_skill_file(path)
        assert skill.name == "my-skill"
        assert skill.source == str(path)


class TestMatching:
    def test_role_match(self):
        skill = Skill(name="s", roles=["backend"], content="x")
        assert skill.applies_to(step(role="backend"))
        assert not skill.applies_to(step(role="frontend"))

    def test_keyword_match_is_case_insensitive(self):
        skill = Skill(name="s", keywords=["incident"], content="x")
        assert skill.applies_to(step(description="Write the INCIDENT report"))
        assert not skill.applies_to(step(description="Write the changelog"))

    def test_custom_role_matches_exactly_even_when_normalise_would_flatten_it(self):
        """'medical_writer' must match its own skill, not fall through to 'writer' rules."""
        skill = Skill(name="s", roles=["medical_writer"], content="x")
        assert skill.applies_to(step(role="medical_writer"))
        assert not skill.applies_to(step(role="writer"))

    def test_untargeted_skill_matches_everything(self):
        skill = Skill(name="s", content="applies broadly")
        assert skill.applies_to(step(role="anything", description="anything at all"))

    def test_targeted_skill_does_not_match_by_default(self):
        skill = Skill(name="s", roles=["backend"], content="x")
        assert not skill.applies_to(step(role="frontend", description="unrelated"))

    def test_either_role_or_keyword_is_sufficient(self):
        skill = Skill(name="s", roles=["backend"], keywords=["incident"], content="x")
        assert skill.applies_to(step(role="backend", description="unrelated"))
        assert skill.applies_to(step(role="writer", description="an incident happened"))
        assert not skill.applies_to(step(role="writer", description="unrelated"))


class TestRendering:
    def test_render_includes_name_and_content(self):
        skill = Skill(name="my-skill", content="the instructions")
        rendered = skill.render()
        assert "my-skill" in rendered and "the instructions" in rendered

    def test_render_truncates_long_content(self):
        skill = Skill(name="s", content="x" * 10000)
        rendered = skill.render(max_chars=100)
        assert len(rendered) < 200
        assert "truncated" in rendered


class TestLibrary:
    def test_load_dir_finds_every_markdown_file(self, tmp_path):
        (tmp_path / "a.md").write_text("---\nname: a\n---\nbody a", encoding="utf-8")
        (tmp_path / "b.md").write_text("---\nname: b\n---\nbody b", encoding="utf-8")
        (tmp_path / "ignore.txt").write_text("not a skill", encoding="utf-8")

        library = SkillLibrary()
        loaded = library.load_dir(str(tmp_path))
        assert sorted(loaded) == ["a", "b"]
        assert len(library) == 2

    def test_missing_directory_is_not_an_error(self, tmp_path):
        library = SkillLibrary()
        assert library.load_dir(str(tmp_path / "nope")) == []
        assert len(library) == 0

    def test_malformed_file_is_skipped_not_fatal(self, tmp_path, capsys):
        (tmp_path / "bad.md").write_text("---\nname: bad\n---\n", encoding="utf-8")  # empty body
        (tmp_path / "good.md").write_text("---\nname: good\n---\nfine", encoding="utf-8")

        library = SkillLibrary()
        loaded = library.load_dir(str(tmp_path))
        assert loaded == ["good"]
        assert "skipping" in capsys.readouterr().out

    def test_match_returns_only_relevant_skills(self):
        library = SkillLibrary([
            Skill(name="backend-only", roles=["backend"], content="x"),
            Skill(name="frontend-only", roles=["frontend"], content="y"),
        ])
        assert [s.name for s in library.match(step(role="backend"))] == ["backend-only"]

    def test_targeted_skills_rank_before_untargeted(self):
        library = SkillLibrary([
            Skill(name="general", content="applies to all"),
            Skill(name="specific", roles=["backend"], content="applies to backend"),
        ])
        matched = library.match(step(role="backend"))
        assert [s.name for s in matched] == ["specific", "general"]

    def test_render_for_joins_matched_skills(self):
        library = SkillLibrary([
            Skill(name="one", roles=["backend"], content="first"),
            Skill(name="two", roles=["backend"], content="second"),
        ])
        block = library.render_for(step(role="backend"))
        assert "first" in block and "second" in block

    def test_render_for_caps_the_number_of_skills(self):
        library = SkillLibrary([Skill(name=f"s{i}", content=f"c{i}") for i in range(10)])
        block = library.render_for(step(), max_skills=2)
        assert block.count("###") == 2

    def test_render_for_with_no_match_is_empty(self):
        library = SkillLibrary([Skill(name="s", roles=["frontend"], content="x")])
        assert library.render_for(step(role="backend")) == ""

    def test_add_replaces_same_named_skill(self):
        library = SkillLibrary([Skill(name="s", content="v1")])
        library.add(Skill(name="s", content="v2"))
        assert len(library) == 1
        assert library.get("s").content == "v2"

    def test_to_dict_never_includes_full_content(self):
        library = SkillLibrary([Skill(name="s", content="x" * 5000)])
        payload = library.to_dict()[0]
        assert "content" not in payload
        assert payload["length"] == 5000


class TestAgentIntegration:
    def test_prompt_includes_the_skills_block_when_given(self):
        agent = Agent("backend")
        prompt = agent.build_prompt(step(), "(none)", None, "### a-skill\ndo this")
        assert "Reference material" in prompt
        assert "do this" in prompt

    def test_prompt_omits_the_section_when_no_skills_apply(self):
        agent = Agent("backend")
        prompt = agent.build_prompt(step(), "(none)", None, "")
        assert "Reference material" not in prompt

    def test_execute_accepts_a_skills_block(self, stub_llm):
        agent = Agent("backend", llm=stub_llm)
        outcome = agent.execute(step(), "(none)", None, None, "### s\nfollow this")
        assert outcome.output  # ran without error


class TestWorkflowIntegration:
    def test_attach_skills_loads_and_reports(self, make_workflow, tmp_path):
        (tmp_path / "s.md").write_text("---\nname: s\nroles: backend\n---\nbody",
                                       encoding="utf-8")
        workflow = make_workflow([step()])
        assert workflow.attach_skills(str(tmp_path)) == ["s"]
        assert "s" in workflow.skills

    def test_missing_skills_dir_is_harmless(self, make_workflow, tmp_path):
        workflow = make_workflow([step()])
        assert workflow.attach_skills(str(tmp_path / "nope")) == []

    def test_skills_do_not_load_unless_asked(self, make_workflow):
        """No implicit directory scanning -- explicit like MCP config."""
        workflow = make_workflow([step()])
        assert len(workflow.skills) == 0

    def test_matched_skill_reaches_the_executed_prompt(self, make_workflow, tmp_path):
        (tmp_path / "s.md").write_text(
            "---\nname: convention\nroles: backend\n---\nALWAYS_USE_THIS_MARKER",
            encoding="utf-8")
        workflow = make_workflow(
            [Step(id="build", description="Build the API.", agent_role="backend")])
        workflow.attach_skills(str(tmp_path))

        captured = {}
        real_execute = workflow.agents.get("backend").execute

        def spy(step_, context, tools=None, gate=None, skills_block="",
                workflow_inputs=None):
            captured["block"] = skills_block
            return real_execute(step_, context, tools, gate, skills_block, workflow_inputs)

        workflow.agents.get("backend").execute = spy
        workflow.run_full()
        assert "ALWAYS_USE_THIS_MARKER" in captured["block"]

    def test_unrelated_step_gets_no_skills_block(self, make_workflow, tmp_path):
        (tmp_path / "s.md").write_text("---\nname: s\nroles: backend\n---\nbody",
                                       encoding="utf-8")
        workflow = make_workflow(
            [Step(id="research", description="Look into something.", agent_role="research")])
        workflow.attach_skills(str(tmp_path))
        assert workflow.skills.render_for(workflow.graph.get("research")) == ""

    def test_shipped_example_skills_all_parse(self):
        library = SkillLibrary()
        loaded = library.load_dir(DEFAULT_SKILLS_DIR)
        assert {"api-design-conventions", "incident-postmortem", "writing-style"} <= set(loaded)

    def test_shipped_skills_match_their_intended_roles(self):
        library = SkillLibrary()
        library.load_dir(DEFAULT_SKILLS_DIR)
        assert "api-design-conventions" in {
            s.name for s in library.match(step(role="backend", description="Build an endpoint"))}
        assert "writing-style" in {
            s.name for s in library.match(step(role="writer", description="anything"))}
