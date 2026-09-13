---
name: incident-postmortem
description: Blameless postmortem format
roles: devops, reviewer
keywords: incident, outage, postmortem, downtime
---

# Postmortem format

Write postmortems that assume good intent and focus on systems, not people.

1. **Summary** — one paragraph: what broke, for how long, who was affected.
2. **Timeline** — UTC timestamps only. Detection time, mitigation time,
   resolution time, each with the concrete action taken.
3. **Impact** — quantified: error rate, affected user count, revenue if known.
4. **Root cause** — the technical cause, stated precisely. "Human error"
   is never a root cause; the missing safeguard that allowed the error is.
5. **What went well** — at least one item. Detection speed, a runbook that
   worked, a rollback that succeeded.
6. **Action items** — each with an owner and a due date, not "the team".
   Prioritise the item that would have prevented this incident entirely.

Never name an individual as the cause of an incident.
