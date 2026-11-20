# CLAUDE.md — Project Rules

## Versioning

Version is in `tortoise_monitor/config.yaml` (`version: "x.y.z"`). The training
server's code (`tortoise_monitor/worker/`) has no version of its own: the
add-on serves it stamped with the add-on's version, and servers update from
the add-on when it changes.

| Change type | Bump |
|---|---|
| Bug fix | patch (`0.0.x`) |
| Substantial feature change/rework | minor (`0.x.0`) |
| New feature added | major (`x.0.0`) |

### Every version bump requires all three:

1. **Update `version` in `tortoise_monitor/config.yaml`**
2. **Prepend a `## x.y.z` entry to `tortoise_monitor/CHANGELOG.md`** — reverse chronological, plain bullet points describing changes
3. **Add a git tag**: `git tag vx.y.z && git push origin vx.y.z`

The HA Supervisor reads `CHANGELOG.md` (alongside `config.yaml`) when showing add-on update notes in the UI. Format must be `## x.y.z` headers with bullet points under each.
