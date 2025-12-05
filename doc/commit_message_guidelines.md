# Commit Message Guidelines

Complete Template:
```
<type>(<scope>): [<disclaimer>] <summary>

<body>

<footer>
```

## Type (mandatory)

Possible types:
* feat: a new feature
* fix: a bug fix
* docs: documentation only
* refactor: code change that doesn’t fix a bug or add a feature
* perf: performance improvement
* test: adding/fixing tests
* chore: maintenance (deps, scripts, configs)
* style: formatting (no logic changes)

## Scope (optional)

A **scope** specifies what part of the project the commit affects.  
It helps readers quickly understand the context without reading the whole diff.

Examples: `auth`, `api`, `ui`, `database`, `tests`.

## Disclaimer (if applicable)

If a commit is not expected to run successfully, add a note in brackets to this effect.

Examples: `WIP` (work in progress), `untested`.

## Summary (mandatory)

* Keep it under 50 characters if possible.
* Use the imperative mood: "Add feature", not "Added feature" or "Adds feature".
* Don’t end with a period.
* Be specific and concise.

Example: "Correct rounding error in price calculation"

## Body (recommended)

* Use the body to explain what changed and why.
* Wrap lines at ~72 characters for readability.
* Bullet list are allowed and advised.
* Mention limitations or follow-up work.
* If the change is obvious, the body can be omitted.
* Keep it concise to improve readability.

## Footer (optional)

* Issue tracking references can be added (e.g, "Fixes #42").
* Mention breaking changes (e.g. "BREAKING CHANGE: Rename `id` to `user_id` in API")
