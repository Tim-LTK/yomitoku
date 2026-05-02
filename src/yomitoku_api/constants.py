"""Project-wide constants. Prompts reference STUDENT_CONTEXT by name — do not duplicate ad-hoc."""

_STUDENT_CONTEXT_LINES = (
    "Student: Timothy Lam Tsz Ki, 40, Hong Kong/Singapore",
    "L1s: Cantonese, Mandarin (Traditional Chinese), English",
    "Level: Upper N4 / Early N3",
    "Kanji: not a bottleneck — Traditional Chinese covers most N1 kanji",
    (
        "Gaps: hiragana-dense grammar segmentation, verb conjugation "
        "(う/る distinction, irregulars), listening"
    ),
    "N5 grammar: は が を に で へ, て-form, ～たい, ～たことがある",
    "N4 grammar: ～てしまう, ～ために, ～ながら, ～たら",
    "Next: ～てもいい, ～てはいけない",
    "Goal: N2 → Japan relocation via internal transfer at Japanese MNC",
)

STUDENT_CONTEXT = "\n".join(_STUDENT_CONTEXT_LINES).strip()

DEFAULT_ANTHROPIC_MODEL = "claude-sonnet-4-20250514"
