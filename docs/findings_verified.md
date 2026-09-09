# Behavioural verification of new glitch-token candidates

Indicator scores are not evidence. This is the test.

## Protocol
Magikarp's repetition criterion (does the model reproduce the token when shown it?),
three prompt templates (few-shot copy, quoted-repeat, Q&A), greedy decoding, run
identically over three groups:

- **KNOWN** -- Magikarp's behaviourally verified under-trained tokens. Positive
  control: if these do not fail, the test is broken.
- **CANDIDATE** -- 25 tokens our detector ranks highest that Magikarp categorised
  `OK` and never verified.
- **CONTROL** -- length-matched ordinary tokens. Negative control: if these fail,
  we are measuring model incompetence rather than glitchiness.

Run on **two independent models**, pythia-1.4b and pythia-2.8b. "Fails" = fails at
least 2 of the 3 templates.

## Result

| group | fails 1.4b | fails 2.8b | fails BOTH |
|---|---|---|---|
| KNOWN (magikarp-verified) | 14/15 | 14/15 | **14/15 (93%)** |
| CANDIDATE (ours, new) | 18/25 | 15/25 | **13/25 (52%)** |
| CONTROL (matched normal) | 3/25 | 1/25 | **1/25 (4%)** |

Both controls behave correctly, so the instrument is valid. Against a 4% base
rate, our candidates confirm at 52% -- **13x enrichment**.

## The 13 confirmed on both models

`'imonit'`, `'ÃÂÃÂÃÂÃÂ'`, `' {¶'`, `'marined'`, `' $[]$'`, `' careg'`,
`'medsc'`, `'1451450014514500'`, `' \xa0\xa0 \xa0\xa0 \xa0\xa0 \xa0\xa0'`,
`' tradem'`, `'\xa0\n '`, `'idemargin'`, `' counc'`

Recognisable families: mojibake (double-encoded UTF-8), truncated word fragments,
non-breaking-space runs, and a repeated numeric string.

## Precision is 52%, not 100%

Twelve candidates did **not** confirm: `' practition'`, `' resemb'`,
`' earthqu'`, `' glimp'`, `' teasp'`, `' remn'`, `'ICENSE'`, `' specim'`,
`'rsfs'`, and three mojibake variants. Rare word fragments are not automatically
glitch tokens -- the model copies plenty of them fine. Roughly half of what the
detector proposes is a false positive, and that is the honest headline number.

One curiosity worth noting: within the mojibake family, `'ÃÂÃÂÃÂÃÂ'` (4x)
confirms while `'ÃÂ'` (1x) and `'ÃÂÃÂÃÂÃÂÃÂÃÂÃÂÃÂ'` (8x) do not. Repetition
length matters in a way a pure frequency account does not explain.

## What this establishes
The detector finds **genuinely new** glitch tokens -- verified behaviourally, on
two models, against published prior work that missed them. It does so at 52%
precision, which makes it a candidate generator for human/behavioural review
rather than a finished oracle.

## Example generations (the qualitative evidence)
The classic signature, reproduced exactly:

    'ecause'    -> ' because'      (KNOWN)
    'ccording'  -> ' according'    (KNOWN)
    'icrosoft'  -> ' Microsoft'    (KNOWN)
    ' careg'    -> ' cat'          (ours)
    'ÃÂÃÂ'      -> ' quick brown fox'  (ours -- copies the wrong demo entirely)
    'marined'   -> ' marine'       (ours)

The model cannot represent the fragment, so it emits the whole word it belongs to,
or falls back to unrelated prompt content.
