# Ethics, safety, and privacy

Serein deals with children's health. These rules are not optional.

## 1. Not a medical device
- Serein forecasts air quality and *estimates* risk. It does not diagnose or
  treat. Every message says to follow the child's asthma action plan from their
  doctor, and lists emergency signs (call 911).
- Personal plans can only be **stricter** than public EPA guidance, never looser
  (`serein/personal/action_plan.py`, tested in `tests/test_personal.py`).

## 2. No real child data without consent
- Development and the science-fair results use **synthetic children only**.
- If we ever run a pilot with real families: written parental permission and
  child assent, an adult supervisor (teacher/mentor), ideally review by a school
  or hospital ethics board (IRB), no names (random IDs), data stored only on the
  team's devices, and deleted after the project. Families can quit anytime.
- Real logs must never be committed to git (`.gitignore` blocks
  `**/participants/` and `**/real_logs/`).

## 3. Translations
- Health instructions are never machine-translated without review by a fluent
  speaker AND a health professional. Each message file has a `review_status`.

## 4. Fairness
- Low-cost sensors are sparser in lower-income neighborhoods. We report forecast
  accuracy separately for sensor-sparse areas (TODO B-9) and say so if it's worse.
- Smoke affects outdoor workers' kids, renters without air purifiers, and homes
  without AC most. Action plans include low-cost options (a box-fan filter, a
  "clean room") alongside HEPA purifiers (TODO C-8).

## 5. Honest results
- We pre-register the backtest (`docs/BACKTEST_PROTOCOL.md`) and report every
  metric, including losses.
- We clearly label simulated results as simulated.

## 6. Secrets
- API keys live only in `.env` (git-ignored). Never paste keys in code, issues,
  or chat.
