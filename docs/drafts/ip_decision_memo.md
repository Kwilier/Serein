# IP decision memo (task E-8), for team discussion

Written 2026-11-07 by a non-lawyer. Conrad's prize partners include ArentFox Schiff
pro-bono patent services for Scholar winners, and the Nov 12 Conrad virtual workshop
is "Business Insight: Protect Your Idea." Attend it, then decide. Nothing below is
legal advice; verify fees and status on uspto.gov.

## What the rules require

- Rules and Regulations: teams own their IP; **U.S. finalists "will be required to
  file a Provisional Patent application" before presenting at the Summit.** Summit
  disclosures are non-confidential.
  https://conrad.spacecenter.org/the-challenge/rules-and-regulations/
- Rubric: Innovation (30%) includes "IP Defensibility (patent, trade secret,
  first-mover, ecosystem capture)." Brief Q4 has an IP-protection sub-question.

## Facts that shape the decision

1. **The code is already public (MIT, GitHub) and has been since early November.**
   In the U.S., an inventor's own public disclosure starts a 12-month grace period
   (35 U.S.C. 102(b)(1)); a provisional filed within 12 months of the first public
   disclosure is not barred by that disclosure. Outside the U.S., most countries have
   no grace period, so foreign patent rights are likely already lost. For a
   student team this is acceptable; say it plainly if asked.
2. **Provisional patent cost: $65 (micro entity).** USPTO fee schedule, revised
   Nov 1, 2026. A provisional is a one-year placeholder; it needs a written
   description and drawings, not claims, and is never examined. It is required of
   finalists anyway, so the only real question is what to describe in it.
   https://www.uspto.gov/learning-and-resources/fees-and-payment/uspto-fee-schedule
3. **What could be described:** the specific combination (a) graph model over
   regulatory monitors that outputs a residual correction to an NWP smoke forecast
   with conformal-calibrated intervals, (b) issue-time data-availability enforcement,
   (c) a hierarchical per-child sensitivity model updated from a short caregiver
   check-in, with a rule that the output can only be stricter than a public
   guideline, and (d) the pipeline from (a) to (c) producing a zoned plan. Abstract
   ideas and pure math are not patentable; a concrete system with these components
   is at least describable. Whether it is *grantable* is a question for the
   ArentFox attorneys if we win; for the brief, "provisional filed on the combined
   system" is enough.
4. **The name "SEREIN" has a live conflict.** A SEREIN mark exists in Classes 9,
   42 and 44 for "downloadable computer software for diagnosing and treating
   psychological and related physiological comorbidities of severe acute and chronic
   diseases," published for opposition January 11, 2022 (serial 90636402).
   https://trademarks.justia.com/906/36/serein-90636402.html
   That is health software in the same classes we would file in. A USPTO examiner
   could refuse our application for likelihood of confusion. An earlier SEREiN mark
   in Class 25 (clothing) was abandoned in 2018 and is not a concern.
   https://trademarkdir.com/owner/serein
   **Action:** check the live status of serial 90636402 on USPTO TSDR
   (tsdr.uspto.gov). If it registered, talk to a mentor about (i) keeping "Serein"
   as a project name for the competition only, (ii) picking a product name we can
   own before the website and video are final, or (iii) filing anyway in Class 42
   with a narrow description and accepting refusal risk. A trademark application is
   $350 per class (electronic); do not spend it until this is resolved.

## Options

| Option | Pros | Cons |
|---|---|---|
| **A. Keep code open (MIT); file a $65 provisional on the combined system; protect the brand** | Health departments and judges can audit the method; matches the rubric's "ecosystem capture" and "first-mover" defensibility; required provisional covered; cheapest; honest about the public repo | Competitors can copy the code; patent scope is uncertain; name may need to change |
| B. Move trained models and AirPlan service to a private repo now; keep the pipeline public | Some trade-secret value in the trained weights and message logic | Weights are reproducible from public data by anyone with the code; splitting the repo mid-competition costs time and invites "what are you hiding?" questions |
| C. Take the whole repo private | Maximum secrecy | Loses the audit story, the open-science credibility, and the CI/test proof of student work; disclosure already happened, so secrecy is partly fictional |

**Recommendation to discuss: Option A.** The defensible moat for Serein is not the
code; it is (1) the validated, pre-registered backtest nobody else has published for
a family product, (2) partnerships with a district and a clinician, and (3) the data
flywheel from check-ins (which we never publish). Say exactly that in Brief Q4.

## Draft text for Brief Q4 "IP protection" (replace the [TEAM decision needed] block)

> Our forecasting code is open source (MIT) so that health departments and judges
> can audit it; we believe trust is worth more than secrecy for a product that
> advises parents about children. We will file a U.S. provisional patent application
> on the combined system (monitor-corrected neighborhood forecast with calibrated
> ranges, the issue-time availability rule, and the per-child sensitivity model with a
> stricter-than-guideline floor) before the Summit, as the rules require. Our
> durable advantages are the pre-registered head-to-head validation, which no
> consumer air-quality product publishes, our partnerships with [district / clinic],
> and the check-in data, which is never published. [TEAM: brand line once the
> trademark question is resolved.]

(About 95 words; fits inside Q4's 750-word budget.)
