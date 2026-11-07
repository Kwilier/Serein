# Drop-in drafts for the Serein repo (2026-11-07)

Everything here mirrors the repo's folder layout. Copy the `docs/` folder over the
repo's `docs/` (nothing is overwritten except that `innovation_brief_q6-q10_DRAFT.md`
is new; merge it into `innovation_brief.md` by hand after team review). Then open a
PR titled "E-4/E-5/S-10 drafts: market, competition, business model, references" and
update the task board rows below.

| File | Repo task | What it is | Status |
|---|---|---|---|
| `docs/drafts/innovation_brief_q6-q10_DRAFT.md` | S-5 (Q6-Q10) | Market, Competition, Go-to-market, Business model, Fundraising, all within word limits, every number sourced | Draft; 4 `[TEAM: ...]` slots left |
| `docs/COMPETITOR_VERIFICATION_2026-11-07.md` | E-5 | Verified competitor table, AirRater deep-dive, research-models paragraph, novelty verdict, 12 claims to soften | Done; E-5 can move to Done after one teammate re-checks |
| `docs/BUSINESS_NUMBERS_2026-11-07.md` | E-4 | Every number behind Q6/Q9/Q10 with URL and year; price-range reasoning; break-even | Done; E-4 still needs interviews to confirm buyer/price |
| `docs/interviews/INTERVIEW_GUIDE.md` | E-1 | Parent / nurse / clinician question sets, outreach text, notes template, rules | Ready to use |
| `docs/drafts/clinician_review_request.md` | C-8 | Email + one-page review sheet; a 20-minute ask | Ready to send |
| `docs/drafts/letter_of_support_template.md` | E-7 | One-page letter they can edit and sign | Ready |
| `docs/drafts/video_script.md` | S-9, S-8 | 4-minute storyboard, one-sentence pitch, shot list, website hero | Ready; swap in final numbers after D-5 |
| `docs/drafts/ip_decision_memo.md` | E-8, Q4 IP | Options, grace-period facts, provisional cost, **a live SEREIN trademark conflict in classes 9/42/44**, draft Q4 IP text | Decide at next team meeting |
| `docs/drafts/references_DRAFT.md` | S-10 | 47 formatted references for the attachment | Draft; merge with METHODS.md/DATA.md citations |

## Four things the brief now says that it did not before, and why

1. **We name our closest competitors instead of hoping judges don't find them.** Google/BreezoMeter (500 m, 96 h fused forecast) at the forecast layer; AirRater (Australia, per-user trigger learning since 2015) at the personalization layer. The rubric's "Verification" descriptor says judges Google for duplicates. Saying it first turns a weakness into credibility.
2. **"First" claims are narrowed to ones we can defend**: first family-facing US product that publishes calibrated uncertainty and held-out head-to-head scores and joins the forecast to a per-child plan. See the 12-item list in the competitor file.
3. **Pricing is anchored to real contracts** ($2,095/building Raptor; $69,913 ParentSquare; $26.66 Medi-Cal unit), not invented. The break-even is honest: ED-only savings support ~$0.50-$2/child; hospitalizations and non-hospital value justify $10-$20.
4. **A persistence baseline must be in the final scoring.** An MIT paper (BAMS 2026) found morning AirNow persistence matched or beat every operational smoke model for day-level decisions. The fallback baseline includes persistence only as a last resort; report plain persistence as its own row in D-5. If a judge knows that paper and we don't have the row, we lose the Practicality argument.

## Open items only a human can do (unchanged, in priority order)

1. **S-1** Confirm in the portal that the Lean Canvas was submitted Oct 29 and which category. Everything downstream depends on it.
2. **C-8** Send `clinician_review_request.md` to one clinician this week.
3. **E-1** Run interviews using the guide; aim for 5 parents, 2 nurses, 1 clinician by Dec 5.
4. **D-4/D-5/D-6b** Freeze the protocol (commit it), rerun the dev suite after the pipeline fixes, then run the final 2024-25 test and transfer test once. Add plain persistence as a baseline row first.
5. **S-3** Team section (names, grades, roles, coach) in the brief and website.
6. **E-8** Decide IP at the next meeting; check USPTO TSDR for serial 90636402 before spending $350 on a trademark.
7. **S-11** Pay the $499 or apply for financial aid (window closes Dec 3, 2026).
8. **S-9** Film the video after D-5 so the numbers are final.
9. Attend the Nov 12 Conrad workshop ("Protect Your Idea") and Dec 10 ("Communicating Your Idea").

## Things I checked and did not change

- Dev results, protocol, model code: untouched. Only the business/pitch side.
- The `[TEAM: ...]` convention, "no em dashes, no hype words" rule, and "numbers must come from a file" rule are followed in all drafts.
