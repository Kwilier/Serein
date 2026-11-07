# Payer and pilot-budget evidence

Checked November 7, 2026. These notes support E-4 and sections 6, 8, 9 and 10 of
the [Innovation Brief draft](drafts/innovation_brief.md). They establish a few
public counts and input prices. They do not establish a subscription price,
customer demand, clinical savings, or a complete operating budget. E-4 remains
open, and the brief's placeholders are unchanged.

## Separate the beneficiary from the buyer

The draft proposes free family access funded by an organization. A school
district is a plausible first interview category because it already decides how
schools respond to poor air quality. Los Angeles Unified describes using its
[Know Your Air network](https://oehs.lausd.org/apps/pages/KnowYourAirNetwork)
for school-level decisions during wildfire and pollution events. This is
evidence of an operational use case, not evidence that a district will buy
Serein, or a price comparison for a forecast-only service. The page does not
provide a contract price; its sensor deployment includes different services.

One reviewed arrangement expressly requires no monetary payment by the district.
Santa Ana Unified's [January 28, 2025 board minutes, item 13.12](https://go.boarddocs.com/ca/sausd/Board.nsf/BD-GetMinutes?open&id=DAZQYX6BA64B)
record approval of an air-monitoring research partnership with OneOC/GREEN-MPNA.
The [posted agreement, section 1.12](https://go.boarddocs.com/ca/sausd/Board.nsf/files/DCW2GV022BFD/$file/OneOC%20%26%20GREEN-MPNA%20No-Cost%20Agreement.pdf)
specifies no district compensation to the partner. Its scope includes sensor
installation, sampling and access to a real-time monitoring dashboard. This is
an example of a partnership rather than a paid forecast subscription. The stated
term runs from August 1, 2024 through July 30, 2026; current continuation and
actual delivery were not verified. No district payment does not mean zero
project cost, and this agreement does not establish Serein's price or demand.

There are existing public alternatives. AirNow's
[Air Quality Flag Program](https://www.airnow.gov/air-quality-flag-program/)
links activity guidance, its app and forecast alerts.
[EnviroFlash's description](https://m.enviroflash.info/about.cfm) says it delivers
local forecasts and alerts by email or mobile phone. A paid product needs to
demonstrate added value over these workflows. The sources do not establish that
schools lack warnings today or that an earlier warning changes their decisions.

School district, air district and health-plan sales should be treated as
different proposals. A school operations dashboard does not establish demand for
a child's symptom-based plan. A health-plan proposal needs payer-specific
evidence and reviewed outcomes before using avoided visits to justify a price.
No avoided-visit revenue or clinical saving is included below.

## Published population context

CDC's [Most Recent Asthma Data](https://www.cdc.gov/asthma-data/about/most-recent-asthma-data.html),
updated October 6, 2026, reports a weighted national estimate of 4,804,000 children
under 18 with current asthma from the 2023 National Health Interview Survey
(NHIS). Prevalence is 6.7%, with a standard error of 0.33 percentage points.
This supports the brief's approximate 4.8 million figure with its 2023 reference
year. It is not a 2026 census or a count of eligible or paying families.

The same page's child state table uses a different survey, the 2023 National
Survey of Children's Health (NSCH):

| State | Published estimated children under 18 with current asthma |
| --- | ---: |
| California | 548,709 |
| Oregon | 26,594 |
| Washington | 83,411 |

These state estimates sum to 658,714. Do not divide that sum by the NHIS national
estimate to claim a coherent regional share. Neither survey identifies how many
families want the product or live in supported, smoke-affected locations.
Only the published aggregate tables were used; no individual health records were
accessed or added.

## Count organizations using explicit definitions

| Geography | Published count | Scope and date | Source |
| --- | ---: | --- | --- |
| California | 935 school districts | Fiscal 2026-27; excludes county offices of education, charter schools, joint powers authorities and other organization types | [CDE district organization list](https://www.cde.ca.gov/re/lr/do/schooldistrictlist.asp), reviewed July 31, 2026 |
| Oregon | 197 school districts | The department's current Schools & Districts page; no reference year stated in its count paragraph | [Oregon Department of Education](https://www.oregon.gov/ode/schools-and-districts/pages/default.aspx) |
| Washington | 295 school districts | OSPI lists state-tribal compact schools and additional jurisdictions separately; no reference year stated beside the count | [OSPI About School Districts](https://ospi.k12.wa.us/about-ospi/about-school-districts) |
| California | 35 air districts | Regional planning, monitoring and permitting organizations; a different buyer category | [CARB California Air Districts](https://ww2.arb.ca.gov/california-air-districts) |

The school-district counts sum to 1,427. This is an inventory across differently
dated official sources, not a synchronized national dataset, a count of
smoke-affected districts, or a sales forecast. California elementary and high
school districts can cover overlapping geography. Do not multiply the counts by
an assumed common enrollment or add school counts to them.

CDE's separate [Fingertip Facts page](https://www.cde.ca.gov/ds/ad/ceffingertipfacts.asp)
lists 1,011 entities for 2025-26, including an "Other" category of 75. Its
district-type rows excluding that category sum to 936, matching the organization
list for 2025-26. The 935 figure belongs to 2026-27. Keep both the year and entity
definition when citing a denominator instead of treating these totals as
interchangeable.

A defensible bottom-up worksheet should start with a named, deduplicated list
of eligible organizations, then record who owns the decision, who owns the
budget, existing contracts, procurement timing, supported locations, and a
validated annual price. Until interviews and a scoped quote establish those
inputs, "eligible districts times price" is a scenario, not measured market size.

## Public cost inputs in USD

Prices were retrieved from the providers' pages with a fresh request on
November 7, 2026. They are public usage prices, not a negotiated Serein quote.
No service is selected or purchased by this document.

| Item | Public price read | What the number excludes or depends on |
| --- | --- | --- |
| U.S. long-code outbound SMS | $0.0083 per segment | Additional carrier fees, registration, numbers, failed-message fees, optional services and taxes |
| U.S. long-code outbound carrier fees | The table lists $0.0035 for AT&T, $0.0045 for T-Mobile, $0.005 for Verizon and US Cellular, and $0.0042 for other carriers | Actual mix of destination networks and any later pricing changes |
| Twilio leased U.S. long-code number | $1.15 per number per month | Message usage and registration fees |
| PurpleAir API points, minimum purchase tier | $10 minimum; 100,000 points per dollar for purchases from $10 to $49 | Point consumption depends on request shape; a point is not one sensor reading or one forecast |

Sources: [Twilio U.S. SMS pricing](https://www.twilio.com/en-us/sms/pricing/us),
which labels its table current as of October 2026, and
[PurpleAir API Data Points](https://www.purpleair.com/products/api-data), which
also lists higher-volume purchase tiers. A $10 purchase in the minimum tier
buys 1,000,000 points. This is not an estimate of the historical download needed
for A-14 or of a recurring annual data bill. Use the repository's PurpleAir
request estimator on the proposed sensors, fields, averaging and dates first.

PurpleAir's [Data License page](https://www.purpleair.com/license), marked
updated July 18, 2022, calls out distribution limits, attribution, a grantback
license and fair use, with links to its
[Terms of Service](https://www.purpleair.com/terms-of-service). Payment for API
points does not establish that the proposed commercial embedding or data resale
is permitted. Review the intended product against those terms before including
PurpleAir in a commercial proposal. This source check does not resolve that
licensing question or change the core EPA-only study.

## A partial messaging scenario

For a hypothetical 100 recipients, 30 days, one outbound message per recipient
per day, one rented long-code number, and one billing month:

| Message size assumption | Base SMS plus the listed low/high outbound carrier fee, and number rental |
| --- | ---: |
| One segment per message | $36.55 to $41.05 |
| Two segments per message | $71.95 to $80.95 |

These are partial messaging subtotals. The recipient count, sending frequency,
duration and segment counts are scenario inputs, not enrolled families or
observed usage. Bounds use the lowest and highest fees in the reviewed long-code
carrier table; they are not uncertainty intervals or guaranteed future bounds.

[Twilio's segment documentation](https://www.twilio.com/docs/glossary/what-sms-character-limit)
distinguishes GSM-7 from UCS-2 encoding. A single GSM-7 segment supports 160
characters, versus 70 for UCS-2; concatenated SMS normally uses 153 or 67 per
segment. Encoding and actual message length must be checked for each language.
Do not equate a translated message with one billable segment. No message text or
family data was submitted to a calculator.

The scenario excludes inbound replies, registration and campaign fees,
failed-message charges, taxes, consent and delivery administration, web hosting,
storage, computation, data acquisition, staff time and clinical or translation
review. Unknown costs are not zero. Obtain the applicable fees and measure
usage before presenting an all-in budget.

The arithmetic below reproduces the counts and partial subtotals without
contacting a service:

```python
from decimal import Decimal as D

print("district inventory:", sum([935, 197, 295]))
print("CA 2025-26 district rows:", sum([345, 515, 76]))
print("CA/OR/WA NSCH child estimates:", sum([548709, 26594, 83411]))
print("NHIS national children, millions:", D("4804000") / D("1000000"))
print("minimum-tier API points:", D("10") * D("100000"))
for segments in (1, 2):
    totals = [D(100) * D(30) * D(segments) * (D("0.0083") + fee) + D("1.15")
              for fee in (D("0.0035"), D("0.005"))]
    print(segments, [str(value.quantize(D("0.01"))) for value in totals])
```

For a later measured budget, use:

`outbound segments * (base rate + carrier rate)`

`+ inbound segment costs + number-months * rental rate + registration/campaign fees`

`+ compute + storage + hosting + data + support + review + applicable taxes`.

Daily inference should be timed separately from training and downloading. The
[local inference profiler](INFERENCE_TIMING.md) measures a resident exported
model's calibrated forward calls and records artifact loading, preparation and
device setup separately. Its documented exclusions still need measurement for
a complete deployment budget. Record
the point count, graph size, forecast horizon, ensemble size, hardware, device,
wall time, peak memory and failure/retry behavior. Apply a cloud rate only after
timing a comparable deployment. The current local development checks are not a
cloud-hosting quote or evidence of a production service commitment.

## Evidence still needed before replacing the brief's placeholders

| Missing input | Evidence that would resolve it |
| --- | --- |
| First buyer and workflow | A genuine interview identifying a specific operational decision, its timing, current tools and who owns it |
| Annual subscription price | A scoped offer tested with the budget owner, with support, coverage and procurement requirements recorded |
| Comparable spending | A paid contract or authorized quote with scope separated into hardware, data, forecasting, alerts and support; the reviewed no-cost partnership does not price a forecast subscription |
| Addressable district count | Reviewed eligibility criteria and a named district list; state totals alone do not resolve smoke exposure, service coverage or purchase intent |
| Health-plan price or savings | Reviewed evidence with the same population, payer, year and outcome; distinguish charges, paid amounts and cost, and establish an effect before claiming avoided visits |
| Complete first-season budget | Measured inference and alert volume plus priced hosting, data rights, staffing, review and applicable messaging fees |

These gaps require team decisions, measurements or real conversations. A public
source check cannot establish an interview, commitment, pilot partner or letter
of support.

## Source ledger

All pages below were read on November 7, 2026. Links above identify the exact
pages, rather than search snippets. Dates here are displayed source dates, not
assumed publication dates. No account, checkout, contact form or alert signup
was used.

| Page | Date shown | Evidence read and limit |
| --- | --- | --- |
| CDE List of School Districts | Reviewed July 31, 2026 | Fiscal-year counts and excluded organization types |
| CDE Fingertip Facts | Reviewed October 15, 2026 | 2025-26 district-type rows and broader entity total |
| CDC Most Recent Asthma Data | Updated October 6, 2026 | 2023 NHIS national and NSCH child state estimates, age definition and standard-error labels; different survey populations |
| Oregon Schools & Districts Home | No count reference year shown | Official current count; not a synchronized regional census |
| OSPI About School Districts | No count reference year shown | Districts, compact schools and other jurisdictions listed separately |
| CARB California Air Districts | No date shown in retrieved body | Air-district count and responsibilities; no buying commitment |
| LA Unified Know Your Air Network | No page-wide date shown | District-described operational need; no verified contract price |
| Santa Ana Unified board minutes, item 13.12 | January 28, 2025 meeting | Approval of the named no-cost partnership; not proof of delivery or renewal |
| Santa Ana Unified/OneOC/GREEN-MPNA posted agreement | Stated term August 1, 2024 through July 30, 2026 | Section 1.12 excludes monetary district compensation; monitoring scope includes equipment and dashboard access |
| AirNow Air Quality Flag Program | No date shown in retrieved body | Existing guidance and alert links; no comparative evaluation |
| EnviroFlash About | No date shown | Delivery of forecasts and alerts; no product trial |
| Twilio U.S. SMS pricing | Pricing labeled October 2026 | Public rates and additional fee categories, retrieved fresh |
| Twilio How long can a message be? | No date shown in retrieved body | Encoding and segment-size rules; actual draft messages not assessed |
| PurpleAir API Data Points | No effective date shown | Minimum purchase, point tiers and organization requirement, retrieved fresh |
| PurpleAir Data License and Terms of Service | Updated July 18, 2022 | Reuse and attribution requirements requiring review for the intended product |
