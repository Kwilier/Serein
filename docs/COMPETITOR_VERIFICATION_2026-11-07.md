# Competitor verification and novelty search (task E-5)

Checked November 7, 2026 against product-owner pages and primary papers. This
completes the verification that `COMPETITOR_SOURCE_CHECK.md` (Nov 6) started and
adds a novelty search for the combination Serein claims. Items marked
**[unverified]** could not be confirmed from a primary source; do not put them in
the brief, video or website.

The claim we must defend (Brief Q4 "What is new"):
1. Neighborhood-level smoke PM2.5 forecast 1 to 3 days ahead that corrects NOAA
   HRRR-Smoke/AQM with live regulatory-monitor readings (GNN), with calibrated
   uncertainty, scored head-to-head against NOAA and AirNow on held-out seasons.
2. Forecasts for places with no monitor.
3. A per-child smoke-sensitivity model from a short daily caregiver check-in,
   producing a green/yellow/red plan in the family's language.

## 1. Corrected competitor table

| Who | What they give (verified) | What they don't | Source |
|---|---|---|---|
| EPA AirNow forecast | Agency forecasts "via reporting areas"; "Reporting areas vary by size and population, covering anywhere from part of a city to an entire county area"; "Forecasts may cover from one to six days"; most show today and tomorrow | No neighborhood or sensor-corrected forecast; "forecasts are not necessarily available for all reporting areas"; no personalization | [AirNow API FAQ](https://docs.airnowapi.org/faq), [AirNow how-to](https://www.airnow.gov/how-to-use-this-site), [AirNowTech forum](https://forum.airnowtech.org/t/what-is-a-reporting-area/109) |
| AirNow Fire and Smoke Map | NowCast AQI from permanent and temporary monitors plus corrected PurpleAir sensors; satellite plumes; text "Smoke Outlooks" from Air Resource Advisors near large staffed fires | No gridded model forecast; sensor data "is not used to generate the current NowCast AQI" on AirNow's dial/app | [FASM FAQ (PDF)](https://document.airnow.gov/airnow-fire-and-smoke-map-faqs.pdf) |
| EnviroFlash | "sends air quality information for your city to your email or mobile phone" | City-level; no personalization | [EnviroFlash](https://m.enviroflash.info/about.cfm) |
| Air Quality Flag Program | Organizations raise a colored flag matching the local AQI forecast | Organization-level; no per-child guidance | [EPA Flag Program](https://www.epa.gov/air-quality/air-quality-flag-program) |
| NOAA HRRR-Smoke | Operational; "forecast cycles go out to 48 hours"; ~3 km. **RRFS v1** (3 km, 84 h, smoke and dust) scheduled operational Nov 3, 2026; HRRR frozen, retire ~2028 (go-live **[unverified]**) | No monitor assimilation; no health layer | [HRRR-Smoke](https://rapidrefresh.noaa.gov/hrrr/HRRRsmoke/), [SCN26-048](https://www.weather.gov/media/notification/pdf_2026/scn26-048_Updated_RRFS_and_REFS_Implementation.aaf.pdf) |
| NOAA AQM / NAQFC | "hourly forecast guidance for O3 and PM2.5 up to 72 hours twice per day"; ~5 km CONUS; AQMv7 since May 2024; a "Bias-Corrected PM2.5" product exists | ~5 km, not neighborhood; no public uncertainty; no health layer | [AWS registry](https://registry.opendata.aws/noaa-nws-naqfc-pds/), [NOAA ARL](https://www.arl.noaa.gov/news-pubs/arl-news-stories/air-quality-model-helps-during-wildfires/) |
| PurpleAir map | "real-time, free air quality map"; promotes partner ExactAQ "current and forecast" map | No forecast of its own; no health personalization | [PurpleAir tips](https://www.purpleair.com/blog/tips-for-using-purpleair-s-free-air-quality-map), [ExactAQ post](https://www.purpleair.com/blog/how-purpleair-data-supports-exactaq-a-hyper-local-air-quality-map) |
| IQAir AirVisual | "7-day forecast developed with machine learning and AI"; threshold notifications; "Stay informed on wildfires, smoke, and air quality events" | No asthma/symptom personalization; no smoke-model forecasts | [IQAir app](https://www.iqair.com/us/air-quality-monitors/air-quality-app), [App Store](https://apps.apple.com/app/id1048912974) |
| Google Air Quality API (ex-BreezoMeter) | Hourly forecast "up to 96 hours"; "500 x 500 meters"; recommendations for "those with asthma"; Maps layer fuses PurpleAir, NOAA satellite smoke | No per-person learning; no public uncertainty; no published skill vs NOAA/AirNow | [API overview](https://developers.google.com/maps/documentation/air-quality/overview), [Maps help](https://support.google.com/maps/answer/11270845?hl=en) |
| Apple Weather | "Air quality information is provided by BreezoMeter for ... the United States" | Forecast horizon **[unverified]**; no personalization | [Apple HT211777](https://support.apple.com/HT211777) |
| California Smoke Spotter 2.0 (CARB) | "Next-day wildfire smoke forecast for localities statewide"; PurpleAir AQI; fire alerts | 24-hour only; no symptom input | [CARB release](https://ww2.arb.ca.gov/news/california-air-resources-board-releases-california-smoke-spotter-2) |
| Propeller Health | Resmed: "has ceased operations"; "Resmed is retaining the Propeller brand, intellectual property, and assets" | Shutdown date **not stated**; successor Hailie app lists no air-quality features | [Resmed](https://www.resmed.com/en-us/health-professionals/solutions/propeller-health/), [Hailie](https://apps.apple.com/us/app/hailie/id421901698) |
| EPA Smoke Sense | "Starting on December 8, 2024, the Smoke Sense app will be unavailable"; "down through Summer 2025"; relaunch date unknown | Not a forecaster; citizen-science diary | [EPA Smoke Sense](https://www.epa.gov/air-research/smoke-sense) |
| AirRater (Australia) | See deep-dive below | | |
| Boise State "Asthma Activity Advisor" (May 2026) | Student app: enter a child's symptom level + location, pulls real-time AQI, recommends activity intensity; English and Spanish; "available shortly" | No forecast; no learning; unreleased | [Boise State](https://www.boisestate.edu/research-resilience/2026/05/08/asthma-attackers-building-a-smarter-way-to-protect-children-during-poor-air-quality/) |
| UW Medicine smoke outreach (2025-26) | Epic EHR + neighborhood pollution data to reach 64,000 asthma/COPD adults by email, text, portal; 14% engaged, 7% booked telehealth; year 2 launched July 2026 | Adults; no forecast stated; no individual learning. **Evidence that health systems will pay for neighborhood smoke outreach.** | [WSHA report](https://members.wsha.org/weekly-newsletter/weekly-report-friday-july-10-thursday-july-16-2026/uw-medicine-data-and-outreach-connects-patient-to-critical-wildfire-smoke-guidance/) |
| Ventura County farmworker smoke alert | Daily texts in English and Spanish, audio in Mixteco and Zapoteco, triggered at AQI 151 | Observation-triggered, not forecast | [VCAPCD](https://www.vcapcd.org/farmworker-wildfire-smoke-alert/) |
| FireSmoke.ca / BlueSky Canada | Forecasts four times a day, Apr to Oct, ~12 km | No per-location time series; no monitor correction; no health layer | [FireSmoke FAQ](https://firesmoke.ca/faq/) |
| Univ. of Utah / Trace AQ (Aug 2025) | CMAQ with ML augmentation; free demo to one day, paid product four days | No monitor-corrected neighborhood claim; no health layer | [Utah ChE](https://www.che.utah.edu/u-scientists-develop-ai-powered-tool-to-forecast-wildfire-smoke/) |
| AsthmaMD, Wizdy, AirCare, Kaiser/Blue Shield alerts | Not fetched in depth | **[unverified]**: do not cite | |

## 2. AirRater deep-dive (closest precedent at the personalization layer)

- Free app developed in 2015 by University of Tasmania (Menzies) with Tasmanian EPA, CSIRO, ANU; rolled out Tasmania 2015, ACT 2017, NSW 2019, then national; since the Nov 2023 merger with Melbourne Pollen it is run by AirHealth Pty Ltd. [MDPI IJERPH 2021](https://www.mdpi.com/1660-4601/18/7/3591), [UTAS 2023](https://www.utas.edu.au/about/news-and-stories/articles/2023/unis-establish-national-air-quality-and-pollen-service)
- Active in 2026: Google Play "Updated on 20 Aug 2026," 10K+ downloads. [Google Play](https://play.google.com/store/apps/details?id=com.senset.airrater&hl=en-AU)
- Data: near-real-time PM2.5, temperature, pollen; beyond 20 km from a station, an estimate from CSIRO's AQFx model. [CSIRO](https://research.csiro.au/aqfx/aqvx-and-airrater/)
- Forecast: historically none (2018 paper lists forecasting as future work); now "real-time and forecast air quality"; v5.0.1 "district-level forecast map." Horizon, resolution, skill **not published**.
- Personalization (documented): "a statistical model using weighted linear discriminant analysis is run for each individual and each of their symptoms" every 24 h; activates after 20+ symptom reports; warnings only when Kappa > 0.6; current app has a "trigger lab" and "Individualised alerts for your symptom triggers." Adult self-report; no caregiver/child mode found; 2021 study: "only available in the English language." [Johnston et al. 2018](https://acuresearchbank.acu.edu.au/download/ea80c7d8cb8878d4146e198cc6c5ccbfded5b9f331db02a99ecf648dd4d7b138/2978923/OA_Johnston_2018_Using_smartphone_technology_to_reduce_health.pdf)

**Bottom line:** AirRater already combines a smoke forecast, model-filled coverage away from monitors, and per-user symptom-trigger learning with alerts. It does not operate outside Australia, correct an NWP forecast with a GNN, publish uncertainty or skill, model children via caregiver check-ins, or produce a Spanish action plan. Name it in the brief as the closest precedent.

## 3. Research models and how Serein differs

- **ML bias correction of HRRR-Smoke** (Dong, Mao, Ahmadov et al., NOAA GSL, *ACS ES&T Air*): RF/CNN correcting HRRR-Smoke with PurpleAir and satellite AOD, Alaska only. Research code; no uncertainty; no health layer. [HAQAST](https://haqast.org/publications/machine-learning-bias-correction-of-hrrr-smoke-surface-pm2-5-forecasts-during-the-alaska-wildfire-season/)
- **GNN PM2.5 forecasting, California** (Liao, Buch, Lamb, Gentine, *Environmental Data Science* 2025): spatio-temporal GNN for hourly PM2.5; monitors only, no NWP correction, no uncertainty. [arXiv 2312.04291](https://arxiv.org/abs/2312.04291)
- **GNNs under distribution shift** (Liu et al., *AI Open* 2024): "GNN models tend to suffer more from distribution shift." Serein's held-out-season test addresses this; cite as motivation. [Illinois Experts](https://experts.illinois.edu/en/publications/pm25-forecasting-under-distribution-shift-a-graph-learning-approa/)
- **AGATNet** (Dimri et al., *JGR: ML & Computation* 2024): adaptive graph attention bias-correction of CMAQ PM2.5, South Korea. Closest methodological precedent for "GNN corrects a chemical-transport forecast." [Wiley](https://agupubs.onlinelibrary.wiley.com/doi/10.1029/2024JH000244)
- **Forecast skill for individual decisions** (Berlinghieri, Burt, Giani, Fiore, Broderick, MIT, *BAMS* 2026): evaluated HRRR-Smoke, CAMS, GEOS-CF, NAQFC, HAQES, Aurora vs AirNow; **no forecast reliably identified high-pollution days; morning AirNow persistence did as well or better.** Serein must report a plain persistence baseline. [arXiv 2409.05866](https://arxiv.org/html/2409.05866v2), [BAMS](https://journals.ametsoc.org/view/journals/bams/107/3/BAMS-D-24-0291.1.xml)
- **Per-child PM-sensitivity model** (JMIR Preprint 84513, Korea, Sept 2025): 272 pediatric asthma patients, 1D CNN-LSTM identifies PM-sensitive patients, individualized alerts cut exacerbation ratio (median -6.6%, P<0.05) vs regional forecasts. Preprint, hospital study, personal sensors, no wildfire smoke, not a consumer product. [JMIR Preprints](https://preprints.jmir.org/preprint/84513)

## 4. Novelty verdict

- **Does any product do all three elements?** No.
- **Does any do two?** AirRater (forecast + per-user learning; Australia, adults, no published skill). Google/BreezoMeter (500 m fused forecast + no-monitor coverage; static advice, no learning, no public skill). Boise State Asthma Activity Advisor (child symptom + real-time AQI; no forecast, no learning, unreleased).
- The defensible "Our difference" paragraph is in `drafts/innovation_brief_q6-q10_DRAFT.md`, section 7.

## 5. Claims in our materials to soften or remove

1. "First/only neighborhood-level corrected smoke forecast" → remove "first." Claim: first to publish calibrated uncertainty and held-out head-to-head scores in a free family product.
2. "AirNow forecasts only one day ahead" → "typically today/tomorrow, by reporting area."
3. "Fire and Smoke Map is nowcast-only" → "no gridded forecast; text outlooks only where ARAs are deployed."
4. "No app personalizes from symptoms" → "no US, child-focused, forecast-linked product does."
5. "First per-child smoke-sensitivity model" → "first in a free consumer product using a daily caregiver check-in and a wildfire-smoke forecast."
6. "HRRR-Smoke is NOAA's operational smoke model" → add "and its successor RRFS"; verify the Nov 3, 2026 go-live before the pitch.
7. "Propeller Health shut down in [date]" → cite without a date.
8. "Smoke Sense discontinued Dec 2024" → use EPA's wording; do not say "permanently."
9. "Apple Weather uses Google data" → "provided by BreezoMeter (Google)"; do not quote an in-app horizon.
10. "IQAir has no wildfire features" → it does; the correct gap is symptom personalization and smoke-model forecasts.
11. "Beats NOAA and AirNow" → add plain persistence as a scored baseline (BAMS 2026).
12. Generic "Kaiser/Blue Shield programs" → replace with the verified UW Medicine outreach.
