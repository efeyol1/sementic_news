# News Sentiment Labeling Guide

Use these rules when filling `reviewed_label` in sentiment QA exports.
The target is news-event sentiment, not the reader's personal preference.

Allowed labels: `negative`, `neutral`, `positive`.

## Negative

Use `negative` when the reported event is clearly harmful or adverse:

- war, attack, death, injury, violence, threat
- corruption, fraud, criminal investigation, abuse of office
- economic loss, inflation pressure, bankruptcy, layoffs
- political crisis, party split, institutional breakdown
- disaster, accident, public risk, service disruption
- sanctions, coercion, intimidation, rights violations

## Neutral

Use `neutral` when the article is mainly factual, procedural, or mixed:

- appointment, meeting, negotiation, diplomatic visit
- policy proposal without concrete accepted impact
- interview, analysis, explainer, program title, ticker
- routine weather without severe harm
- sports/culture/lifestyle item without strong evaluation
- mixed article where positive and negative cues cancel out

## Positive

Use `positive` when the reported event is clearly beneficial:

- improvement, relief, recovery, rescue, successful reform
- lower cost, easier access, stronger rights, better service
- achievement, award, breakthrough, successful event
- cultural/lifestyle item with explicit positive framing
- concrete agreement or deal with clearly beneficial outcome

## Tie Breakers

- Diplomatic meetings are `neutral` unless a concrete beneficial deal is reported.
- Policy proposals are `neutral` until effects are concrete.
- A positive quote from a politician does not make the event `positive`.
- Lifestyle recommendations are `neutral` unless the article itself strongly praises the outcome.
- Program/ticker headlines such as "Schlagzeilen" or "Wirtschaft vor acht" are `neutral`.
- If harm is central and benefits are incidental, use `negative`.
- If unsure between `neutral` and a polar label, prefer `neutral`.
