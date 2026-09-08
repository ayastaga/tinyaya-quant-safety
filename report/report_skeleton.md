# Does It Survive the Phone?
### Auditing Safety, Reasoning, and Structured Output in Tiny Aya's Deployed Quantizations
(Fallback title if Evals D+E are both cut: "Does the Safety Floor Survive the Phone?")

**Abstract.** Tiny Aya (2026) reports a best-in-class multilingual safety floor
(min 87.0 / mean 91.1 MultiJail safe-rate) and highly consistent language
adherence — measured at BF16. Its deployment story requires 4-bit GGUF
quantization: the authors show full precision exhausts memory even on current
phones. We audit whether safety, language adherence, multilingual reasoning,
and structured-output validity survive in the shipped Q8_0/Q4_K_M/Q4_0
artifacts across all four released variants. [FINDINGS]. We localize mechanism
via per-tensor quantization maps and repair degradation with a [N]M-parameter
QLoRA adapter, validated by re-quantizing and re-auditing the deployed GGUF.
Code, adapters, and artifacts released.

## 1 Introduction
Deployed model ≠ evaluated model. Fig. 15 argument generalized: their own data
shows quantization damage concentrates in low-web-presence languages — for
safety, reasoning, and format alike, uniform survival would be the
extraordinary claim. Contributions (qualify per novelty table + §3.1 search).

## 2 Related Work
Marchisio et al. 2024; KISA EN/KO quantization-jailbreak study; OpenSafetyMini;
"Which Quantization Should I Use"; multilingual safety-gap literature; +
whatever the §3.1 searches surface for quantization×reasoning and
quantization×function-calling. Precise positioning sentence per axis.

## 3 Setup
Models/precisions; own vs official GGUFs; pinned versions; judge + calibration
gate vs Table 7; **per-cell n (judge tiers)**; **Phase 0.5 gate outcomes**
(including any cut axis — publish the gate result); template check result.

## 4 Audit
4.1 Safety floor vs precision (headline).
4.2 Per-language safety deltas + web-presence correlation.
4.3 Language confusion under quantization.
4.4 Translation ChrF.
4.5 Reasoning: accuracy / CoT adherence / truncation, each vs precision;
    English-vs-local gap vs precision. State token-budget confound first.
4.6 Structured output: validity ladder + key-name drift. Format robustness,
    NOT agentic capability.
4.7 Merged-vs-SFT safety fragility (Earth/Fire/Water vs Global).
4.8 Mechanism: tensor maps; does embedding fragility PREDICT which axes
    degrade? (State prediction before checking.)
4.9 Cross-axis synthesis: do all axes fail in the same languages?
    (fig_webpresence_all_axes — likely the strongest single figure.)

## 5 Repair
Recipe; closed loop (merge→BF16→re-quantize→re-audit GGUF incl. D/E + XSTest);
pre-registered criteria incl. no-reasoning-regression; if robust: stress-quant
methods demonstration, framed honestly.

## 6 Limitations
Judge approximation; subsampling + per-cell n; MT-benchmark noise (incl. any
machine-translated MGSM splits); greedy single-run, no seed variance;
NF4/GGUF training-deployment mismatch; CoT token-budget confound; Eval E scope;
web-presence bins placeholder vs paper's exact CommonCrawl counts.

## 7 Conclusion
Takeaway for practitioners deploying multilingual SLMs on-device.
