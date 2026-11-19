# parameter_tuning

Optimiert Reranker-Gewichte und Answerability-Gate-Thresholds des FAQ-Chatbots
gegen das Golden-Set. Läuft nach jedem größeren Update der Wissensdatenbank –
Overfitting aufs aktuelle KB ist hier gewollt, da die Thresholds die aktuelle
Datenlage adaptieren sollen.

Ersetzt den früheren Grid-Sweep (`hyperparameter_tuning.py`), der bei
~15 Mio. Konfigurationen × 118 Cases × Gate-Aufrufen nicht mehr terminierte.

## Ansatz

Der Sweep ist zu groß, nicht das Problem selbst. Drei Ursachen werden direkt
adressiert:

1. **Gate-Thresholds werden nie gesampelt.** Das Gate prüft nur
    `top_score >= min_score` und `gap >= min_gap`. Für eine feste
    Reranker-Konfiguration kann man daher aus ~118 (top, gap)-Paaren die
    komplette Konfusionsmatrix über *alle* Threshold-Kombinationen analytisch
    ableiten (Landscape). Das Grid über min_score/min_gap entfällt komplett –
    und die Thresholds kommen aus den beobachteten Werten statt aus geratenen
    Grenzen.
2. **Feature-Scores sind konfigurationsunabhängig.** Intent liefert einen
    Faktor (Bonus / Penalty / neutral), Keyword einen Faktor, DM-Scores sind
    fix. Alle Matrizen werden einmal vorberechnet; eine Konfiguration ist nur
    noch ein gewichteter Summenzugriff (numpy), kein Reranker-Lauf.
3. **Optuna (TPE) statt Grid** für die Gewichtungsdimensionen. Pro Trial:
    Score-Matrix → Ranking → Landscape → bestes F0.5. Ein Trial kostet
    Millisekunden.
4. **Threshold-Kandidaten sind Mittelpunkte zwischen beobachteten Werten**
    statt der Werte selbst. Der Decision-Provider (Ollama) jittert Scores um
    ~±0.001–0.004 zwischen Prozessläufen (Thread-Scheduling ändert die
    Summationsreihenfolge der Floats); ein Threshold, der exakt auf einem
    beobachteten Score liegt, ist eine Messerkante – ein Grenzfall kippt
    accept/reject und verschiebt das gesamte F0.5 (beobachtet am
    12.11.2026: 0.904 vs. 0.897 bei identischen Settings). Mittelpunkte
    halten auf beiden Seiten halben lokalen Abstand Puffer.

## Phasen

| Phase | Modul | Was passiert |
|-------|-------|--------------|
| 0 – Cache | `cache.py` + `service.py` | Einmal Retrieval pro Golden-Case (`quick_retrieve(eval=True)`); sammelt Base-Scores, Intent-Faktoren, Keyword-Faktoren (je Normalisierung), DM-Scores pro Prompt. Der einzige teure Teil (Ollama/Chroma). |
| 1 – Matrizen | `matrices.py` | Cases zu dichten numpy-Arrays stapeln (`(C, K, …)`). |
| 2 – Suche | `search.py` | Optuna-Study über Reranker-Gewichte; Gate-Parameter sind **nicht** im Suchraum. |
| 3 – Landscape | `landscape.py` | Analytische F0.5-Fläche über alle (min_score, min_gap)-Paare (Mittelpunkt-Grid); Cross-Check gegen das echte `AnswerabilityGate`. |
| 4 – Finalisierung | `finalize.py` | Sieger wählen, Plateau-Check, **Jitter-robuste Threshold-Auswahl**, Boundary-Warnings, Export. |

## Benutzung

    ```bash
    python -m parameter_tuning --trials 2000 \
        --study-db sqlite:///tuning.db \
        --output recommended_settings.json
    ```

Alle Flags optional mit den Defaults: `--trials` (2000), `--output`
(`recommended_settings.json`), `--study-db` (ohne: in-memory), `--seed`.

Ohne `--study-db` ist ein Lauf ein Wegwerf-Lauf. Mit SQLite-Storage lässt sich
die Studie fortsetzen (`load_if_exists=True`) und über Zeit vergleichen.
Das Beispiel benutzt eine eingerückte Codezeile statt eines Fenced Blocks –


## Output

**Terminal:** bestes F0.5, die empfohlenen Settings, Boundary-Warnings und
eine Warnung, falls das echte Gate von der analytischen Zählung abweicht.

**JSON-Report:** zusätzlich Confusion-Counts am Optimum, Plateau-Analyse
(wie flach ist die Threshold-Umgebung des Optimums?), Gate-Validierung und
Robustheit: Unter `robustness` simuliert der Report die beobachtete
Score-Jitter des Decision-Providers (Monte-Carlo: `TOP_JITTER`/`MARGIN_JITTER`
auf jedem Case, `ROBUST_REPEATS` Wiederholungen, fester Seed) und meldet
`f05_at_recommendation`, `simulated_mean_f05` und `simulated_worst_f05` am
empfohlenen Paar. Differieren mean/worst deutlich vom Nominalwert, sitzt die
empfohlene Schwelle trotzdem nahe an einem Grenzfall.

Die `recommendation`-Keys entsprechen 1:1 den `settings.*`-Namen
(INTENT_WEIGHT, INTENT_MATCH_BONUS, INTENT_MISMATCH_PENALTY, KEYWORD_WEIGHT,
KEYWORD_MATCH_NORMALIZATION, DECISION_MODEL_WEIGHTS, RAG_MIN_SCORE,
RAG_MIN_SCORE_GAP) und können direkt ins Config-Repository übernommen werden.

## Interpretation

- **Boundary-Warnings:** Liegt das Optimum an einer Range-Grenze, war die
    Range zu eng → in `search.py` (`FLOAT_RANGES`/`DM_RANGE`) erweitern und mit
    gleichem Storage weiteroptimieren. Damit ist die Frage „traue ich den
    Erfahrungswerten?" datenbasiert beantwortet.
- **Plateau:** Viele Threshold-Paare nahe am Bestwert ⇒ robuste Wahl.
    Nadelpunkt ⇒ Einstellung ist Zufall der Stichprobe, Vorsicht beim
    Übernehmen. Die empfohlenen `RAG_MIN_SCORE`/`RAG_MIN_SCORE_GAP` sind
    davon unabhängig immer das jitter-robusteste Plateau-Paar (höchster
    simulierter Mittelwert-F0.5), nicht der Nominal-Argmax – deshalb kann
    `f05_at_recommendation` geringfügig unter `best_f05` liegen; das ist
    gewollt (Trading der letzten Nachkommastelle gegen Stabilität).
- **Robustheit:** `simulated_worst_f05` ≈ `simulated_mean_f05` am empfohlenen
    Paar ⇒ die Thresholds vertragen die reale Score-Jitter des Providers.
    Große Lücke ⇒ nachbessern.
- **Gate-Validierung:** Muss immer übereinstimmen. Falls nicht: eine Annahme
    über das Gate stimmt nicht (siehe unten).

## Annahmen / Prüfpunkte

Diese Annahmen gehen in den Code ein und sollten bei Änderungen am Rest des
Systems einmal verifiziert werden:

1. Jeder Golden-Case liefert **genau 5 Chroma-Kandidaten**; der Reranker
    sortiert nur um (der Gap ist daher immer definiert).
2. Das Gate entscheidet ausschließlich über Score- und Gap-Schwellen (kein
    weiteres Kriterium in `evaluate_quick`).
3. `IntentFeature` gibt bei Mismatch einen **negativen** Faktor zurück, dessen
    Betrag die (positive) Penalty-Magnitude ist; die Detection in `cache.py`
    (`> 0` → Bonus, `< 0` → Penalty`) hängt daran.
4. Fehlende Scores sind `0.0`, niemals `None`.
5. `rr.feature_scores` ist pro Kandidat-Id (mit Versionssuffix) und pro
    Prompt-Key getastet.

## Abhängigkeiten

- `numpy` (ohnehin transitiv via ChromaDB vorhanden)
- `optuna` (pip, MIT-Lizenz)
- Rest wie das Bewertungssystem (App-Module; Ollama und ChromaDB müssen für
    Phase 0 laufen)

## Nicht im Umfang

Latency-/Resource-KPIs, Kategorie-Gewichtung der Objective, Persistenz des
Retrieval-Caches (bei Bedarf: `--cache-file`-Flag nachrüsten), Holdout-Split
(bewusst weggelassen, da Overfitting aufs KB gewollt ist; den Golden-Set-Stand
daher im Report-Header mitloggen, sobald Datensätze versioniert werden).