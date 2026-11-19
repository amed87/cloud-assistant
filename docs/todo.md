# TODO / Roadmap — FAQ-Bot

Stand: 2026-11-19. Kurzfristige Arbeitspakete und mittlere Roadmap-Items
(Teams, Auth, Golden-Dataset-Automatisierung) in einem Dokument; ausstehende
Punkte aus der laufenden Session oben, Roadmap unten.

---

## 1. Kurzfristig (laufende Session)

- [ ] **`evaluate_rag` mit Trial-1025-Werten (mit DM) laufen lassen** — beide
      DM-Ersatz-Varianten sind nach Tuning end-to-end schlechter:
      DM=0/Fallback=False: Overall 101/138 (73.19 %) vs. Baseline 110/138.
      DM=0/Fallback=True: Tuning-F0.5 0.859 (schlechtester Lauf; der 0.5b-DM
      ratet bei unsicheren Facetten falsche Labels statt neutral zu bleiben).
      `.env` ist auf die Trial-1025-Werte (mit DM) zurückgerollt.
- [ ] **DM-Ersatz-Varianten dokumentieren (negativ abgeschlossen)** — Lektion:
      Tuning-F0.5 (β=0.5) toleriert viele FN, Overall (tp+tn) nicht; fallweise
      End-to-End-Eval als Ausschlaggeber nutzen. Falls die DM-Kosten trotzdem
      drücken: höheres FLOOR/MARGIN (seltenerer DM-Call) oder größeres
      Modell nur für den Fallback testen — jeweils mit Cache-Rebuild.
- [ ] **request_looper / Load-Profiler** — Latenzbilanz des Live-Configs
      messen, sobald die mit-DM-Konfiguration den 110er-Baseline bestätigt.
- [ ] **Dokumentation heben** — READMEs auf die Output-Ordnerstruktur
      (`Qualitätsevaluation`, `Loadtest`, `KPIs`, `Tuning`), den Zentroid-Workflow
      und das Tuning-Verhalten (Cache, Seed, `tuning_config.yaml`) machen.
      Tuning-README ist schon aktuell.
- [ ] **Anwendungs-Anleitungen** prüfen/ergänzen (siehe `docs/`):
      - Büromanagement: `content_config.yaml` bearbeiten, Health-Check nutzen
      - IT: DB-Update inkl. Zentroiden, Tuning, `evaluate_rag`, KPIs
- [ ] **Codebase-Review** — DRY/YAGNI/Namenskonventionen, „test behaviour
      not implementation". Bekannte Kandidaten:
      - `app/dependencies/chat.py` baut ein `AnswerabilityGate`, übergibt aber
        `answerability_gate=None` (Altstand klären)
      - Feature-Muster-Uniformität: Intent-Regex (content_config
        `fragetyp_filter`) vs. Centroid-Facetten — Entscheidung treffen,
        ob/von wann ein Feature-Konzept das andere ersetzt
- [ ] **`CENTROID_DM_FALLBACK=True` praktisch testen** — Qualitätsgewinn vs.
      Latenz (P95 ~4.5 s am 5-s-Budget, Load-Profiler nutzen).
- [ ] **Git-Commit** — weite Teile des Working Trees sind uncommitted
      (parameter_tuning/, Zentroid-Module, Feature-Factory, Output-Umsortierung).
      Kleinere, thematische Commits empfohlen.

## 2. Grüner Berg — warten auf Tuning/eval-Ergebnisse

- [ ] Tuning-Ergebnis der Centroid-Gruppe auswerten (Bonus/Penalty/Gewichte vs.
      Gate-Flips — die 4drops des Sondierungslaufs waren reine Gate-Margin-Effekte)
- [ ] Bei gutem Ergebnis: Intent-Regex vs. Centroid-fragetyp im Tuning
      gegeneinander messen (Gewicht-0-Test) und Regex ggf. ablösen
- [ ] B.I.-Bericht/Schulungsfolien aktualisieren, sobald Zahlen stehen

---

## 3. Roadmap

### 3.1 Golden-Dataset automatisch erzeugen

- [ ] Flag in der `.env`/`tuning_config.yaml`: golden dataset bei jedem
      DB-Update neu generieren (default aus, weil manuelle Kontrolle)
- [ ] Generierungsansatz:
      1. Positiv-Fragen direkt aus den KB-Fragen (bestmögliche "positiv"-Cases)
      2. Paraphrasen via Decision-Model (Restbudget vorhanden → im Tuning-Modus
         abwägen; alternativ einfache heuristische Paraphrasen)
      3. Grenzfälle generisch ableiten (Kleinschreibung, Tippfehler-Ersetzung,
         Fragmente/„-{Stichwort}-Stil", lange Ausschreibungen)
      4. Negativ-/Fast-Treffer-Fälle aus thematisch verwandten KB-Einträgen
- [ ] `erwartete_entry_id` ohne LLM direkt aus dem Entry ableiten;
      nicht-deterministische DM-Paraphrasen mindestens versionieren
- [ ] Output unter `eval/data/` (z. B. `golden_dataset_auto_YYYY-MM-DD.csv`);
      jeder generierte Datensatz sollte vor dem produktiven Einsatz manuell
      durchgesehen werden (Autogeneration erzeugt keine Wahrheit)
- [ ] Wechselwirkungen beachten: Tuning-Cachedaten sind query-abhängig → nach
      Golden-Set-Wechsel ist `--refresh-cache` Pflicht (steht so im
      Tuning-README, aber an der Stelle nochmal hervorheben)

### 3.2 Teams-Anbindung

- [ ] Produktionaler Kanal: Azure Bot Service → Microsoft Teams (adaptive
      Cards statt Chat-UI); klären, wie Datei-/Karten-Antworten transportiert
      werden
- [ ] Recherche Microsoft-Test-Ökosystem (statt eigenem Test-Setup):
      - Bot Framework **Emulator** für lokale End-to-End-Tests
      - **Microsoft 365 Agents Toolkit** (ehem. Teams Toolkit) für
        Lokaldebugging und Provisionierung
      - Unit-Tests der Bot-Logik unabhängig vom Kanal (BotAdapter-Abstraktion)
- [ ] Langlaufende Antworten: vom 5-s-Budget lösen — Loading-Nachricht via
      Teams (aktivität "typing") + async "Antwort folgt"-Karte, späteres
      Update der Adaptive Card wenn der Score-Rechner fertig ist
- [ ] Deployment-Trennung beachten: aktueller Service läuft lokal mit Ollama;
      Teams-Anbindung braucht definierten Hosting-Weg (Server in der Firma
      vs. Azure Bot Service-Tunneling) — mit IT klären
- [ ] Rate-Limits/Retry-Verhalten des Teams-Kanals beim Load-Profiler mittesten

### 3.3 Sicherheit

- [ ] **Rollenbasierte Auth** (Prod: firmenweit einheitliche Auth → Microsoft
      Entra ID, vermutlich via SSO/OIDC):
      - Rollenmodell klären: standard (Fragen stellen) vs. Büro
        (content_config, Health-Check) vs. IT (Admin-Endpoints wie
        `/admin/update-db`, Tuning/Deploys) — aktuell sind die Admin-Endpoints
        offen
      - Test-Bibliothek wie für Teams recherchieren: Microsoft liefert für
        Entra ID **MSAL** (microsoft-authentication-library-for-python) plus
        Beispiele/Integrationstests; Alternativen: Local-Test-Tenant oder
        Entra ID **app registrations** mit Test-Scope
- [ ] **Prompt-Injection-Schutz** — Risikobild analysieren: Die User-Query
      fließt in Decision-Model-Prompts ein; die DM-Ausgabe wählt aber nur ein
      Label/eine Wahrscheinlichkeit aus einer geschlossenen Wahl, ohne
      freie Antworten an den User. Priorität trotzdem messen und zumindest:
      - Query-Vorfilter (Länge, Steuerzeichen, Prompt-Struktur-Signaturen)
      - DM-Outputs nur als Label-Auswahl aus der erwarteten Liste akzeptieren
        (passiert bereits; formalisieren/absichern)
      - Logging von verdächtigen Queries für spätere Musteranalyse
- [ ] **Restliche Basics durchziehen** (wenn Auth steht):
      - HTTPS-Erzwingung/Konfiguration hinter Reverse Proxy klären
      - Secrets nicht in Repo-Dateien (`.env` review, Deployment sauber trifft)
      - Debug-Endpoints entfernen oder absichern
      - Rate-Limiting pro Channel
- [ ] Security-Review mit der IT vereinbaren, sobald Teams-Anbindung und
      Auth-Anbindung eine erste Version haben

---

## Archiv / erledigt (Auswahl)

- [x] Zentroid-Klassifikation (Store, Cascade-Classifier, Match-Feature,
      Factory)
- [x] DB-Update + Zentroid-Rebuild verdrahtet (Integrationstest)
- [x] Feature-Factory (`app/quality/ranking/feature_factory`), 4 Duplikate
      entfernt; evaluate_rag misst jetzt das Live-Feature-Set inkl. Zentroide
- [x] Tuning centroid-aware + konfigurierbare Gruppen (`tuning_config.yaml`)
- [x] Tuning-Cache persistiert + Sampler-Seed + Weights-Snapshot
- [x] Output-Unterordner Tuning/Loadtest/Qualitätsevaluation/KPIs
- [x] Fortschrittsbalken in `evaluate_rag`
- [x] Heuristischer Sondierlauf mit aktiven Centroid-Gewichten (106/138;
      Resultat: Gate-Margin-Effekt ungetunter Parameter — bestätigt Bedarf
      für das Centroid-Tuning)
