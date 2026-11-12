# FAQ-Wissensbasis

Die Wissensbasis wird als CSV gepflegt, beim Import in FAQ-Dokumente umgewandelt, eingebettet und in ChromaDB gespeichert.

## CSV-Format

- UTF-8
- Semikolon als Spaltentrenner
- Erforderliche Spalten: `ID`, `Frage`, `Antwort`
- Verwendete optionale Spalten: `Themen`, `Fragetyp`, `Schlüsselwörter`

Der FAQ-Loader kann die drei Metadatenspalten auslassen. Der Demo-Paket-Builder verlangt dagegen alle sechs Spalten, bevor er eine Demo-FAQ erstellt.

Beispiel für den Aufbau (keine echten FAQ-Inhalte):

```csv
ID;Frage;Antwort;Themen;Fragetyp;Schlüsselwörter
beispiel-01;Wie erreiche ich das Team?;Über das interne Portal.;Kontakt Zuständigkeit;Zuständigkeit;Team Kontakt
```

Der Loader zerlegt `Themen` und `Schlüsselwörter` an Whitespace. Verwende daher keine mehrteiligen Begriffe in diesen Feldern, sofern die Importlogik nicht angepasst wird. Fehlende ID, Frage oder Antwort führen dazu, dass die Zeile übersprungen wird.

## IDs und Updates

Der gespeicherte FAQ-Schlüssel wird aus der stabilen `ID` und einem Hash aus Frage, Antwort, Themen, Fragetyp und Schlüsselwörtern gebildet. Eine Inhaltsänderung erzeugt dadurch eine neue Versions-ID.

Der Importer vergleicht die Einträge einer Quelle mit bereits gespeicherten IDs und löscht veraltete Einträge dieser Quelle. Behalte deshalb die fachliche `ID` beim Bearbeiten eines bestehenden Eintrags bei. Weise bei einem fachlich neuen Eintrag eine neue ID zu. Importiere nicht dieselbe CSV mit wechselnden Quellpfaden, weil der Importer die Herkunft zur Zuordnung verwendet.

## Prüfen und importieren

1. Stelle sicher, dass `FAQ_INPUT_FILE` auf die zu prüfende Datei zeigt.
2. Starte Ollama und prüfe, dass das Embedding-Modell verfügbar ist.
3. Führe zuerst den FAQ Health Check aus:

   ```powershell
   $body = @{ file_path = "..\data\DB-Input\Tabelle_Chatbotfragen.csv" } |
       ConvertTo-Json
   Invoke-RestMethod -Uri http://127.0.0.1:8000/admin/faq-health-check `
       -Method Post -ContentType "application/json" -Body $body
   ```

4. Prüfe Status, Self-Retrieval-Ausfälle, ähnliche FAQ-Paare, geringe Margins und Embedding-Probleme.
5. Nach fachlicher Freigabe den Import starten:

   ```powershell
   Invoke-RestMethod -Uri http://127.0.0.1:8000/admin/update-db `
       -Method Post
   ```

   Der Import läuft im Hintergrund. Der HTTP-Erfolg bestätigt nur, dass der Auftrag gestartet wurde.
6. Eine bekannte FAQ über Quick Chat prüfen und bei Bedarf den aktuellen Index beziehungsweise die Logs kontrollieren.

Die Admin-API hat derzeit keine Authentifizierung. Führe diese Aufrufe nur lokal oder in einer geschützten Umgebung aus. Für einen freigegebenen Update-Prozess Datenbank-Backup und Rückfallplan vorsehen.

## Health-Check-Bericht

Der Health Check nutzt einen temporären Chroma-Store, importiert dort die Datei und räumt den temporären Store nach dem Lauf auf. Der Bericht wird zusätzlich unter `..\eval\output` abgelegt. Prüfe den JSON-Bericht und die API-Antwort gemeinsam; Fehlermeldungen aus Modell- oder Dateizugriffen stehen in den Serverlogs.

## Inhaltliche Pflege

- Antworten sollen die Frage direkt und ohne nicht belegte Zusatzannahmen beantworten.
- Veraltete Informationen entfernen oder ersetzen; danach erneut prüfen und importieren.
- `Fragetyp` sollte zu den Labels und Regex-Filtern in `content_config.yaml` passen.
- Keywords sind optionale Suchsignale und ersetzen keine gut formulierte Frage.
- Keine Passwörter oder andere Zugangsdaten in die Wissensbasis aufnehmen.
