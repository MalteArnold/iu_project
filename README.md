# iu_project
Hausarbeit für das Modul Programmieren mit Python
Autor: Malte Arnold
## Aufgaben
### Voraussetzungen
Du erhältst:
A) 4 Training-Datensätze
B) einen Test-Datensatz
C) einen Datensatz, der 50 ideale Funktionen beschreibt
### Aufgaben
Deine Aufgabe ist, ein Python Programm zu schreiben, welches mittels der vier Trainingsdatensätze (A) die vier
besten Passungen / Fits aus dem Datensatz von 50 idealen Funktionen (C) findet. Die folgenden Kriterien sollen
beachtet werden:
1. Das Kriterium zur Selektion idealer Funktionen für den Training-Datensatz ist die Minimierung der
Summe aller quadratischen y-Abweichungen (Least-Square).
2. Dein Programm muss den Test-Datensatz B zur Validierung der Selektion benutzen. Hierbei soll für jedes
x-y-Paar im Test-Datensatz überprüft werden, ob die Werte zu den vier idealen Funktionen passen.
a. Benutze ein Kriterium, welches sicherstellt, dass die maximale Abweichung zwischen der vorher
ermittelten idealen Funktion und den Testwerten nicht die maximale Abweichung zwischen den
Trainingsdaten (A) und den vier idealen Funktionen aus (C) um mehr als den Faktor Wurzel aus
zwei (sqrt(2)) übersteigt.
b. Sollten die Testdaten an die von Dir gefundenen vier Funktionen anpassbar sein, speichere für
jeden Testdatensatz die entsprechenden Abweichungen ab.
4. Alle Daten sollten logisch visualisiert werden
5. Schreibe Unit-Tests, wo immer möglich
### Details
• Du erhältst vier Trainingsdatensätze in Form von CSV-Dateien. Dein Python-Programm muss in der Lage
sein, eine SQLite-Datenbank (Datei) idealerweise über sqlalchemy unabhängig zu kompilieren und die
Trainingsdaten in eine einzelne, fünfspaltige Tabelle zu laden. Die erste Spalte zeigt die x-Werte aller
Funktionen. Tabelle 1 am Ende dieses Unterabschnitts zeigt Dir, welche Struktur Deine Tabelle
voraussichtlich haben wird.
• Die fünfzig idealen Funktionen, die auch über eine CSV-Datei bereitgestellt werden, müssen in eine
andere Tabelle geladen werden. Ebenso zeigt die erste Spalte die x-Werte, was bedeutet, dass
insgesamt 51 Spalten vorhanden sind. Tabelle 2 am Ende dieses Unterabschnitts beschreibt
schematisch, welche Struktur erwartet wird.
• Nachdem die Trainingsdaten und die idealen Funktionen in die Datenbank geladen wurden, müssen die
Testdaten (B) Zeile für Zeile aus einer anderen CSV-Datei geladen und - wenn sie das Kriterium im
Unterabschnitt 2 erfüllt - mit einer der vier abgeglichen Funktionen abgespeichert werden.
• Anschließend müssen die Ergebnisse in einer anderen vierspaltigen Tabelle in der SQLite-Datenbank
gespeichert werden. Gemäß Tabelle 3 am Ende dieses Unterabschnitts enthält diese Tabelle vier
Spalten mit x- und y-Werten sowie die entsprechend gewählte ideale Funktion und die damit
verbundene Abweichung.
• Schließlich werden die Trainingsdaten, die Testdaten, die gewählten Idealfunktionen sowie die
entsprechenden / zugewiesenen Datensätze unter einer entsprechend gewählten Darstellung der
Abweichung visualisiert.
### Struktur des Programms
• Das Programm soll soweit wie möglich Objekt-orientiert sein.
• Es soll mindestens eine Vererbungshierarchie (inheritance) haben.
• Benutze sowohl Standard als auch user-definiertes Exception Handling.
• Für die Programmlogik solltest Du Pandas benutzen, aber auch Visualisierung mittels Bokeh, matlibplot
etc.
• Schreibe Unit-Tests, wo immer es sich anbietet.
• Dokumentiere Dein Programm vollständig und mache von docstrings Gebrauch.
