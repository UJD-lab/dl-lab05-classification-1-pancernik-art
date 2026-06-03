[![Open in Visual Studio Code](https://classroom.github.com/assets/open-in-vscode-2e0aaae1b6195c2367325f4f02e2d04e9abb55f0b24a779b69b11b9e10269abc.svg)](https://classroom.github.com/online_ide?assignment_repo_id=24072953&assignment_repo_type=AssignmentRepo)
# Sztuczne sieci neuronowe i głębokie uczenie – sprawozdania (GitHub Classroom)

Temat i instrukcje do konkretnego laboratorium znajdziesz w Moodle.  
To repo służy do pracy i oddawania sprawozdań w formie notebooka `report.ipynb`.

## Jak to działa?

1. Wrzucone zostało jedno **repo-szablon (template)** z plikami startowymi (np. `report.ipynb`).
2. Dla każdego laboratorium dostajesz link do **GitHub Classroom** (z Moodle).
3. Po kliknięciu linku GitHub Classroom **tworzy dla Ciebie nowe, prywatne repozytorium** i **kopiuje do niego** zawartość template.

✅ To znaczy: w Twoim repo od razu pojawi się `report.ipynb` i ten plik `README.md`.  
Template jest tylko “źródłem startowym” — później pracujesz wyłącznie w swoim repo.

## Co robić w repo

1. Otwórz `report.ipynb`.
2. Uzupełnij treść (Markdown + Python).
3. Zapisz zmiany.
4. Zrób **commit** i **push** (wyślij zmiany na GitHuba).

## Co oddać

- `report.ipynb`
-  wykresy/pliki w `figures/` (opcjonalnie, jesli laboratorium tego wymaga)

> Nie wrzucaj dużych plików z danymi do repo. Datasety będą dostępne na Moodle lub link do pobrania ich na dysk.

## Jak wysłać (push) sprawozdanie do repo?

Masz dwa zalecane sposoby: **GitHub Desktop** albo **konsola**.

### GitHub Desktop

1. Zainstaluj GitHub Desktop: https://desktop.github.com/
2. Zaloguj się w GitHub Desktop na swoje konto GitHub.
3. Otwórz swoje repo w przeglądarce (link z GitHub Classroom) i kliknij zielony przycisk **Code**.
4. Kliknij **Open with GitHub Desktop**  
   (albo w GitHub Desktop: `File → Clone repository...` i wklej adres repo).
5. Wybierz folder na komputerze, gdzie ma się pobrać repo.
6. Otwórz `report.ipynb`, uzupełnij i **zapisz**.
7. W GitHub Desktop:
   - wpisz opis w polu **Summary** (np. `Lab – sprawozdanie`)
   - kliknij **Commit to main**
   - kliknij **Push origin**

✅ Po **Push origin** prowadząca powinna zobaczyć Twoje zmiany.

---

Git console

> Wymaga zainstalowanego `git` (https://git-scm.com/) i zalogowania do GitHuba (np. przez token HTTPS lub SSH).

1. Sklonuj swoje repo (adres skopiujesz z przycisku **Code** na GitHubie):

   ```bash
   git clone ADRES_REPO
   cd NAZWA_REPO

2. Dla każdego laboratorium dostajesz link do **GitHub Classroom** (z Moodle).
3. Po kliknięciu linku GitHub Classroom **tworzy dla Ciebie nowe, prywatne repozytorium** i **kopiuje do niego** zawartość template.

## Jak sprawdzić, czy wszystko się zapisało?

1. Wejdź na stronę swojego repo na GitHubie.
2. Otwórz zakładkę **Commits** i sprawdź, czy widać Twój ostatni commit (z Twoim opisem i datą).
3. Kliknij plik `report.ipynb` i upewnij się, że zawiera Twoje zmiany.
4. Jeśli ostatni commit jest widoczny, a plik ma Twoje poprawki — wszystko jest zapisane poprawnie ✅

---

## Jeśli widzisz „Repository Access Issue” (naprawa)

1. Wróć do linku z Moodle (GitHub Classroom) i jeszcze raz kliknij **Accept assignment** / **Go to your repository**.
2. Jeśli pojawi się ekran z prośbą o dostęp/zaproszenie — kliknij **Accept**.
3. Odśwież stronę repo (F5) i spróbuj wejść ponownie.

Jeśli dalej nie działa: zrób zrzut ekranu komunikatu i wyślij prowadzącej.
