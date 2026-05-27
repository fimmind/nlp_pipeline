# Assisted Reader UI Design

## 1. Product direction

The app should be a **reader first** and a vocabulary-learning system second.

The default experience must be calm, text-centered, and distraction-free. Vocabulary support should appear only when it helps the user continue reading with less friction.

The product should feel closer to a clean ebook reader than to a quiz or gamified learning app.

Core hierarchy:

1. Read comfortably.
2. Understand the current passage.
3. Prepare for upcoming sections.
4. Update the learner model with minimal friction.

---

## 2. Design principles

### Reader-first

The book text is the primary object on the page. Everything else is secondary.

Do not make vocabulary cards, quizzes, charts, or model outputs visually dominate the reader.

### Quiet assistance

Assistance should be easy to access, easy to dismiss, and never feel like an interruption.

### Local and contextual

Unknown-word support should be tied to the current paragraph, current chapter, or upcoming reading section.

### Minimal but complete

The interface should expose all required functionality without becoming a dashboard.

### Modular implementation

UI components must not contain algorithmic logic directly. They should call functions from dedicated algorithm modules.

---

## 3. Main app surfaces

The app has four primary surfaces:

1. **First-run onboarding**
2. **Library / profile home**
3. **Initial vocabulary quiz**
4. **Reader view**

---

## 4. First-run onboarding

Show onboarding only when no profile exists in localStorage.

The onboarding flow should be linear and low-friction:

1. Welcome
2. Create profile
3. Open embedded book
4. Take initial vocabulary quiz
5. Start reading

### Welcome screen

Use a centered card with a short explanation.

Suggested copy:

> Read real books with quiet vocabulary assistance.  
> Create a profile, take a short vocabulary check, and start reading with help only where you need it.

Primary CTA:

> Create profile

Secondary CTA:

> Continue as guest

Only show guest mode if it still persists locally.

### Create profile screen

Fields:

- Profile name
- Optional native language / preferred definition language
- Optional reading-assistance preference

Keep this screen sparse.

Suggested copy:

> Your profile stores reading progress and vocabulary estimates locally in this browser.

### Initial quiz intro

Explain that the quiz calibrates assistance.

Suggested copy:

> This short vocabulary check helps the reader decide which words to explain. Choose “Known” only if you recognize the word.

CTA:

> Start vocabulary check

---

## 5. Library / profile home

The library page appears after onboarding.

It should contain:

- Active profile selector
- Current book card
- Reading progress
- Vocabulary estimate summary
- Continue reading button
- Small profile/settings controls

Avoid a dense dashboard.

### Layout

~~~text
┌──────────────────────────────────────────────┐
│ Assisted Reader                              │
│ Profile: Anna                                │
├──────────────────────────────────────────────┤
│                                              │
│  Current book                                │
│  The Wind in the Willows                     │
│  Chapter 2 · 34%                             │
│                                              │
│  [Continue reading]                          │
│                                              │
│  Vocabulary assistance: Quiet                │
│  Estimated support level: Intermediate       │
│                                              │
└──────────────────────────────────────────────┘
~~~

### Empty states

#### No profile

> Create your reading profile  
> Your profile stores reading progress and vocabulary estimates locally in this browser.

CTA:

> Create profile

#### Quiz not completed

> Take a short vocabulary check  
> This helps the reader decide which words to explain.

CTA:

> Start vocabulary check

---

## 6. Reader view

The reader is the central product surface.

It contains:

- Top reader bar
- Main reading column
- Optional assistance rail on desktop
- Paragraph-level assistance cards on mobile
- Reading progress and navigation controls

---

## 7. Desktop reader layout

Use a centered reader with an optional right-side assistance rail.

~~~text
┌──────────────────────────────────────────────────────────────┐
│ Top bar: book title | chapter | progress | assistance toggle │
├──────────────────────────────────────────────────────────────┤
│                                                              │
│        Main reading column             Assistance rail        │
│        680–760px wide                  300–360px wide         │
│                                                              │
│        Paragraph text                  Definition cards       │
│        Subtle highlights               Unknown-word cards     │
│                                                              │
└──────────────────────────────────────────────────────────────┘
~~~

Recommended dimensions:

- Reading column: `680px–760px`
- Assistance rail: `300px–360px`
- Page max width: `1180px–1240px`
- Reader font size: default `18px`
- Reader font size range: `16px–22px`
- Line height: `1.65–1.8`
- Paragraph spacing: `1.1em–1.4em`

The assistance rail should never make the reading column feel cramped.

---

## 8. Mobile reader layout

Use a single-column layout.

Definitions should appear below the corresponding paragraph.

~~~text
Paragraph text...

[Word card 1]
[Word card 2]

Next paragraph...
~~~

If a paragraph contains too many predicted unknown words, show only the top 2 or 3 by default.

Then show a collapsed control:

> 3 more possible unknown words

Do not expand extra assistance automatically.

---

## 9. Top reader bar

The top reader bar should be sticky but visually light.

### Desktop contents

Left:

- Back to library
- Book title
- Chapter selector

Center:

- Reading progress, for example: `Chapter 3 · 42%`

Right:

- Assistance mode toggle
- Prepare words button
- Reader settings
- Profile menu

### Mobile behavior

Collapse secondary controls into a menu.

Visible mobile controls:

- Back
- Book title or chapter title
- Assistance toggle
- Menu button

### Scroll behavior

When scrolling down, the top bar may shrink or fade slightly.

When scrolling up, it should reappear.

---

## 10. Assistance modes

The app should support three assistance modes.

### Off

Pure reading.

No automatic highlights.

No automatic definition cards.

Clicking a word may still open a lookup popover if that feature already exists, but the page itself should look like a normal reader.

### Quiet

Subtle highlights for predicted unknown words.

Definitions appear only when the user clicks or taps a highlighted word.

This should be the default mode after onboarding.

### Assisted

The app automatically shows definition cards for the most important predicted unknown words in each paragraph.

Desktop:

- Cards appear in the right-side assistance rail.
- Cards should align roughly with the paragraph that triggered them.

Mobile:

- Cards appear below the corresponding paragraph.

In assisted mode, show at most 2–3 automatic cards per paragraph.

---

## 11. Word highlighting

Use restrained visual treatment.

| State | Visual treatment |
|---|---|
| Predicted unknown | Soft dotted underline |
| High-priority unknown | Subtle warm background |
| Marked known | No highlight |
| Marked unknown | Slightly stronger underline |
| Proper noun / excluded | No highlight |
| User-hidden | No highlight |

Unknown words should not look like errors. Avoid aggressive red/orange styling.

---

## 12. Definition cards

Definition cards should be compact and skimmable.

### Card contents

~~~text
word
/pronunciation/ · part of speech

Short definition or translation

Example:
"... sentence from the book with the word highlighted ..."

[Known] [Still unknown]
~~~

### Optional secondary actions

- Hide for this book
- More examples
- Add to review

### Card rules

- Prefer short definitions over long dictionary entries.
- Prefer examples from the current book.
- Do not show unrelated example sentences by default.
- Do not show more than 2–3 cards per paragraph automatically.
- Allow the user to mark a word as known or still unknown immediately.

---

## 13. Prepare words feature

The reader should include a **Prepare words** button.

This opens a modal or side sheet.

Title:

> Prepare vocabulary

The user chooses a scope:

- Next section
- Current chapter
- Entire book

Then the app shows a ranked list of recommended words.

### Recommended word card

~~~text
word · POS
Likely useful for the next chapter

Frequency in scope: 12 occurrences
Estimated knowledge: 34%

Best sentence:
"... sentence from the text ..."

[Mark known] [Learn] [Hide]
~~~

### Explanation copy

Show a short explanation:

> These words are selected because they occur often and appear in sentences where they are likely to be the main unfamiliar word.

### Ranking logic

Keep this scoring logic configurable and separate from UI components.

A reasonable initial scoring model:

~~~text
recommendation_score =
  0.35 * normalized_scope_frequency
+ 0.30 * one_unknown_sentence_score
+ 0.20 * predicted_unknown_probability
+ 0.10 * local_proximity_score
+ 0.05 * model_uncertainty
~~~

Definitions:

- `normalized_scope_frequency`: how often the word appears in the selected scope.
- `one_unknown_sentence_score`: how many good sentences exist where this is likely the only unknown word.
- `predicted_unknown_probability`: probability that the user does not know the word.
- `local_proximity_score`: whether the word appears soon.
- `model_uncertainty`: how informative the word is for improving the learner model.

The weights should live in `algorithms/recommendations.ts`.

---

## 14. Initial vocabulary quiz

The quiz should feel fast and non-punitive.

### Layout

~~~text
Vocabulary check

Help us estimate which words you already know.
No need to be perfect. Choose "Known" only if you recognize the word.

        abandon

[Unknown] [Known]

Progress: 12 / 30
~~~

### Interactions

- Keyboard shortcut `K`: Known
- Keyboard shortcut `U`: Unknown
- Large buttons on mobile
- No leaderboard
- No score animation
- No “wrong answer” feedback

### Completion screen

Suggested copy:

> Profile ready  
> We will show help for words that are likely to interrupt comprehension.

CTA:

> Start reading

---

## 15. Profile and localStorage design

The app must be standalone and static.

There is no backend.

All profiles, learner models, reading progress, and settings are stored in localStorage.

Use versioned localStorage keys:

~~~text
assisted_reader:v1:profiles
assisted_reader:v1:active_profile_id
assisted_reader:v1:books
assisted_reader:v1:settings
~~~

Add migration scaffolding from the beginning.

### Profile type

~~~ts
type ReaderProfile = {
  id: string;
  name: string;
  createdAt: string;
  lastActiveAt: string;
  quizCompleted: boolean;
  settings: ReaderSettings;
  learnerModel: LearnerModelState;
  bookStates: Record<string, BookReadingState>;
};
~~~

### Reader settings type

~~~ts
type ReaderSettings = {
  assistanceMode: "off" | "quiet" | "assisted";
  definitionLanguage: string;
  showTranslations: boolean;
  showDefinitions: boolean;
  fontSize: number;
  lineHeight: number;
  theme: "light" | "sepia" | "dark";
};
~~~

### Book state type

~~~ts
type BookReadingState = {
  bookId: string;
  currentLocation: string;
  currentChapterId: string;
  progressRatio: number;
  knownOverrides: Record<string, boolean>;
  hiddenWords: string[];
};
~~~

---

## 16. Suggested component structure

Keep UI and algorithmic code separate.

~~~text
src/
  app/
    App.tsx
    routes.tsx

  components/
    layout/
      AppShell.tsx
      TopReaderBar.tsx
      SidePanel.tsx

    onboarding/
      WelcomeScreen.tsx
      ProfileCreate.tsx
      InitialQuizIntro.tsx

    quiz/
      QuizView.tsx
      QuizCard.tsx
      QuizProgress.tsx

    reader/
      ReaderView.tsx
      ReaderParagraph.tsx
      WordHighlight.tsx
      AssistanceRail.tsx
      DefinitionCard.tsx
      ReaderSettingsPopover.tsx

    recommendations/
      PrepareWordsDialog.tsx
      ScopeSelector.tsx
      RecommendedWordCard.tsx

    library/
      LibraryView.tsx
      BookCard.tsx
      ProfileSwitcher.tsx

  algorithms/
    groupedResidualIrt.ts
    quizSelection.ts
    tokenization.ts
    properNouns.ts
    recommendations.ts
    learnerModel.ts

  data/
    books.ts
    embeddedBookData.ts
    wordMetadata.ts

  storage/
    localStorageStore.ts
    migrations.ts

  styles/
    tokens.css
    reader.css
~~~

The UI may import from `algorithms/`.

The algorithm modules should not import UI components.

---

## 17. Visual design system

Use a minimal, warm, modern design.

### Light theme

~~~text
Background:       #FAF8F3
Surface:          #FFFFFF
Text primary:     #1F2933
Text secondary:   #6B7280
Border:           #E5E0D8
Accent:           #3B6F6A
Accent soft:      #E4F0EE
Warning soft:     #FFF2CC
~~~

### Dark theme

~~~text
Background:       #111315
Surface:          #181B1F
Text primary:     #ECECEC
Text secondary:   #A8ADB4
Border:           #2A2F35
Accent:           #7DBDB4
Accent soft:      #1E3532
~~~

### Sepia theme

~~~text
Background:       #F4ECD8
Surface:          #FFF8E8
Text primary:     #2D2418
Text secondary:   #76664E
Border:           #E1D3B4
Accent:           #6E7051
~~~

### Typography

Use a reader-friendly serif for book text and a clean sans-serif for UI.

Recommended:

- UI: system sans-serif
- Reader: Georgia, Charter, Literata, or a safe serif fallback

~~~css
.reader-text {
  font-family: Georgia, "Times New Roman", serif;
  font-size: var(--reader-font-size);
  line-height: var(--reader-line-height);
}

.ui {
  font-family: Inter, ui-sans-serif, system-ui, -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif;
}
~~~

### Shape and spacing

- Border radius: `14px–18px`
- Cards: soft border, very light shadow
- Buttons: quiet and flat
- Avoid glossy effects
- Avoid heavy gradients
- Use generous whitespace

---

## 18. Accessibility requirements

The app must support:

- Full keyboard navigation
- Large tap targets on mobile
- High contrast text
- `aria-label` for icon buttons
- No meaning conveyed by color alone
- `prefers-reduced-motion`
- Reader font size controls
- No hover-only interactions

On mobile, definition cards should appear immediately after the relevant paragraph in DOM order.

---

## 19. Implementation priorities

Build in this order:

1. Static layout shell
2. Profile creation and localStorage persistence
3. Initial quiz UI connected to existing quiz-selection algorithm
4. Reader view with clean book rendering
5. Assistance mode toggle
6. Paragraph-level unknown-word detection and definition cards
7. Prepare-words dialog
8. Reader settings
9. Mobile polish
10. Accessibility pass

---

## 20. Acceptance checklist

The implementation is complete when:

- A first-time user sees onboarding.
- A user can create a profile.
- Profile data persists after reload.
- The initial quiz runs.
- Quiz responses update the learner model.
- The reader opens after the quiz.
- Reading progress persists.
- Assistance modes work: off, quiet, assisted.
- Definition cards appear near relevant paragraphs.
- Known / Still unknown buttons update the learner model.
- Proper nouns are excluded from recommendations.
- Prepare-words dialog returns ranked recommendations.
- Recommendations include example sentences from the book.
- Desktop layout has a clean assistance rail.
- Mobile layout places cards below paragraphs.
- The app works as a standalone static site.
- No backend is required.

---

## 21. Implementation prompt for the coding agent

~~~text
Redesign the existing `vocab_test_site/` into a standalone static assisted reader app.

The existing implementation is primarily a vocabulary test. Preserve and reuse its algorithmic parts, but redesign the product around reading. The app should feel like a clean modern reader with optional language-learning assistance.

Core requirements:

1. Reader-first UX
   - The main screen must be a comfortable book reader.
   - Learning support must be optional, contextual, and unobtrusive.
   - Avoid gamification-heavy UI.

2. First-run onboarding
   - If no profile exists in localStorage, show a guided flow:
     a. welcome
     b. create profile
     c. select/open the embedded book
     d. take initial vocabulary quiz
     e. start reading
   - Profiles must persist in localStorage.

3. Reader view
   - Desktop: centered reading column with optional right-side assistance rail.
   - Mobile: single-column layout, with assistance cards below relevant paragraphs.
   - Include top reader bar with book title, chapter, progress, assistance mode, prepare-words button, reader settings, and profile menu.

4. Assistance modes
   - Off: pure reading.
   - Quiet: subtle highlights only; definitions appear on click/tap.
   - Assisted: automatically show cards for the most important predicted unknown words.
   - In any paragraph, show at most 2–3 automatic cards by default.

5. Definition cards
   Each card should include:
   - word
   - pronunciation if available
   - part of speech if available
   - short definition and/or translation
   - example sentence from the book
   - Known / Still unknown buttons

6. Prepare-words feature
   - Add a “Prepare words” button.
   - Let the user choose scope: next section, current chapter, or entire book.
   - Show ranked word recommendations with example sentences.
   - Ranking should balance:
     a. high frequency in selected scope
     b. availability of good example sentences where this word is likely the only unknown word
     c. probability that the user does not know the word
     d. proximity in the upcoming text
     e. model uncertainty
   - Keep scoring weights configurable in `algorithms/recommendations.ts`.

7. Preserve algorithmic functionality from `vocab_test_site/`
   Reuse or adapt:
   - grouped residual IRT for knowledge prediction
   - quiz construction by uncertainty minimization with light randomization
   - text tokenization and sentence splitting
   - proper noun detection and exclusion
   - known/unknown updates from user actions

8. Static standalone constraint
   - The app must run as a standalone static site.
   - No backend.
   - No server persistence.
   - All profile and progress data stored in localStorage.
   - Use versioned localStorage keys and migration scaffolding.

9. Code organization
   Keep UI and algorithms separate:
   - `components/reader`
   - `components/quiz`
   - `components/onboarding`
   - `components/recommendations`
   - `algorithms/`
   - `storage/`
   - `styles/`

10. Visual design
   - Minimal, warm, modern.
   - Light/sepia/dark reader themes.
   - Generous typography.
   - Soft cards, subtle borders, restrained accent color.
   - No cluttered dashboards.

Deliver a fully functional implementation, not just static mockups. After implementation, verify:
- new user onboarding works
- profile persists after reload
- quiz updates learner model
- reader opens after quiz
- assistance modes work
- definition cards can mark words known/unknown
- prepare-words dialog returns ranked recommendations
- layout works on desktop and mobile
```
~~~
