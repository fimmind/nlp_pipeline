---
name: Vocabulary Reader Assistant
description: A minimal, book-like reading assistant for vocabulary estimation and guided reading.
colors:
  paper-light: "#F4EFE7"
  surface-light: "#FCFAF6"
  text-light: "#23201D"
  text-muted-light: "#5D554D"
  border-light: "#CFC4B6"
  accent: "#2C2824"
  accent-hover: "#12100F"
  success: "#2C2824"
  danger: "#2C2824"
  paper-dark: "#1B1815"
  surface-dark: "#24201B"
  text-dark: "#EDE4D8"
  text-muted-dark: "#C4B8A8"
  border-dark: "#4A4036"
typography:
  display:
    fontFamily: "Fraunces, Georgia, serif"
    fontSize: "clamp(2rem, 4.8vw, 3.2rem)"
    fontWeight: 700
    lineHeight: 1.08
    letterSpacing: "normal"
  headline:
    fontFamily: "Fraunces, Georgia, serif"
    fontSize: "clamp(1.3rem, 2.3vw, 1.85rem)"
    fontWeight: 600
    lineHeight: 1.18
    letterSpacing: "normal"
  body:
    fontFamily: "Source Serif 4, Georgia, serif"
    fontSize: "1rem"
    fontWeight: 400
    lineHeight: 1.6
    letterSpacing: "normal"
  label:
    fontFamily: "Source Serif 4, Georgia, serif"
    fontSize: "0.9rem"
    fontWeight: 600
    lineHeight: 1.35
    letterSpacing: "0.02em"
rounded:
  sm: "6px"
  md: "10px"
  lg: "14px"
  pill: "999px"
spacing:
  xs: "6px"
  sm: "10px"
  md: "16px"
  lg: "24px"
  xl: "34px"
components:
  panel:
    backgroundColor: "{colors.surface-light}"
    rounded: "{rounded.lg}"
    padding: "{spacing.md}"
  button-primary:
    backgroundColor: "{colors.accent}"
    textColor: "{colors.surface-light}"
    rounded: "{rounded.sm}"
    padding: "10px 14px"
  button-primary-hover:
    backgroundColor: "{colors.accent-hover}"
    textColor: "{colors.surface-light}"
    rounded: "{rounded.sm}"
    padding: "10px 14px"
  chip:
    backgroundColor: "{colors.surface-light}"
    textColor: "{colors.text-muted-light}"
    rounded: "{rounded.pill}"
    padding: "6px 10px"
---

# Design System: Vocabulary Reader Assistant

## 1. Overview

**Creative North Star: "Quiet Reading Desk"**

The interface should feel like opening a well-designed reading notebook. It is calm, direct, and intentionally plain, with enough hierarchy to guide action and enough restraint to avoid visual noise.

The system favors legibility and trust. Every interactive element should feel useful, not performative. The product is primarily a reader with algorithmic support, so visual emphasis should serve reading decisions, not visual spectacle.

## 2. Colors

Use black-and-white contrast on top of warm paper neutrals, with restrained beige support tones only. Light mode is default; dark mode is optional and equivalent in contrast structure.

### Primary

- **Ink Action** (`#2C2824`): primary actions and interaction focus.

### Secondary

- **Ink Support** (`#2C2824`): completion states without introducing extra hue.

### Tertiary

- **Ink Outline** (`#2C2824`): destructive/reset actions with neutral contrast.

### Neutral

- **Paper Light** (`#F4EFE7`): main background.
- **Surface Light** (`#FCFAF6`): panels/cards.
- **Ink Light** (`#23201D`): primary text.
- **Muted Ink Light** (`#5D554D`): helper text.
- **Line Light** (`#CFC4B6`): borders/dividers.
- **Paper Dark** (`#1B1815`): dark background.
- **Surface Dark** (`#24201B`): dark panels.
- **Ink Dark** (`#EDE4D8`): dark primary text.
- **Muted Ink Dark** (`#C4B8A8`): dark helper text.
- **Line Dark** (`#4A4036`): dark borders/dividers.

## 3. Typography

**Display Font:** Fraunces (fallback Georgia, serif)
**Body Font:** Source Serif 4 (fallback Georgia, serif)
**Label Font:** Source Serif 4 (same family, stronger weight)

Character: classic and readable, with editorial warmth.

### Hierarchy

- **Display** (700, `clamp(2rem, 4.8vw, 3.2rem)`, `1.08`): page-level purpose.
- **Headline** (600, `clamp(1.3rem, 2.3vw, 1.85rem)`, `1.18`): section titles.
- **Body** (400, `1rem`, `1.6`): explanatory and reading-oriented text.
- **Label** (600, `0.9rem`, `1.35`): form labels and metadata.

## 4. Elevation

Depth is subtle. Surfaces are distinguished mostly by paper tone and thin borders, with light shadows used sparingly.

### Shadow Vocabulary

- **Panel Lift** (`0 10px 24px rgba(45, 36, 30, 0.08)`): only for major containers.

## 5. Components

### Buttons

- Compact shape (`6px` radius), text-first clarity.
- Primary uses deep neutral ink; hover shifts to stronger black.
- Destructive action is outlined and calm, not aggressive.

### Chips

- Light, low-contrast context pills for model/book metadata.

### Cards / Panels

- Soft paper surfaces, medium corner radius (`14px`), thin border.
- Medium density spacing, no heavy decorative framing.

### Inputs / Fields

- Simple bordered inputs with visible focus ring.
- Checklist rows provide subtle hover feedback.

### Navigation

- Single-page progressive workflow.

## 6. Do's and Don'ts

### Do:

- **Do** keep visual weight low and text readability high.
- **Do** preserve medium spacing rhythm and clear section hierarchy.
- **Do** keep motion subtle and brief.
- **Do** provide equal readability in light and dark themes.

### Don't:

- **Don't** introduce non-neutral accents, texture overlays, gradients, or visual gimmicks.
- **Don't** make the interface feel like a dashboard showcase.
- **Don't** overload one section with many competing actions.
- **Don't** sacrifice legibility for decoration.
