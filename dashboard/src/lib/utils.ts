import { clsx, type ClassValue } from "clsx";
import { twMerge } from "tailwind-merge";

export function cn(...inputs: ClassValue[]) {
  return twMerge(clsx(inputs));
}

export function sentimentColor(label: string): string {
  if (label === "positive") return "#16a34a";
  if (label === "negative") return "#dc2626";
  return "#d97706";
}

export function sentimentLabel(label: string): string {
  if (label === "positive") return "Pozitif";
  if (label === "negative") return "Negatif";
  return "Nötr";
}

// Sprint 7.5: locale + timezone helpers — frontend pulls language /
// timezone off /api/countries instead of inheriting the legacy
// Europe/Istanbul + tr-TR defaults baked into the V1 Turkey pilot.

const LANGUAGE_TO_LOCALE: Record<string, string> = {
  tr: "tr-TR",
  de: "de-DE",
  fr: "fr-FR",
  it: "it-IT",
  es: "es-ES",
  en: "en-GB",
};

/** Map a country's `language` code ("tr", "de", …) to a BCP-47 locale. */
export function localeForLanguage(language: string | undefined): string {
  if (!language) return "en-US";
  return LANGUAGE_TO_LOCALE[language] ?? `${language}-${language.toUpperCase()}`;
}

export function formatNumber(n: number, locale: string = "en-US"): string {
  return n.toLocaleString(locale);
}

export function formatPercent(n: number): string {
  return `%${(n * 100).toFixed(1)}`;
}
