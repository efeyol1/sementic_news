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

export function formatNumber(n: number): string {
  return n.toLocaleString("tr-TR");
}

export function formatPercent(n: number): string {
  return `%${(n * 100).toFixed(1)}`;
}
