import { clsx, type ClassValue } from "clsx";
import { twMerge } from "tailwind-merge";

export function cn(...inputs: ClassValue[]) {
  return twMerge(clsx(inputs));
}

export function sentimentColor(label: string): string {
  if (label === "positive") return "#10B981";
  if (label === "negative") return "#EF4444";
  return "#F59E0B";
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
