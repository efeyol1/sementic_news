const API_BASE = process.env.NEXT_PUBLIC_API_URL || "http://localhost:8000";

export interface SentimentCounts {
  positive: number;
  negative: number;
  neutral?: number;
}

export interface SentimentPercentages {
  positive: number;
  negative: number;
  neutral?: number;
}

export interface TodayResponse {
  date: string;
  total_items: number;
  turkish_items: number;
  sources: Record<string, number>;
  sentiment: {
    counts: SentimentCounts;
    percentages: SentimentPercentages;
  };
  top_entities: { PER: string[]; ORG: string[]; LOC: string[] };
  cluster_count: number;
  top_clusters: { cluster_id: number; title: string; size: number; keywords: string[] }[];
  available_dates: string[];
}

export interface TopicResponse {
  date: string;
  cluster_id: number;
  keywords: string[];
  size: number;
  sentiment_distribution: SentimentCounts;
  news: NewsItem[];
}

export interface NewsItem {
  title: string;
  source_name: string;
  published_date: string;
  link?: string;
  sentiment_label: string;
  sentiment_score: number;
  entities: { PER: string[]; ORG: string[]; LOC: string[] };
}

export interface SourceStats {
  total: number;
  sentiment_counts: SentimentCounts;
  sentiment_percentages: SentimentPercentages; // 0–100 (not 0–1)
  avg_confidence: number;
}

export interface SourceComparison {
  date: string;
  sources: Record<string, SourceStats>;
}

export interface SimilarNewsItem {
  title: string;
  source_name: string;
  date: string;
  sentiment_label: string;
  link: string | null;
  similarity: number;
}

export interface SimilarNewsResponse {
  query_title: string;
  results: SimilarNewsItem[];
}

async function apiFetch<T>(path: string, params?: Record<string, string>): Promise<T> {
  const url = new URL(`${API_BASE}${path}`);
  if (params) {
    Object.entries(params).forEach(([k, v]) => url.searchParams.set(k, v));
  }
  const res = await fetch(url.toString(), { next: { revalidate: 300 } });
  if (!res.ok) throw new Error(`API error ${res.status}: ${path}`);
  return res.json();
}

// Default country — Phase 6 introduces multi-country selection; until a
// user picks one (or until Phase 7 ships Germany), every call defaults to
// the V1 Turkey pilot so existing bookmarks and the no-param case keep
// returning the same data as before.
export const DEFAULT_COUNTRY = "turkey";

export interface CountryInfo {
  code: string;     // "TR"
  slug: string;     // "turkey"
  name: string;     // "Turkey"
  language: string; // "tr"
  status: string;   // "active"
}

function withCountry(
  country: string | undefined,
  params?: Record<string, string>,
): Record<string, string> {
  return { ...(params ?? {}), country: country || DEFAULT_COUNTRY };
}

export interface DatesResponse {
  dates: string[];
}

export interface TrendPoint {
  date: string;
  positive: number;
  negative: number;
  neutral?: number;
  total: number;
  positive_pct: number;
}

export interface TrendResponse {
  points: TrendPoint[];
}

export const api = {
  today: (date?: string, country?: string) =>
    apiFetch<TodayResponse>("/api/today", withCountry(country, date ? { date } : undefined)),

  topic: (id: number, date?: string, country?: string) =>
    apiFetch<TopicResponse>(`/api/topic/${id}`, withCountry(country, date ? { date } : undefined)),

  sources: (date?: string, country?: string) =>
    apiFetch<SourceComparison>("/api/source-comparison", withCountry(country, date ? { date } : undefined)),

  similar: (q: string, n = 5, country?: string) =>
    apiFetch<SimilarNewsResponse>("/api/similar", withCountry(country, { q, n: String(n) })),

  dates: (country?: string) =>
    apiFetch<DatesResponse>("/api/dates", withCountry(country)),

  trend: (days = 30, country?: string) =>
    apiFetch<TrendResponse>("/api/trend", withCountry(country, { days: String(days) })),

  countries: () => apiFetch<CountryInfo[]>("/api/countries"),
};

export function todayDate(): string {
  // toISOString() returns UTC, which lags Türkiye by 3 hours and would point
  // the dashboard at "tomorrow" or a not-yet-ingested day around midnight TR.
  // en-CA locale produces YYYY-MM-DD; Europe/Istanbul keeps it in local time.
  return new Date().toLocaleDateString("en-CA", { timeZone: "Europe/Istanbul" });
}
