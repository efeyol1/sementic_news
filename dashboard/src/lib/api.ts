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
  calibrated_sentiment_score?: number | null;
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
  timezone: string; // "Europe/Istanbul" — Sprint 7.5
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

// --- Entity profiles (Sprint 8 API / Sprint 9 dashboard) -------------------

export interface Collocate {
  lemma: string;
  pos: string;
  c11_window: number;
  pmi: number;
  llr: number;
}

export interface EntityProfile {
  country_code: string;
  canonical: string;
  entity_type: string; // PER | ORG | LOC
  wikidata_qid: string | null;
  window_days: number;
  end_date: string;
  coverage_days: number;
  total_mentions: number;
  total_cooccurrences: number;
  window_total: number;
  avg_pmi: number | null;
  avg_log_likelihood: number | null;
  top_collocates: Collocate[];
}

export interface EntityDirectoryItem {
  canonical: string;
  wikidata_qid: string | null;
  entity_type: string;
  total_mentions: number;
  total_cooccurrences: number;
  avg_pmi: number | null;
  coverage_days: number;
}

export interface EntityDirectoryResponse {
  country_code: string;
  window_days: number;
  end_date: string | null;
  entities: EntityDirectoryItem[];
}

export interface EntityCompareResponse {
  reference: string;
  window_days: number;
  countries: EntityProfile[];
}

export interface EntityTimelinePoint {
  date: string;
  mention_count: number;
  total_cooccurrences: number;
  avg_pmi: number | null;
  avg_log_likelihood: number | null;
}

export interface EntityTimelineResponse {
  reference: string;
  country_code: string;
  points: EntityTimelinePoint[];
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

  // Entity profile endpoints (Sprint 8). `ref` is a Wikidata QID (Q22686)
  // or a canonical name — encodeURIComponent handles names with spaces.
  entities: (
    opts: { q?: string; entityType?: string; windowDays?: number; limit?: number } = {},
    country?: string,
  ) => {
    const params: Record<string, string> = {};
    if (opts.q) params.q = opts.q;
    if (opts.entityType) params.entity_type = opts.entityType;
    if (opts.windowDays) params.window_days = String(opts.windowDays);
    if (opts.limit) params.limit = String(opts.limit);
    return apiFetch<EntityDirectoryResponse>("/api/entities", withCountry(country, params));
  },

  entityProfile: (ref: string, windowDays = 30, country?: string) =>
    apiFetch<EntityProfile>(
      `/api/entity/${encodeURIComponent(ref)}/profile`,
      withCountry(country, { window_days: String(windowDays) }),
    ),

  // compare is cross-country by nature — no ?country, optional ?countries=DE,FR.
  entityCompare: (ref: string, windowDays = 30, countries?: string[]) => {
    const params: Record<string, string> = { window_days: String(windowDays) };
    if (countries && countries.length) params.countries = countries.join(",");
    return apiFetch<EntityCompareResponse>(
      `/api/entity/${encodeURIComponent(ref)}/compare`,
      params,
    );
  },

  entityTimeline: (ref: string, days = 30, country?: string) =>
    apiFetch<EntityTimelineResponse>(
      `/api/entity/${encodeURIComponent(ref)}/timeline`,
      withCountry(country, { days: String(days) }),
    ),
};

export function todayDate(timeZone: string = "Europe/Istanbul"): string {
  // toISOString() returns UTC, which lags local time by hours and would
  // point the dashboard at "tomorrow" or a not-yet-ingested day around
  // midnight in the user's region. en-CA locale produces YYYY-MM-DD; the
  // caller's `timeZone` keeps it in the active country's local day.
  // Sprint 7.5 made this a parameter so non-TR countries (DE/FR/IT/ES/UK)
  // can pass their own zone — see /api/countries.timezone.
  return new Date().toLocaleDateString("en-CA", { timeZone });
}

/** Look up a country by slug from a /api/countries response. */
export function findCountry(
  countries: CountryInfo[],
  slug: string,
): CountryInfo | undefined {
  return countries.find((c) => c.slug === slug);
}
