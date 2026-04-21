const API_BASE = process.env.NEXT_PUBLIC_API_URL || "http://localhost:8000";

export interface SentimentCounts {
  positive: number;
  negative: number;
}

export interface SentimentPercentages {
  positive: number;
  negative: number;
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

async function apiFetch<T>(path: string, params?: Record<string, string>): Promise<T> {
  const url = new URL(`${API_BASE}${path}`);
  if (params) {
    Object.entries(params).forEach(([k, v]) => url.searchParams.set(k, v));
  }
  const res = await fetch(url.toString(), { next: { revalidate: 300 } });
  if (!res.ok) throw new Error(`API error ${res.status}: ${path}`);
  return res.json();
}

export const api = {
  today: (date?: string) =>
    apiFetch<TodayResponse>("/api/today", date ? { date } : undefined),

  topic: (id: number, date?: string) =>
    apiFetch<TopicResponse>(`/api/topic/${id}`, date ? { date } : undefined),

  sources: (date?: string) =>
    apiFetch<SourceComparison>("/api/source-comparison", date ? { date } : undefined),
};

export function todayDate(): string {
  return new Date().toISOString().split("T")[0];
}
