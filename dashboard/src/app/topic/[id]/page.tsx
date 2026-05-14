import { Suspense } from "react";
import Link from "next/link";
import { api, DEFAULT_COUNTRY, todayDate, type SimilarNewsItem } from "@/lib/api";
import { SentimentGauge } from "@/components/ui/SentimentGauge";
import { EntityCloud } from "@/components/ui/EntityCloud";
import { ArrowLeft, ExternalLink, Hash, Sparkles } from "lucide-react";
import { sentimentColor, sentimentLabel } from "@/lib/utils";

interface Props {
  params: { id: string };
  searchParams: { date?: string; country?: string };
}

async function TopicContent({ id, date, country }: { id: number; date: string; country: string }) {
  const data = await api.topic(id, date, country).catch(() => null);
  const similarData = data
    ? await api.similar(data.keywords.slice(0, 2).join(" "), 4, country).catch(() => null)
    : null;

  if (!data) {
    return (
      <div className="flex flex-col items-center justify-center min-h-[60vh] gap-3">
        <div className="w-12 h-12 rounded-xl bg-white border border-slate-200 flex items-center justify-center">
          <Hash className="w-5 h-5 text-slate-300" />
        </div>
        <p className="text-slate-500 text-sm">Küme #{id} bulunamadı.</p>
        <Link href="/" className="text-green-600 text-sm hover:underline">
          ← Ana sayfaya dön
        </Link>
      </div>
    );
  }

  const entityMap = data.news.reduce(
    (acc, n) => {
      (n.entities?.PER ?? []).forEach((e) => acc.PER.add(e));
      (n.entities?.ORG ?? []).forEach((e) => acc.ORG.add(e));
      (n.entities?.LOC ?? []).forEach((e) => acc.LOC.add(e));
      return acc;
    },
    { PER: new Set<string>(), ORG: new Set<string>(), LOC: new Set<string>() }
  );

  const entities = {
    PER: Array.from(entityMap.PER).slice(0, 10),
    ORG: Array.from(entityMap.ORG).slice(0, 10),
    LOC: Array.from(entityMap.LOC).slice(0, 10),
  };

  return (
    <div className="space-y-7">
      {/* Back + header */}
      <div>
        <Link
          href={`/?date=${date}`}
          className="inline-flex items-center gap-1.5 text-sm text-slate-500 hover:text-slate-800 transition-colors mb-4"
        >
          <ArrowLeft className="w-4 h-4" />
          Genel bakışa dön
        </Link>
        <div className="flex items-start justify-between">
          <div className="flex items-center gap-3">
            <div className="w-10 h-10 rounded-lg bg-green-50 border border-green-100 flex items-center justify-center">
              <Hash className="w-5 h-5 text-green-600" />
            </div>
            <div>
              <div className="text-xs text-slate-400">Küme #{id}</div>
              <h1 className="text-xl font-bold text-slate-900">
                {data.keywords.slice(0, 3).join(" · ")}
              </h1>
            </div>
          </div>
          <div className="card px-4 py-2 text-center">
            <div className="text-xl font-bold font-mono text-slate-800">{data.size}</div>
            <div className="text-xs text-slate-400">haber</div>
          </div>
        </div>

        <div className="flex flex-wrap gap-1.5 mt-4">
          {data.keywords.map((kw) => (
            <span key={kw} className="keyword-chip">{kw}</span>
          ))}
        </div>
      </div>

      {/* Content grid */}
      <div className="grid grid-cols-1 lg:grid-cols-3 gap-5">
        <div className="card p-5">
          <h2 className="text-sm font-semibold text-slate-700 mb-4">Duygu Dağılımı</h2>
          <SentimentGauge counts={data.sentiment_distribution} />
        </div>
        <div className="lg:col-span-2 card p-5">
          <h2 className="text-sm font-semibold text-slate-700 mb-4">Öne Çıkan Varlıklar</h2>
          <EntityCloud entities={entities} />
        </div>
      </div>

      {/* News list */}
      <div className="card p-6">
        <h2 className="text-sm font-semibold text-slate-700 mb-4">
          Bu Kümeden Haberler
          <span className="text-slate-400 font-normal ml-1.5">({data.news.length})</span>
        </h2>
        <div className="divide-y divide-slate-50">
          {data.news.slice(0, 30).map((item, i) => (
            <div key={i} className="flex items-start gap-3 py-3 group">
              <div
                className="w-1 min-h-[36px] rounded-full shrink-0 mt-0.5"
                style={{ background: sentimentColor(item.sentiment_label) }}
              />
              <div className="flex-1 min-w-0">
                {item.link ? (
                  <a
                    href={item.link}
                    target="_blank"
                    rel="noopener noreferrer"
                    className="text-sm font-medium text-slate-700 hover:text-green-700 transition-colors line-clamp-2 flex items-start gap-1.5"
                  >
                    {item.title}
                    <ExternalLink className="w-3 h-3 text-slate-300 shrink-0 mt-0.5 opacity-0 group-hover:opacity-100 transition-opacity" />
                  </a>
                ) : (
                  <p className="text-sm font-medium text-slate-700 line-clamp-2">{item.title}</p>
                )}
                <div className="flex items-center gap-2 mt-1">
                  <span className="text-xs text-slate-400">{item.source_name}</span>
                  <span className="text-xs text-slate-300">·</span>
                  <span className="text-xs font-medium" style={{ color: sentimentColor(item.sentiment_label) }}>
                    {sentimentLabel(item.sentiment_label)}{" "}
                    {item.sentiment_score != null && `(${(item.sentiment_score * 100).toFixed(0)}%)`}
                  </span>
                </div>
              </div>
            </div>
          ))}
        </div>
      </div>

      {/* Similar news */}
      {similarData && similarData.results.length > 0 && (
        <div className="card p-6">
          <div className="flex items-center gap-2 mb-4">
            <Sparkles className="w-4 h-4 text-green-600" />
            <h2 className="text-sm font-semibold text-slate-700">Benzer Haberler</h2>
            <span className="text-xs text-slate-400 ml-1">semantik arama</span>
          </div>
          <div className="divide-y divide-slate-50">
            {similarData.results.map((item: SimilarNewsItem, i: number) => (
              <div key={i} className="flex items-start gap-3 py-3 group">
                <div className="w-8 h-8 rounded-lg bg-green-50 flex items-center justify-center shrink-0 mt-0.5">
                  <span className="text-xs font-mono text-green-600">{(item.similarity * 100).toFixed(0)}%</span>
                </div>
                <div className="flex-1 min-w-0">
                  {item.link ? (
                    <a
                      href={item.link}
                      target="_blank"
                      rel="noopener noreferrer"
                      className="text-sm font-medium text-slate-700 hover:text-green-700 transition-colors line-clamp-2 flex items-start gap-1.5"
                    >
                      {item.title}
                      <ExternalLink className="w-3 h-3 text-slate-300 shrink-0 mt-0.5 opacity-0 group-hover:opacity-100 transition-opacity" />
                    </a>
                  ) : (
                    <p className="text-sm font-medium text-slate-700 line-clamp-2">{item.title}</p>
                  )}
                  <div className="flex items-center gap-2 mt-1">
                    <span className="text-xs text-slate-400">{item.source_name}</span>
                    <span className="text-xs text-slate-300">·</span>
                    <span className="text-xs text-slate-400">{item.date}</span>
                    <span className="text-xs text-slate-300">·</span>
                    <span className="text-xs font-medium" style={{ color: sentimentColor(item.sentiment_label) }}>
                      {sentimentLabel(item.sentiment_label)}
                    </span>
                  </div>
                </div>
              </div>
            ))}
          </div>
        </div>
      )}
    </div>
  );
}

export default function TopicPage({ params, searchParams }: Props) {
  const id = parseInt(params.id, 10);
  const date = searchParams.date || todayDate();
  const country = searchParams.country || DEFAULT_COUNTRY;

  return (
    <div className="max-w-7xl mx-auto px-6 py-8">
      <Suspense
        fallback={
          <div className="animate-pulse space-y-4">
            <div className="h-8 w-60 bg-slate-200 rounded-lg" />
            <div className="h-60 bg-white border border-slate-200 rounded-xl" />
          </div>
        }
      >
        <TopicContent id={id} date={date} country={country} />
      </Suspense>
    </div>
  );
}
