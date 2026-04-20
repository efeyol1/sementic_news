import { Suspense } from "react";
import Link from "next/link";
import { api, todayDate } from "@/lib/api";
import { SentimentGauge } from "@/components/ui/SentimentGauge";
import { EntityCloud } from "@/components/ui/EntityCloud";
import { ArrowLeft, ExternalLink, Layers, Hash } from "lucide-react";
import { sentimentColor, sentimentLabel } from "@/lib/utils";

interface Props {
  params: { id: string };
  searchParams: { date?: string };
}

async function TopicContent({ id, date }: { id: number; date: string }) {
  const data = await api.topic(id, date).catch(() => null);

  if (!data) {
    return (
      <div className="flex flex-col items-center justify-center min-h-[60vh] gap-4">
        <Layers className="w-12 h-12 text-text-muted" />
        <p className="text-text-secondary">Küme #{id} bulunamadı.</p>
        <Link href="/" className="text-accent-glow text-sm hover:underline">
          ← Ana sayfaya dön
        </Link>
      </div>
    );
  }

  const entityMap = data.news.reduce(
    (acc, n) => {
      n.entities.PER.forEach((e) => acc.PER.add(e));
      n.entities.ORG.forEach((e) => acc.ORG.add(e));
      n.entities.LOC.forEach((e) => acc.LOC.add(e));
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
    <div className="space-y-8">
      {/* Back + header */}
      <div>
        <Link
          href={`/?date=${date}`}
          className="inline-flex items-center gap-1.5 text-sm text-text-secondary hover:text-text-primary transition-colors mb-4"
        >
          <ArrowLeft className="w-4 h-4" />
          Genel bakışa dön
        </Link>
        <div className="flex items-start justify-between">
          <div className="flex items-center gap-3 mb-2">
            <div className="w-10 h-10 rounded-xl bg-accent-muted border border-border-glow flex items-center justify-center">
              <Hash className="w-5 h-5 text-accent-glow" />
            </div>
            <div>
              <div className="text-xs text-text-muted">Küme #{id}</div>
              <h1 className="text-2xl font-bold text-text-primary">
                {data.keywords.slice(0, 3).join(" · ")}
              </h1>
            </div>
          </div>
          <div className="glass-card px-4 py-2 text-center">
            <div className="text-2xl font-bold font-mono text-text-primary">{data.size}</div>
            <div className="text-xs text-text-muted">haber</div>
          </div>
        </div>

        <div className="flex flex-wrap gap-2 mt-4">
          {data.keywords.map((kw) => (
            <span key={kw} className="keyword-chip">
              {kw}
            </span>
          ))}
        </div>
      </div>

      {/* Content grid */}
      <div className="grid grid-cols-1 lg:grid-cols-3 gap-6">
        <div className="glass-card p-6">
          <h2 className="text-base font-semibold text-text-primary mb-4">Duygu Dağılımı</h2>
          <SentimentGauge counts={data.sentiment_distribution} />
        </div>
        <div className="lg:col-span-2 glass-card p-6">
          <h2 className="text-base font-semibold text-text-primary mb-4">Öne Çıkan Varlıklar</h2>
          <EntityCloud entities={entities} />
        </div>
      </div>

      {/* News list */}
      <div className="glass-card p-6">
        <h2 className="text-base font-semibold text-text-primary mb-4">
          Bu Kümeden Haberler
          <span className="text-text-muted font-normal text-sm ml-2">({data.news.length})</span>
        </h2>
        <div className="space-y-3">
          {data.news.slice(0, 30).map((item, i) => (
            <div
              key={i}
              className="flex items-start gap-4 p-3.5 rounded-xl hover:bg-bg-hover/30 transition-colors group"
            >
              <div
                className="w-1 min-h-[40px] rounded-full shrink-0 mt-1"
                style={{ background: sentimentColor(item.sentiment_label) }}
              />
              <div className="flex-1 min-w-0">
                {item.link ? (
                  <a
                    href={item.link}
                    target="_blank"
                    rel="noopener noreferrer"
                    className="font-medium text-text-primary hover:text-accent-glow transition-colors line-clamp-2 flex items-start gap-2"
                  >
                    {item.title}
                    <ExternalLink className="w-3 h-3 text-text-muted shrink-0 mt-0.5 opacity-0 group-hover:opacity-100 transition-opacity" />
                  </a>
                ) : (
                  <p className="font-medium text-text-primary line-clamp-2">{item.title}</p>
                )}
                <div className="flex items-center gap-3 mt-1.5">
                  <span className="text-xs text-text-muted">{item.source_name}</span>
                  <span className="text-xs text-text-muted">·</span>
                  <span
                    className="text-xs font-medium"
                    style={{ color: sentimentColor(item.sentiment_label) }}
                  >
                    {sentimentLabel(item.sentiment_label)} ({(item.sentiment_score * 100).toFixed(0)}%)
                  </span>
                </div>
              </div>
            </div>
          ))}
        </div>
      </div>
    </div>
  );
}

export default function TopicPage({ params, searchParams }: Props) {
  const id = parseInt(params.id, 10);
  const date = searchParams.date || todayDate();

  return (
    <div className="max-w-7xl mx-auto px-6 py-8">
      <Suspense
        fallback={
          <div className="animate-pulse space-y-4">
            <div className="h-8 w-64 bg-bg-card rounded-xl" />
            <div className="h-64 bg-bg-card rounded-2xl" />
          </div>
        }
      >
        <TopicContent id={id} date={date} />
      </Suspense>
    </div>
  );
}
