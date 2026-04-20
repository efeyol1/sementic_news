import { Suspense } from "react";
import { api, todayDate } from "@/lib/api";
import { StatCard } from "@/components/ui/StatCard";
import { SentimentGauge } from "@/components/ui/SentimentGauge";
import { EntityCloud } from "@/components/ui/EntityCloud";
import { ClusterGrid } from "@/components/ui/ClusterGrid";
import { SentimentPieChart } from "@/components/charts/SentimentPieChart";
import { Globe2, Calendar } from "lucide-react";

interface Props {
  searchParams: { date?: string };
}

async function DashboardContent({ date }: { date: string }) {
  const data = await api.today(date).catch(() => null);

  if (!data) {
    return (
      <div className="flex flex-col items-center justify-center min-h-[60vh] gap-4">
        <div className="w-16 h-16 rounded-2xl bg-bg-card border border-border-subtle flex items-center justify-center">
          <Globe2 className="w-8 h-8 text-text-muted" />
        </div>
        <div className="text-center">
          <h2 className="text-xl font-semibold text-text-primary mb-2">Veri bulunamadı</h2>
          <p className="text-text-secondary text-sm">{date} tarihi için analiz henüz hazır değil.</p>
        </div>
      </div>
    );
  }

  const positivePct = data.sentiment.percentages.positive ?? 0;
  const negativePct = data.sentiment.percentages.negative ?? 0;
  const dominantLabel = negativePct > positivePct ? "Ağırlıklı Negatif" : "Ağırlıklı Pozitif";
  const dominantClass = negativePct > positivePct ? "negative-badge" : "positive-badge";

  return (
    <div className="space-y-8">
      {/* Header */}
      <div className="flex items-start justify-between">
        <div>
          <h1 className="text-3xl font-bold gradient-text mb-1">Günlük Analiz</h1>
          <div className="flex items-center gap-2 text-text-secondary text-sm">
            <Calendar className="w-4 h-4" />
            <span>
              {new Date(date).toLocaleDateString("tr-TR", {
                weekday: "long",
                year: "numeric",
                month: "long",
                day: "numeric",
              })}
            </span>
          </div>
        </div>
        <div className="glass-card px-4 py-2 flex items-center gap-2">
          <div className="w-2 h-2 rounded-full bg-sentiment-positive animate-pulse" />
          <span className="text-xs text-text-secondary">Pipeline güncel</span>
        </div>
      </div>

      {/* Stat cards */}
      <div className="grid grid-cols-2 lg:grid-cols-4 gap-4">
        <StatCard label="Toplam Haber" value={data.total_items} iconName="Newspaper" delay={0} />
        <StatCard label="Türkçe Haber" value={data.turkish_items} iconName="Globe2" delay={80} />
        <StatCard
          label="Pozitif Oran"
          value={positivePct}
          suffix="%"
          decimals={1}
          iconName="TrendingUp"
          accent="positive"
          delay={160}
        />
        <StatCard
          label="Konu Kümesi"
          value={data.cluster_count}
          iconName="Layers"
          delay={240}
        />
      </div>

      {/* Main grid */}
      <div className="grid grid-cols-1 lg:grid-cols-3 gap-6">
        <div className="lg:col-span-2 glass-card p-6">
          <div className="flex items-center justify-between mb-6">
            <div>
              <h2 className="text-lg font-semibold text-text-primary">Duygu Analizi</h2>
              <p className="text-sm text-text-secondary mt-0.5">Haberlerin sentiment dağılımı</p>
            </div>
            <span className={dominantClass}>{dominantLabel}</span>
          </div>
          <div className="grid grid-cols-1 md:grid-cols-2 gap-6 items-center">
            <SentimentPieChart counts={data.sentiment.counts} />
            <SentimentGauge counts={data.sentiment.counts} />
          </div>
        </div>

        <div className="glass-card p-6">
          <h2 className="text-lg font-semibold text-text-primary mb-1">Öne Çıkan Varlıklar</h2>
          <p className="text-sm text-text-secondary mb-6">Kişi, kurum ve coğrafi isimler</p>
          <EntityCloud entities={data.top_entities} />
        </div>
      </div>

      {/* Clusters */}
      <div>
        <div className="flex items-center justify-between mb-4">
          <div>
            <h2 className="text-lg font-semibold text-text-primary">Konu Kümeleri</h2>
            <p className="text-sm text-text-secondary">Haberlerin otomatik gruplandırması</p>
          </div>
          <span className="text-xs text-text-muted">{data.top_clusters.length} küme</span>
        </div>
        <ClusterGrid clusters={data.top_clusters} date={date} />
      </div>
    </div>
  );
}

export default function HomePage({ searchParams }: Props) {
  const date = searchParams.date || todayDate();

  return (
    <div className="max-w-7xl mx-auto px-6 py-8">
      <div
        className="fixed top-0 left-1/2 -translate-x-1/2 w-[800px] h-[400px] pointer-events-none"
        style={{
          background:
            "radial-gradient(ellipse at center top, rgba(99, 102, 241, 0.08) 0%, transparent 70%)",
        }}
      />
      <Suspense fallback={<DashboardSkeleton />}>
        <DashboardContent date={date} />
      </Suspense>
    </div>
  );
}

function DashboardSkeleton() {
  return (
    <div className="space-y-8 animate-pulse">
      <div className="h-10 w-48 bg-bg-card rounded-xl" />
      <div className="grid grid-cols-4 gap-4">
        {Array.from({ length: 4 }).map((_, i) => (
          <div key={i} className="h-28 bg-bg-card rounded-2xl" />
        ))}
      </div>
      <div className="h-64 bg-bg-card rounded-2xl" />
    </div>
  );
}
