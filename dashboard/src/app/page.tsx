import { Suspense } from "react";
import { api, DEFAULT_COUNTRY, findCountry, todayDate } from "@/lib/api";
import { StatCard } from "@/components/ui/StatCard";
import { SentimentGauge } from "@/components/ui/SentimentGauge";
import { EntityCloud } from "@/components/ui/EntityCloud";
import { ClusterGrid } from "@/components/ui/ClusterGrid";
import { SentimentPieChart } from "@/components/charts/SentimentPieChart";
import { DatePicker } from "@/components/ui/DatePicker";
import { SentimentTrendChart } from "@/components/charts/SentimentTrendChart";
import { localeForLanguage } from "@/lib/utils";
import { Globe2 } from "lucide-react";

interface Props {
  searchParams: { date?: string; country?: string };
}

async function DashboardContent({ date, country }: { date: string; country: string }) {
  const [data, trendData, countries] = await Promise.all([
    api.today(date, country).catch(() => null),
    api.trend(30, country).catch(() => null),
    api.countries().catch(() => []),
  ]);

  // Sprint 7.5: format the dashboard header date in the active country's
  // locale instead of always tr-TR. Falls back to en-US when /api/countries
  // is briefly unreachable so we never crash on locale lookup.
  const meta = findCountry(countries, country);
  const locale = localeForLanguage(meta?.language);

  if (!data) {
    return (
      <div className="flex flex-col items-center justify-center min-h-[60vh] gap-3">
        <div className="w-14 h-14 rounded-xl bg-white border border-slate-200 flex items-center justify-center">
          <Globe2 className="w-7 h-7 text-slate-300" />
        </div>
        <div className="text-center">
          <h2 className="text-lg font-semibold text-slate-800 mb-1">Veri bulunamadı</h2>
          <p className="text-slate-500 text-sm">{date} tarihi için analiz henüz hazır değil.</p>
        </div>
      </div>
    );
  }

  const positivePct = data.sentiment.percentages.positive ?? 0;
  const negativePct = data.sentiment.percentages.negative ?? 0;
  const neutralPct = data.sentiment.percentages.neutral ?? 0;
  const dominantPct = Math.max(positivePct, negativePct, neutralPct);
  let dominantLabel = "Ağırlıklı Pozitif";
  let dominantClass = "positive-badge";
  if (dominantPct === negativePct && negativePct > 0) {
    dominantLabel = "Ağırlıklı Negatif";
    dominantClass = "negative-badge";
  } else if (dominantPct === neutralPct && neutralPct > 0) {
    dominantLabel = "Ağırlıklı Nötr";
    dominantClass = "neutral-badge";
  }

  return (
    <div className="space-y-7">
      {/* Header */}
      <div className="flex items-start justify-between flex-wrap gap-3">
        <div>
          <h1 className="text-2xl font-bold text-slate-900 mb-1">Günlük Analiz</h1>
          <p className="text-slate-500 text-sm">
            {new Date(date + "T12:00:00").toLocaleDateString(locale, {
              weekday: "long",
              year: "numeric",
              month: "long",
              day: "numeric",
            })}
          </p>
        </div>
        <DatePicker availableDates={data.available_dates} currentDate={date} />
      </div>

      {/* Stat cards */}
      <div className="grid grid-cols-2 lg:grid-cols-4 gap-4">
        <StatCard label="Toplam Haber" value={data.total_items} iconName="Newspaper" />
        <StatCard label="Analiz Edilen" value={data.turkish_items} iconName="Globe2" />
        <StatCard
          label="Pozitif Oran"
          value={positivePct}
          suffix="%"
          decimals={1}
          iconName="TrendingUp"
          accent="positive"
        />
        <StatCard label="Konu Kümesi" value={data.cluster_count} iconName="Layers" />
      </div>

      {/* Main grid */}
      <div className="grid grid-cols-1 lg:grid-cols-3 gap-5">
        <div className="lg:col-span-2 card p-6">
          <div className="flex items-center justify-between mb-5">
            <div>
              <h2 className="text-base font-semibold text-slate-800">Duygu Analizi</h2>
              <p className="text-sm text-slate-500 mt-0.5">Haberlerin sentiment dağılımı</p>
            </div>
            <span className={dominantClass}>{dominantLabel}</span>
          </div>
          <div className="grid grid-cols-1 md:grid-cols-2 gap-6 items-center">
            <SentimentPieChart counts={data.sentiment.counts} />
            <SentimentGauge counts={data.sentiment.counts} locale={locale} />
          </div>
        </div>

        <div className="card p-6">
          <h2 className="text-base font-semibold text-slate-800 mb-1">Öne Çıkan Varlıklar</h2>
          <p className="text-sm text-slate-500 mb-5">Kişi, kurum ve coğrafi isimler</p>
          <EntityCloud entities={data.top_entities} />
        </div>
      </div>

      {/* Trend chart */}
      {trendData && trendData.points.length > 1 && (
        <div className="card p-6">
          <div className="flex items-center justify-between mb-5 flex-wrap gap-3">
            <div>
              <h2 className="text-base font-semibold text-slate-800">Sentiment Trendi</h2>
              <p className="text-sm text-slate-500 mt-0.5">Son 30 günlük haber sayısı</p>
            </div>
            <div className="flex items-center gap-4 text-xs text-slate-500">
              <span className="flex items-center gap-1.5">
                <span className="w-3 h-0.5 bg-green-500 rounded inline-block" />
                Pozitif
              </span>
              {trendData.points.some((p) => (p.neutral ?? 0) > 0) && (
                <span className="flex items-center gap-1.5">
                  <span className="w-3 h-0.5 bg-slate-400 rounded inline-block" style={{ borderTop: "1.5px dashed #94a3b8", height: 0 }} />
                  Nötr
                </span>
              )}
              <span className="flex items-center gap-1.5">
                <span className="w-3 h-0.5 bg-red-400 rounded inline-block" />
                Negatif
              </span>
            </div>
          </div>
          <SentimentTrendChart points={trendData.points} />
        </div>
      )}

      {/* Clusters */}
      <div>
        <div className="flex items-center justify-between mb-3">
          <div>
            <h2 className="text-base font-semibold text-slate-800">Konu Kümeleri</h2>
            <p className="text-sm text-slate-500">Haberlerin otomatik gruplandırması</p>
          </div>
          <span className="text-xs text-slate-400">{data.top_clusters.length} küme</span>
        </div>
        <ClusterGrid clusters={data.top_clusters} date={date} country={country} />
      </div>
    </div>
  );
}

export default function HomePage({ searchParams }: Props) {
  const date = searchParams.date || todayDate();
  const country = searchParams.country || DEFAULT_COUNTRY;

  return (
    <div className="max-w-7xl mx-auto px-6 py-8">
      <Suspense fallback={<DashboardSkeleton />}>
        <DashboardContent date={date} country={country} />
      </Suspense>
    </div>
  );
}

function DashboardSkeleton() {
  return (
    <div className="space-y-7 animate-pulse">
      <div className="h-8 w-44 bg-slate-200 rounded-lg" />
      <div className="grid grid-cols-4 gap-4">
        {Array.from({ length: 4 }).map((_, i) => (
          <div key={i} className="h-24 bg-white border border-slate-200 rounded-xl" />
        ))}
      </div>
      <div className="h-60 bg-white border border-slate-200 rounded-xl" />
    </div>
  );
}
