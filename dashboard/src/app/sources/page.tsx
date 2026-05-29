import { Suspense } from "react";
import { api, DEFAULT_COUNTRY, findCountry, todayDate } from "@/lib/api";
import { SourceHeatmap } from "@/components/charts/SourceHeatmap";
import { DatePicker } from "@/components/ui/DatePicker";
import { localeForLanguage } from "@/lib/utils";
import { BarChart3, TrendingDown, TrendingUp, Minus } from "lucide-react";

interface Props {
  searchParams: { date?: string; country?: string };
}

async function SourcesContent({ date, country }: { date: string; country: string }) {
  const [data, datesData, countries] = await Promise.all([
    api.sources(date, country).catch(() => null),
    api.dates(country).catch(() => null),
    api.countries().catch(() => []),
  ]);

  // Sprint 7.5: locale picked off the active country (falls back to en-US
  // when /api/countries is briefly unreachable).
  const meta = findCountry(countries, country);
  const locale = localeForLanguage(meta?.language);

  if (!data) {
    return (
      <div className="flex flex-col items-center justify-center min-h-[60vh] gap-3">
        <BarChart3 className="w-10 h-10 text-slate-300" />
        <p className="text-slate-500 text-sm">{date} için kaynak verisi bulunamadı.</p>
      </div>
    );
  }

  const sources = Object.entries(data.sources).sort((a, b) => b[1].total - a[1].total);
  const mostNegative = sources.reduce((a, b) =>
    (b[1].sentiment_percentages.negative ?? 0) > (a[1].sentiment_percentages.negative ?? 0) ? b : a
  );
  const mostPositive = sources.reduce((a, b) =>
    (b[1].sentiment_percentages.positive ?? 0) > (a[1].sentiment_percentages.positive ?? 0) ? b : a
  );
  const totalNews = sources.reduce((sum, [, s]) => sum + s.total, 0);

  return (
    <div className="space-y-7">
      <div className="flex items-start justify-between flex-wrap gap-3">
        <div>
          <h1 className="text-2xl font-bold text-slate-900 mb-1">Kaynak Analizi</h1>
          <p className="text-slate-500 text-sm">
            {new Date(date + "T12:00:00").toLocaleDateString(locale, { year: "numeric", month: "long", day: "numeric" })} —{" "}
            {sources.length} kaynak, {totalNews.toLocaleString(locale)} haber
          </p>
        </div>
        {datesData && (
          <DatePicker availableDates={datesData.dates} currentDate={date} />
        )}
      </div>

      {/* Highlight cards */}
      <div className="grid grid-cols-1 md:grid-cols-3 gap-4">
        <div className="card p-5">
          <div className="flex items-center gap-2 mb-3">
            <TrendingDown className="w-4 h-4 text-red-400" />
            <span className="text-xs font-medium text-slate-500 uppercase tracking-wider">En Negatif</span>
          </div>
          <div className="text-lg font-semibold text-slate-800">{mostNegative[0]}</div>
          <div className="text-2xl font-mono font-bold text-red-500 mt-1">
            %{(mostNegative[1].sentiment_percentages.negative ?? 0).toFixed(1)}
          </div>
          <div className="text-xs text-slate-400 mt-1">{mostNegative[1].total} haber</div>
        </div>

        <div className="card p-5">
          <div className="flex items-center gap-2 mb-3">
            <TrendingUp className="w-4 h-4 text-green-500" />
            <span className="text-xs font-medium text-slate-500 uppercase tracking-wider">En Pozitif</span>
          </div>
          <div className="text-lg font-semibold text-slate-800">{mostPositive[0]}</div>
          <div className="text-2xl font-mono font-bold text-green-600 mt-1">
            %{(mostPositive[1].sentiment_percentages.positive ?? 0).toFixed(1)}
          </div>
          <div className="text-xs text-slate-400 mt-1">{mostPositive[1].total} haber</div>
        </div>

        <div className="card p-5">
          <div className="flex items-center gap-2 mb-3">
            <Minus className="w-4 h-4 text-slate-400" />
            <span className="text-xs font-medium text-slate-500 uppercase tracking-wider">Toplam</span>
          </div>
          <div className="text-2xl font-mono font-bold text-slate-800">{totalNews.toLocaleString(locale)}</div>
          <div className="text-xs text-slate-400 mt-1">{sources.length} kaynak taranıyor</div>
        </div>
      </div>

      {/* Heatmap */}
      <div className="card p-6">
        <h2 className="text-base font-semibold text-slate-800 mb-1">Kaynak × Sentiment Dağılımı</h2>
        <p className="text-sm text-slate-500 mb-5">Her kaynağın haber tonu dağılımı</p>
        <SourceHeatmap data={data} />
      </div>

      {/* Detail table */}
      <div className="card p-6">
        <h2 className="text-base font-semibold text-slate-800 mb-4">Detaylı Tablo</h2>
        <div className="overflow-x-auto">
          <table className="w-full text-sm">
            <thead>
              <tr className="border-b border-slate-100">
                <th className="text-left py-2.5 px-3 text-slate-400 font-medium">Kaynak</th>
                <th className="text-right py-2.5 px-3 text-slate-400 font-medium">Haber</th>
                <th className="text-right py-2.5 px-3 text-green-600 font-medium">Pozitif</th>
                <th className="text-right py-2.5 px-3 text-slate-500 font-medium">Nötr</th>
                <th className="text-right py-2.5 px-3 text-red-500 font-medium">Negatif</th>
                <th className="text-right py-2.5 px-3 text-slate-400 font-medium">Güven</th>
              </tr>
            </thead>
            <tbody>
              {sources.map(([name, stats]) => (
                <tr key={name} className="border-b border-slate-50 hover:bg-slate-50 transition-colors">
                  <td className="py-2.5 px-3 font-medium text-slate-700">{name}</td>
                  <td className="py-2.5 px-3 text-right font-mono text-slate-500">{stats.total}</td>
                  <td className="py-2.5 px-3 text-right font-mono text-green-600">
                    %{(stats.sentiment_percentages.positive ?? 0).toFixed(1)}
                  </td>
                  <td className="py-2.5 px-3 text-right font-mono text-slate-500">
                    %{(stats.sentiment_percentages.neutral ?? 0).toFixed(1)}
                  </td>
                  <td className="py-2.5 px-3 text-right font-mono text-red-500">
                    %{(stats.sentiment_percentages.negative ?? 0).toFixed(1)}
                  </td>
                  <td className="py-2.5 px-3 text-right font-mono text-slate-400">
                    {stats.avg_confidence != null
                      ? `%${(stats.avg_confidence * 100).toFixed(1)}`
                      : "—"}
                  </td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      </div>
    </div>
  );
}

export default function SourcesPage({ searchParams }: Props) {
  const date = searchParams.date || todayDate();
  const country = searchParams.country || DEFAULT_COUNTRY;

  return (
    <div className="max-w-7xl mx-auto px-6 py-8">
      <Suspense
        fallback={
          <div className="animate-pulse space-y-4">
            <div className="h-8 w-44 bg-slate-200 rounded-lg" />
            <div className="h-60 bg-white border border-slate-200 rounded-xl" />
          </div>
        }
      >
        <SourcesContent date={date} country={country} />
      </Suspense>
    </div>
  );
}
