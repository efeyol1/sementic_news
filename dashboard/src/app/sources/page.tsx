import { Suspense } from "react";
import { api, todayDate } from "@/lib/api";
import { SourceHeatmap } from "@/components/charts/SourceHeatmap";
import { BarChart3, TrendingDown, TrendingUp, Minus } from "lucide-react";

interface Props {
  searchParams: { date?: string };
}

async function SourcesContent({ date }: { date: string }) {
  const data = await api.sources(date).catch(() => null);

  if (!data) {
    return (
      <div className="flex flex-col items-center justify-center min-h-[60vh] gap-4">
        <BarChart3 className="w-12 h-12 text-text-muted" />
        <p className="text-text-secondary">{date} için kaynak verisi bulunamadı.</p>
      </div>
    );
  }

  const sources = Object.entries(data.sources).sort((a, b) => b[1].total - a[1].total);
  const mostNegative = sources.reduce((a, b) =>
    b[1].sentiment_percentages.negative > a[1].sentiment_percentages.negative ? b : a
  );
  const mostPositive = sources.reduce((a, b) =>
    b[1].sentiment_percentages.positive > a[1].sentiment_percentages.positive ? b : a
  );
  const totalNews = sources.reduce((sum, [, s]) => sum + s.total, 0);

  return (
    <div className="space-y-8">
      <div>
        <h1 className="text-3xl font-bold gradient-text mb-1">Kaynak Analizi</h1>
        <p className="text-text-secondary text-sm">
          {new Date(date).toLocaleDateString("tr-TR", { year: "numeric", month: "long", day: "numeric" })} —{" "}
          {sources.length} kaynak, {totalNews.toLocaleString("tr-TR")} haber
        </p>
      </div>

      {/* Highlight cards */}
      <div className="grid grid-cols-1 md:grid-cols-3 gap-4">
        <div className="glass-card p-5">
          <div className="flex items-center gap-2 mb-3">
            <TrendingDown className="w-4 h-4 text-sentiment-negative" />
            <span className="text-xs font-medium text-text-secondary uppercase tracking-wider">En Negatif</span>
          </div>
          <div className="text-xl font-bold text-text-primary">{mostNegative[0]}</div>
          <div className="text-2xl font-mono font-bold text-sentiment-negative mt-1">
            %{mostNegative[1].sentiment_percentages.negative.toFixed(1)}
          </div>
          <div className="text-xs text-text-muted mt-1">{mostNegative[1].total} haber</div>
        </div>

        <div className="glass-card p-5">
          <div className="flex items-center gap-2 mb-3">
            <TrendingUp className="w-4 h-4 text-sentiment-positive" />
            <span className="text-xs font-medium text-text-secondary uppercase tracking-wider">En Pozitif</span>
          </div>
          <div className="text-xl font-bold text-text-primary">{mostPositive[0]}</div>
          <div className="text-2xl font-mono font-bold text-sentiment-positive mt-1">
            %{mostPositive[1].sentiment_percentages.positive.toFixed(1)}
          </div>
          <div className="text-xs text-text-muted mt-1">{mostPositive[1].total} haber</div>
        </div>

        <div className="glass-card p-5">
          <div className="flex items-center gap-2 mb-3">
            <Minus className="w-4 h-4 text-text-secondary" />
            <span className="text-xs font-medium text-text-secondary uppercase tracking-wider">Toplam</span>
          </div>
          <div className="text-2xl font-mono font-bold text-text-primary">{totalNews.toLocaleString("tr-TR")}</div>
          <div className="text-xs text-text-muted mt-1">{sources.length} kaynak taranıyor</div>
        </div>
      </div>

      {/* Heatmap */}
      <div className="glass-card p-6">
        <h2 className="text-lg font-semibold text-text-primary mb-1">Kaynak × Sentiment Dağılımı</h2>
        <p className="text-sm text-text-secondary mb-6">Her kaynağın haber tonu dağılımı</p>
        <SourceHeatmap data={data} />
      </div>

      {/* Detail table */}
      <div className="glass-card p-6">
        <h2 className="text-lg font-semibold text-text-primary mb-4">Detaylı Tablo</h2>
        <div className="overflow-x-auto">
          <table className="w-full text-sm">
            <thead>
              <tr className="border-b border-border-subtle">
                <th className="text-left py-3 px-4 text-text-muted font-medium">Kaynak</th>
                <th className="text-right py-3 px-4 text-text-muted font-medium">Haber</th>
                <th className="text-right py-3 px-4 text-sentiment-positive font-medium">Pozitif</th>
                <th className="text-right py-3 px-4 text-sentiment-negative font-medium">Negatif</th>
                <th className="text-right py-3 px-4 text-text-muted font-medium">Güven</th>
              </tr>
            </thead>
            <tbody>
              {sources.map(([name, stats]) => (
                <tr
                  key={name}
                  className="border-b border-border-subtle/50 hover:bg-bg-hover/30 transition-colors"
                >
                  <td className="py-3 px-4 font-medium text-text-primary">{name}</td>
                  <td className="py-3 px-4 text-right font-mono text-text-secondary">{stats.total}</td>
                  <td className="py-3 px-4 text-right font-mono text-sentiment-positive">
                    %{stats.sentiment_percentages.positive.toFixed(1)}
                  </td>
                  <td className="py-3 px-4 text-right font-mono text-sentiment-negative">
                    %{stats.sentiment_percentages.negative.toFixed(1)}
                  </td>
                  <td className="py-3 px-4 text-right font-mono text-text-muted">
                    %{(stats.avg_confidence * 100).toFixed(1)}
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

  return (
    <div className="max-w-7xl mx-auto px-6 py-8">
      <div
        className="fixed top-0 left-1/2 -translate-x-1/2 w-[800px] h-[400px] pointer-events-none"
        style={{
          background:
            "radial-gradient(ellipse at center top, rgba(99, 102, 241, 0.06) 0%, transparent 70%)",
        }}
      />
      <Suspense
        fallback={
          <div className="animate-pulse space-y-4">
            <div className="h-8 w-48 bg-bg-card rounded-xl" />
            <div className="h-64 bg-bg-card rounded-2xl" />
          </div>
        }
      >
        <SourcesContent date={date} />
      </Suspense>
    </div>
  );
}
