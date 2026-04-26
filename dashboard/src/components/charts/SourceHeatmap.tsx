import type { SourceComparison } from "@/lib/api";

interface Props {
  data: SourceComparison;
}

export function SourceHeatmap({ data }: Props) {
  const sources = Object.entries(data.sources).sort(
    (a, b) => (b[1].sentiment_percentages.negative ?? 0) - (a[1].sentiment_percentages.negative ?? 0)
  );

  return (
    <div className="space-y-2">
      {sources.map(([name, stats]) => {
        const pos = stats.sentiment_percentages.positive ?? 0;
        const neu = stats.sentiment_percentages.neutral ?? 0;
        const neg = stats.sentiment_percentages.negative ?? 0;

        return (
          <div key={name} className="flex items-center gap-3">
            <div className="w-24 text-xs text-slate-500 font-medium truncate shrink-0 text-right">
              {name}
            </div>
            <div className="flex-1 h-5 rounded-md overflow-hidden flex gap-0.5 bg-slate-100">
              <div
                className="h-full bg-green-400 flex items-center justify-center text-xs font-semibold text-white"
                style={{ width: `${pos}%`, minWidth: pos > 0 ? 4 : 0 }}
              >
                {pos > 15 && `${pos.toFixed(0)}%`}
              </div>
              {neu > 0 && (
                <div
                  className="h-full bg-slate-400 flex items-center justify-center text-xs font-semibold text-white"
                  style={{ width: `${neu}%`, minWidth: 4 }}
                >
                  {neu > 15 && `${neu.toFixed(0)}%`}
                </div>
              )}
              <div
                className="h-full bg-red-400 flex items-center justify-center text-xs font-semibold text-white"
                style={{ width: `${neg}%`, minWidth: neg > 0 ? 4 : 0 }}
              >
                {neg > 15 && `${neg.toFixed(0)}%`}
              </div>
            </div>
            <div className="w-10 text-right">
              <span className="text-xs font-mono text-slate-400">{stats.total}</span>
            </div>
          </div>
        );
      })}

      <div className="flex items-center gap-4 pt-3 border-t border-slate-100">
        {[
          { color: "bg-green-400", label: "Pozitif" },
          { color: "bg-slate-400", label: "Nötr" },
          { color: "bg-red-400", label: "Negatif" },
        ].map(({ color, label }) => (
          <div key={label} className="flex items-center gap-1.5">
            <div className={`w-2.5 h-2.5 rounded-sm ${color}`} />
            <span className="text-xs text-slate-400">{label}</span>
          </div>
        ))}
      </div>
    </div>
  );
}
