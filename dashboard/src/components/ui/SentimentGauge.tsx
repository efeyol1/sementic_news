import type { SentimentCounts } from "@/lib/api";

interface Props {
  counts: SentimentCounts;
}

export function SentimentGauge({ counts }: Props) {
  const pos = counts.positive ?? 0;
  const neg = counts.negative ?? 0;
  const neu = counts.neutral ?? 0;
  const total = pos + neg + neu || 1;
  const positivePct = (pos / total) * 100;
  const neutralPct = (neu / total) * 100;
  const negativePct = (neg / total) * 100;
  const showNeutral = neu > 0;

  return (
    <div className="space-y-3">
      <div className="flex h-2.5 rounded-full overflow-hidden gap-0.5 bg-slate-100">
        <div
          className="h-full bg-green-500 transition-all duration-500"
          style={{ width: `${positivePct}%`, minWidth: pos > 0 ? 4 : 0 }}
        />
        {showNeutral && (
          <div
            className="h-full bg-slate-400 transition-all duration-500"
            style={{ width: `${neutralPct}%`, minWidth: neu > 0 ? 4 : 0 }}
          />
        )}
        <div
          className="h-full bg-red-400 transition-all duration-500"
          style={{ width: `${negativePct}%`, minWidth: neg > 0 ? 4 : 0 }}
        />
      </div>
      <div className={`grid gap-3 ${showNeutral ? "grid-cols-3" : "grid-cols-2"}`}>
        <div>
          <div className="text-xl font-bold font-mono text-green-600">{positivePct.toFixed(1)}%</div>
          <div className="text-xs text-slate-500">Pozitif</div>
          <div className="text-xs text-slate-400">{pos.toLocaleString("tr-TR")} haber</div>
        </div>
        {showNeutral && (
          <div>
            <div className="text-xl font-bold font-mono text-slate-500">{neutralPct.toFixed(1)}%</div>
            <div className="text-xs text-slate-500">Nötr</div>
            <div className="text-xs text-slate-400">{neu.toLocaleString("tr-TR")} haber</div>
          </div>
        )}
        <div>
          <div className="text-xl font-bold font-mono text-red-500">{negativePct.toFixed(1)}%</div>
          <div className="text-xs text-slate-500">Negatif</div>
          <div className="text-xs text-slate-400">{neg.toLocaleString("tr-TR")} haber</div>
        </div>
      </div>
    </div>
  );
}
