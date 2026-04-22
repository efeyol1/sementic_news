import type { SentimentCounts } from "@/lib/api";

interface Props {
  counts: SentimentCounts;
}

export function SentimentGauge({ counts }: Props) {
  const pos = counts.positive ?? 0;
  const neg = counts.negative ?? 0;
  const total = pos + neg || 1;
  const positivePct = (pos / total) * 100;
  const negativePct = (neg / total) * 100;

  return (
    <div className="space-y-3">
      <div className="flex h-2.5 rounded-full overflow-hidden gap-0.5 bg-slate-100">
        <div
          className="h-full bg-green-500 rounded-l-full transition-all duration-500"
          style={{ width: `${positivePct}%`, minWidth: pos > 0 ? 4 : 0 }}
        />
        <div
          className="h-full bg-red-400 rounded-r-full transition-all duration-500"
          style={{ width: `${negativePct}%`, minWidth: neg > 0 ? 4 : 0 }}
        />
      </div>
      <div className="grid grid-cols-2 gap-3">
        <div>
          <div className="text-xl font-bold font-mono text-green-600">{positivePct.toFixed(1)}%</div>
          <div className="text-xs text-slate-500">Pozitif</div>
          <div className="text-xs text-slate-400">{pos.toLocaleString("tr-TR")} haber</div>
        </div>
        <div>
          <div className="text-xl font-bold font-mono text-red-500">{negativePct.toFixed(1)}%</div>
          <div className="text-xs text-slate-500">Negatif</div>
          <div className="text-xs text-slate-400">{neg.toLocaleString("tr-TR")} haber</div>
        </div>
      </div>
    </div>
  );
}
