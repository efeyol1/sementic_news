import type { Collocate } from "@/lib/api";

interface Props {
  collocates: Collocate[];
}

/** Ranked co-occurring words for an entity. Bars are scaled by LLR — the
 *  signal the aggregator ranks on (PMI saturates for exclusive collocates,
 *  so it's shown only in the hover title). */
export function CollocateList({ collocates }: Props) {
  if (!collocates.length) {
    return (
      <p className="text-sm text-slate-400 italic">Collocate verisi yok</p>
    );
  }

  const maxLlr = Math.max(...collocates.map((c) => c.llr), 1);

  return (
    <ol className="space-y-2">
      {collocates.map((c, i) => (
        <li
          key={`${c.lemma}-${c.pos}`}
          className="flex items-center gap-3"
          title={`LLR ${c.llr.toFixed(1)} · PMI ${c.pmi.toFixed(2)} · ${c.c11_window}×`}
        >
          <span className="text-xs font-mono text-slate-300 w-5 text-right">
            {i + 1}
          </span>
          <span className="text-sm font-medium text-slate-700 w-36 truncate">
            {c.lemma}
          </span>
          <span className="text-[10px] uppercase tracking-wider text-slate-400 w-10">
            {c.pos}
          </span>
          <div className="flex-1 h-2 bg-slate-100 rounded-full overflow-hidden">
            <div
              className="h-full bg-green-500/70 rounded-full"
              style={{ width: `${(c.llr / maxLlr) * 100}%` }}
            />
          </div>
          <span className="text-xs font-mono text-slate-500 w-14 text-right">
            {c.llr.toFixed(1)}
          </span>
        </li>
      ))}
    </ol>
  );
}
