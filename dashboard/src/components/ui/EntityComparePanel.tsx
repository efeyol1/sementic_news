import type { CountryInfo, EntityProfile } from "@/lib/api";
import { cn } from "@/lib/utils";

interface Props {
  profiles: EntityProfile[];
  countries: CountryInfo[];
  activeCountry?: string; // ISO code of the page's active country, highlighted
}

/** Cross-country framing view: the same entity's top collocates side by side
 *  per country. The headline of the entity-narrative track. */
export function EntityComparePanel({ profiles, countries, activeCountry }: Props) {
  if (!profiles.length) {
    return (
      <p className="text-sm text-slate-400 italic">
        Bu entity henüz başka bir ülke profilinde görünmüyor.
      </p>
    );
  }

  const nameOf = (code: string) =>
    countries.find((c) => c.code === code)?.name ?? code;

  // Active country first, then by mention volume.
  const ordered = [...profiles].sort((a, b) => {
    if (a.country_code === activeCountry) return -1;
    if (b.country_code === activeCountry) return 1;
    return b.total_mentions - a.total_mentions;
  });

  return (
    <div className="grid grid-cols-1 md:grid-cols-2 xl:grid-cols-3 gap-4">
      {ordered.map((p) => (
        <div
          key={p.country_code}
          className={cn(
            "rounded-lg border p-4",
            p.country_code === activeCountry
              ? "border-green-300 bg-green-50/40"
              : "border-slate-200",
          )}
        >
          <div className="flex items-center justify-between mb-1">
            <span className="font-semibold text-slate-800 text-sm">
              {nameOf(p.country_code)}
            </span>
            <span className="text-[10px] font-mono text-slate-400">
              {p.country_code}
            </span>
          </div>
          <div className="text-xs text-slate-500 mb-3">
            {p.total_mentions} mention · {p.coverage_days}/{p.window_days} gün
          </div>
          <div className="flex flex-wrap gap-1.5">
            {p.top_collocates.slice(0, 8).map((c) => (
              <span
                key={`${c.lemma}-${c.pos}`}
                className="px-2 py-0.5 rounded-md text-xs bg-white text-slate-600 border border-slate-200"
                title={`LLR ${c.llr.toFixed(1)} · PMI ${c.pmi.toFixed(2)}`}
              >
                {c.lemma}
              </span>
            ))}
            {p.top_collocates.length === 0 && (
              <span className="text-xs text-slate-400 italic">collocate yok</span>
            )}
          </div>
        </div>
      ))}
    </div>
  );
}
