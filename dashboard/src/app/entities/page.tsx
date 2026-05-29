import { Suspense } from "react";
import Link from "next/link";
import { Network, ChevronRight } from "lucide-react";
import { api, DEFAULT_COUNTRY, findCountry } from "@/lib/api";
import { localeForLanguage, entityTypeMeta } from "@/lib/utils";
import { buildEntityUrl } from "@/lib/url";
import { EntitySearchBar } from "@/components/ui/EntitySearchBar";

interface Props {
  searchParams: { country?: string; q?: string; type?: string; window?: string };
}

async function EntitiesContent({
  country,
  q,
  type,
  windowDays,
}: {
  country: string;
  q?: string;
  type?: string;
  windowDays: number;
}) {
  const [data, countries] = await Promise.all([
    api.entities({ q, entityType: type, windowDays, limit: 100 }, country).catch(() => null),
    api.countries().catch(() => []),
  ]);

  const meta = findCountry(countries, country);
  const locale = localeForLanguage(meta?.language);
  const entities = data?.entities ?? [];

  return (
    <div className="space-y-6">
      <div>
        <h1 className="text-2xl font-bold text-slate-900 mb-1">Entity Araştır</h1>
        <p className="text-slate-500 text-sm">
          {meta?.name ?? country} · son {windowDays} gün
          {data?.end_date ? ` · ${data.end_date}` : ""}
        </p>
      </div>

      <EntitySearchBar initialQuery={q ?? ""} entityType={type ?? ""} />

      {entities.length === 0 ? (
        <div className="flex flex-col items-center justify-center min-h-[40vh] gap-3">
          <Network className="w-10 h-10 text-slate-300" />
          <p className="text-slate-500 text-sm">
            {q ? `"${q}" ile eşleşen entity yok.` : "Bu ülke için entity profili bulunamadı."}
          </p>
        </div>
      ) : (
        <div className="card divide-y divide-slate-100">
          {entities.map((e) => {
            const ref = e.wikidata_qid ?? e.canonical;
            const m = entityTypeMeta(e.entity_type);
            return (
              <Link
                key={`${e.canonical}-${e.entity_type}`}
                href={buildEntityUrl(ref, country)}
                className="flex items-center gap-3 px-4 py-3 hover:bg-slate-50 transition-colors"
              >
                <span
                  className={`px-1.5 py-0.5 rounded text-[10px] font-medium border ${m.chip}`}
                >
                  {m.label}
                </span>
                <span className="font-medium text-slate-800 text-sm flex-1 truncate">
                  {e.canonical}
                </span>
                {e.wikidata_qid && (
                  <span className="text-[10px] font-mono text-slate-400">
                    {e.wikidata_qid}
                  </span>
                )}
                <span className="text-xs font-mono text-slate-500 shrink-0">
                  {e.total_mentions.toLocaleString(locale)} mention
                </span>
                <ChevronRight className="w-4 h-4 text-slate-300 shrink-0" />
              </Link>
            );
          })}
        </div>
      )}
    </div>
  );
}

export default function EntitiesPage({ searchParams }: Props) {
  const country = searchParams.country || DEFAULT_COUNTRY;
  const q = searchParams.q || undefined;
  const type = searchParams.type || undefined;
  const windowDays = searchParams.window === "7" ? 7 : 30;

  return (
    <div className="max-w-5xl mx-auto px-6 py-8">
      <Suspense
        fallback={
          <div className="animate-pulse space-y-4">
            <div className="h-8 w-44 bg-slate-200 rounded-lg" />
            <div className="h-11 bg-white border border-slate-200 rounded-xl" />
            <div className="h-60 bg-white border border-slate-200 rounded-xl" />
          </div>
        }
      >
        <EntitiesContent country={country} q={q} type={type} windowDays={windowDays} />
      </Suspense>
    </div>
  );
}
