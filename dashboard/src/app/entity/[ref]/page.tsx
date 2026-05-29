import { Suspense } from "react";
import Link from "next/link";
import { ArrowLeft, ExternalLink, Network } from "lucide-react";
import { api, DEFAULT_COUNTRY, findCountry } from "@/lib/api";
import { localeForLanguage, entityTypeMeta } from "@/lib/utils";
import { buildEntitiesUrl } from "@/lib/url";
import { CollocateList } from "@/components/ui/CollocateList";
import { EntityComparePanel } from "@/components/ui/EntityComparePanel";
import { EntityTimelineChart } from "@/components/charts/EntityTimelineChart";
import { WindowToggle } from "@/components/ui/WindowToggle";

interface Props {
  params: { ref: string };
  searchParams: { country?: string; window?: string };
}

function StatCard({ label, value, sub }: { label: string; value: string; sub?: string }) {
  return (
    <div className="card p-5">
      <div className="text-xs font-medium text-slate-500 uppercase tracking-wider mb-2">
        {label}
      </div>
      <div className="text-2xl font-mono font-bold text-slate-800">{value}</div>
      {sub && <div className="text-xs text-slate-400 mt-1">{sub}</div>}
    </div>
  );
}

async function EntityContent({
  entityRef,
  country,
  windowDays,
}: {
  entityRef: string;
  country: string;
  windowDays: number;
}) {
  const [profile, compare, timeline, countries] = await Promise.all([
    api.entityProfile(entityRef, windowDays, country).catch(() => null),
    api.entityCompare(entityRef, windowDays).catch(() => null),
    api.entityTimeline(entityRef, 30, country).catch(() => null),
    api.countries().catch(() => []),
  ]);

  const meta = findCountry(countries, country);
  const locale = localeForLanguage(meta?.language);
  const activeCode = meta?.code;

  // Pull display name/type from the active-country profile when present,
  // otherwise from the first compare row (entity may not exist in this
  // country but still exist elsewhere).
  const head = profile ?? compare?.countries?.[0] ?? null;

  if (!head) {
    return (
      <div className="flex flex-col items-center justify-center min-h-[50vh] gap-3">
        <Network className="w-10 h-10 text-slate-300" />
        <p className="text-slate-500 text-sm">
          <span className="font-mono">{entityRef}</span> için entity profili bulunamadı.
        </p>
        <Link href={buildEntitiesUrl(country)} className="text-sm text-green-600 hover:underline">
          ← Entity listesine dön
        </Link>
      </div>
    );
  }

  const m = entityTypeMeta(head.entity_type);
  const isQid = /^Q\d+$/.test(entityRef);

  return (
    <div className="space-y-7">
      <div>
        <Link
          href={buildEntitiesUrl(country)}
          className="inline-flex items-center gap-1.5 text-sm text-slate-500 hover:text-slate-700 mb-3"
        >
          <ArrowLeft className="w-4 h-4" /> Entity listesi
        </Link>
        <div className="flex items-start justify-between flex-wrap gap-3">
          <div className="flex items-center gap-3 flex-wrap">
            <h1 className="text-2xl font-bold text-slate-900">{head.canonical}</h1>
            <span className={`px-2 py-0.5 rounded-md text-xs font-medium border ${m.chip}`}>
              {m.label}
            </span>
            {isQid && (
              <a
                href={`https://www.wikidata.org/wiki/${entityRef}`}
                target="_blank"
                rel="noopener noreferrer"
                className="inline-flex items-center gap-1 text-xs font-mono text-slate-400 hover:text-slate-600"
              >
                {entityRef} <ExternalLink className="w-3 h-3" />
              </a>
            )}
          </div>
          <WindowToggle current={windowDays} />
        </div>
        <p className="text-slate-500 text-sm mt-1">
          {meta?.name ?? country} · son {windowDays} gün
        </p>
      </div>

      {profile ? (
        <>
          <div className="grid grid-cols-2 md:grid-cols-4 gap-4">
            <StatCard
              label="Mention"
              value={profile.total_mentions.toLocaleString(locale)}
              sub={`${profile.coverage_days}/${profile.window_days} gün`}
            />
            <StatCard
              label="Cooccurrence"
              value={profile.total_cooccurrences.toLocaleString(locale)}
            />
            <StatCard
              label="Ø PMI"
              value={profile.avg_pmi != null ? profile.avg_pmi.toFixed(2) : "—"}
            />
            <StatCard
              label="Ø LLR"
              value={
                profile.avg_log_likelihood != null
                  ? profile.avg_log_likelihood.toFixed(1)
                  : "—"
              }
            />
          </div>

          <div className="grid grid-cols-1 lg:grid-cols-2 gap-5">
            <div className="card p-6">
              <h2 className="text-base font-semibold text-slate-800 mb-1">
                Top Collocate'ler
              </h2>
              <p className="text-sm text-slate-500 mb-5">
                {meta?.name ?? country} medyasında bu entity ile en güçlü birlikte
                geçen kelimeler (LLR sıralı)
              </p>
              <CollocateList collocates={profile.top_collocates} />
            </div>

            <div className="card p-6">
              <h2 className="text-base font-semibold text-slate-800 mb-1">
                Mention Zaman Serisi
              </h2>
              <p className="text-sm text-slate-500 mb-5">Son 30 günde günlük anılma</p>
              <EntityTimelineChart points={timeline?.points ?? []} />
            </div>
          </div>
        </>
      ) : (
        <div className="card p-5 text-sm text-slate-500">
          Bu entity <span className="font-medium">{meta?.name ?? country}</span> medyasında
          son {windowDays} günde profil oluşturacak kadar geçmiyor — aşağıda diğer
          ülkelerdeki çerçeveleme görülebilir.
        </div>
      )}

      <div className="card p-6">
        <h2 className="text-base font-semibold text-slate-800 mb-1">
          Ülkeler Arası Karşılaştırma
        </h2>
        <p className="text-sm text-slate-500 mb-5">
          Aynı entity'nin farklı ülke medyalarındaki top collocate'leri — çerçeveleme
          farkları
        </p>
        <EntityComparePanel
          profiles={compare?.countries ?? []}
          countries={countries}
          activeCountry={activeCode}
        />
      </div>
    </div>
  );
}

export default function EntityPage({ params, searchParams }: Props) {
  // Next decodes dynamic params once; the api client re-encodes for the fetch.
  const ref = params.ref;
  const country = searchParams.country || DEFAULT_COUNTRY;
  const windowDays = searchParams.window === "7" ? 7 : 30;

  return (
    <div className="max-w-7xl mx-auto px-6 py-8">
      <Suspense
        fallback={
          <div className="animate-pulse space-y-4">
            <div className="h-8 w-60 bg-slate-200 rounded-lg" />
            <div className="grid grid-cols-2 md:grid-cols-4 gap-4">
              {[0, 1, 2, 3].map((i) => (
                <div key={i} className="h-24 bg-white border border-slate-200 rounded-xl" />
              ))}
            </div>
            <div className="h-60 bg-white border border-slate-200 rounded-xl" />
          </div>
        }
      >
        <EntityContent entityRef={ref} country={country} windowDays={windowDays} />
      </Suspense>
    </div>
  );
}
