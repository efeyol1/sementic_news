import { Suspense } from "react";
import type { ReactNode } from "react";
import { Activity, ArrowUp, ArrowDown, Minus } from "lucide-react";
import { api, DEFAULT_COUNTRY, findCountry, type DriftRatios } from "@/lib/api";
import { localeForLanguage, cn } from "@/lib/utils";
import { DriftPsiChart } from "@/components/charts/DriftPsiChart";

interface Props {
  searchParams: { country?: string };
}

// Severity → badge styling. `insufficient` also covers status="insufficient_data".
const SEVERITY: Record<string, { label: string; dot: string; text: string }> = {
  stable: { label: "Stabil", dot: "bg-emerald-500", text: "text-emerald-600" },
  moderate: { label: "Orta", dot: "bg-amber-500", text: "text-amber-600" },
  significant: { label: "Belirgin", dot: "bg-red-500", text: "text-red-600" },
  insufficient: { label: "Yetersiz", dot: "bg-slate-400", text: "text-slate-500" },
};

const CLASS_META: { key: keyof DriftRatios; label: string; text: string }[] = [
  { key: "negative", label: "Negatif", text: "text-red-500" },
  { key: "neutral", label: "Nötr", text: "text-slate-500" },
  { key: "positive", label: "Pozitif", text: "text-green-600" },
];

function StatCard({ label, children, sub }: { label: string; children: ReactNode; sub?: string }) {
  return (
    <div className="card p-5">
      <div className="text-xs font-medium text-slate-500 uppercase tracking-wider mb-2">
        {label}
      </div>
      <div className="text-2xl font-mono font-bold text-slate-800">{children}</div>
      {sub && <div className="text-xs text-slate-400 mt-1">{sub}</div>}
    </div>
  );
}

function pct(v: number | undefined | null): string {
  return v != null ? `%${(v * 100).toFixed(1)}` : "—";
}

/** Per-class shift in percentage points. A delta is neither good nor bad on its
 *  own, so it reads neutral (slate) — the arrow only conveys direction. */
function DeltaCell({ value }: { value: number | undefined | null }) {
  if (value == null) return <span className="font-mono text-slate-400">—</span>;
  const pp = value * 100;
  const flat = Math.abs(pp) < 0.05;
  const Icon = flat ? Minus : pp > 0 ? ArrowUp : ArrowDown;
  return (
    <span className={cn("inline-flex items-center justify-end gap-1 font-mono", flat ? "text-slate-400" : "text-slate-700")}>
      <Icon className="w-3.5 h-3.5" />
      {pp > 0 ? "+" : ""}
      {pp.toFixed(1)} pp
    </span>
  );
}

async function HealthContent({ country }: { country: string }) {
  const [latest, history, countries] = await Promise.all([
    api.driftLatest(country).catch(() => null),
    api.driftHistory(60, country).catch(() => null),
    api.countries().catch(() => []),
  ]);

  const meta = findCountry(countries, country);
  const locale = localeForLanguage(meta?.language);

  if (!latest) {
    return (
      <div className="flex flex-col items-center justify-center min-h-[60vh] gap-3">
        <Activity className="w-10 h-10 text-slate-300" />
        <p className="text-slate-500 text-sm">
          Henüz drift raporu yok — daily pipeline'ın drift adımını bekleyin.
        </p>
      </div>
    );
  }

  const sev = SEVERITY[latest.severity] ?? SEVERITY.insufficient;
  const insufficient = latest.status === "insufficient_data";

  const points = (history?.reports ?? [])
    .filter((r) => r.psi != null)
    .map((r) => ({ date: r.date, psi: r.psi as number, severity: r.severity }));

  return (
    <div className="space-y-7">
      <div>
        <h1 className="text-2xl font-bold text-slate-900 mb-1">Veri Sağlığı</h1>
        <p className="text-slate-500 text-sm">
          {meta?.name ?? country} · son rapor {latest.date}
        </p>
      </div>

      {insufficient && (
        <div className="card p-4 border-amber-200 bg-amber-50/60 text-sm text-amber-700">
          Baseline için yeterli gün yok ({latest.baseline_days} gün) — PSI güvenilir değil.
        </div>
      )}

      <div className="grid grid-cols-2 md:grid-cols-4 gap-4">
        <StatCard label="Durum">
          <span className={cn("inline-flex items-center gap-2", sev.text)}>
            <span className={cn("w-2.5 h-2.5 rounded-full", sev.dot)} />
            {sev.label}
          </span>
        </StatCard>
        <StatCard label="PSI">{latest.psi != null ? latest.psi.toFixed(3) : "—"}</StatCard>
        <StatCard label="Baseline" sub="gün sayısı">
          {latest.baseline_days}
        </StatCard>
        <StatCard label="Bugünkü Haber">{latest.today_total.toLocaleString(locale)}</StatCard>
      </div>

      <div className="card p-6">
        <h2 className="text-base font-semibold text-slate-800 mb-1">
          Dağılım — Bugün vs Baseline
        </h2>
        <p className="text-sm text-slate-500 mb-5">
          30 günlük baseline'a göre sentiment sınıflarının kayması
        </p>
        <div className="overflow-x-auto">
          <table className="w-full text-sm">
            <thead>
              <tr className="border-b border-slate-100">
                <th className="text-left py-2.5 px-3 text-slate-400 font-medium">Sınıf</th>
                <th className="text-right py-2.5 px-3 text-slate-400 font-medium">Bugün</th>
                <th className="text-right py-2.5 px-3 text-slate-400 font-medium">Baseline</th>
                <th className="text-right py-2.5 px-3 text-slate-400 font-medium">Δ (pp)</th>
              </tr>
            </thead>
            <tbody>
              {CLASS_META.map(({ key, label, text }) => (
                <tr key={key} className="border-b border-slate-50">
                  <td className={cn("py-2.5 px-3 font-medium", text)}>{label}</td>
                  <td className="py-2.5 px-3 text-right font-mono text-slate-600">
                    {pct(latest.today_ratios?.[key])}
                  </td>
                  <td className="py-2.5 px-3 text-right font-mono text-slate-500">
                    {pct(latest.baseline_ratios?.[key])}
                  </td>
                  <td className="py-2.5 px-3 text-right">
                    <DeltaCell value={latest.per_class_delta?.[key]} />
                  </td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      </div>

      <div className="card p-6">
        <h2 className="text-base font-semibold text-slate-800 mb-1">PSI Zaman Serisi</h2>
        <p className="text-sm text-slate-500 mb-5">
          Günlük sentiment dağılımının baseline'dan sapması (eşikler: 0.10 orta, 0.25 belirgin)
        </p>
        <DriftPsiChart points={points} locale={locale} />
      </div>
    </div>
  );
}

export default function HealthPage({ searchParams }: Props) {
  const country = searchParams.country || DEFAULT_COUNTRY;

  return (
    <div className="max-w-7xl mx-auto px-6 py-8">
      <Suspense
        fallback={
          <div className="animate-pulse space-y-4">
            <div className="h-8 w-44 bg-slate-200 rounded-lg" />
            <div className="grid grid-cols-2 md:grid-cols-4 gap-4">
              {[0, 1, 2, 3].map((i) => (
                <div key={i} className="h-24 bg-white border border-slate-200 rounded-xl" />
              ))}
            </div>
            <div className="h-72 bg-white border border-slate-200 rounded-xl" />
          </div>
        }
      >
        <HealthContent country={country} />
      </Suspense>
    </div>
  );
}
