"use client";

import { useState } from "react";
import { api, type EntityExplainResponse } from "@/lib/api";

interface Props {
  entityRef: string;
  country?: string;
  windowDays?: number;
  lang?: string;
}

// Lazy "Why? / Neden?" panel: fetches the grounded explanation only when the
// user expands it (cost control). Renders nothing if the feature is off for
// the country or the request fails.
export function EntityExplainPanel({ entityRef, country, windowDays = 30, lang = "tr" }: Props) {
  const [open, setOpen] = useState(false);
  const [loading, setLoading] = useState(false);
  const [data, setData] = useState<EntityExplainResponse | null>(null);
  const [failed, setFailed] = useState(false);

  async function toggle() {
    const next = !open;
    setOpen(next);
    if (next && data === null && !failed) {
      setLoading(true);
      try {
        const res = await api.entityExplain(entityRef, windowDays, country, lang);
        setData(res);
      } catch {
        setFailed(true);
      } finally {
        setLoading(false);
      }
    }
  }

  // Hide the whole panel when the feature is off for this country.
  if (data && !data.available) return null;

  const citById = new Map((data?.citations ?? []).map((c) => [c.id, c]));

  return (
    <div className="card p-6">
      <button
        onClick={toggle}
        className="flex w-full items-center justify-between text-left"
      >
        <h2 className="text-base font-semibold text-slate-800">
          Neden? · Çerçeveleme açıklaması
        </h2>
        <span className="text-sm text-slate-400">{open ? "−" : "+"}</span>
      </button>

      {open && (
        <div className="mt-4 space-y-3">
          {loading && <p className="text-sm text-slate-400">Yükleniyor…</p>}
          {failed && (
            <p className="text-sm text-slate-400">Açıklama şu an üretilemedi.</p>
          )}
          {data && data.available && (
            <>
              {data.frame_summary && (
                <p className="text-sm font-medium text-slate-700">
                  {data.frame_summary}
                </p>
              )}
              {data.insufficient_evidence && data.claims.length === 0 && (
                <p className="text-sm text-slate-400">
                  Bu entity için yeterli kanıt yok.
                </p>
              )}
              <ul className="space-y-2">
                {data.claims.map((claim, i) => (
                  <li key={i} className="text-sm text-slate-600">
                    {claim.text}{" "}
                    {claim.citations.map((cid) => {
                      const c = citById.get(cid);
                      const chip = (
                        <span className="ml-1 rounded bg-slate-100 px-1.5 py-0.5 text-xs font-mono text-slate-500">
                          {cid}
                        </span>
                      );
                      return c?.link ? (
                        <a key={cid} href={c.link} target="_blank" rel="noopener noreferrer">
                          {chip}
                        </a>
                      ) : (
                        <span key={cid}>{chip}</span>
                      );
                    })}
                  </li>
                ))}
              </ul>
              <p className="text-xs text-slate-400">
                {data.backend === "anthropic" ? "Claude" : "Şablon"}
                {data.cached ? " · önbellek" : ""} · yalnızca ({entityRef},{" "}
                {data.country_code}) kapsamında — ülke geneli iddiası değildir
              </p>
            </>
          )}
        </div>
      )}
    </div>
  );
}
