"use client";

import { useEffect, useState, useTransition } from "react";
import { usePathname, useRouter, useSearchParams } from "next/navigation";
import { Globe } from "lucide-react";
import { api, DEFAULT_COUNTRY, type CountryInfo } from "@/lib/api";

/**
 * Header country selector — Phase 6.
 *
 * Fetches `/api/countries` on mount and lets the user switch the active
 * country via the `?country=<slug>` URL param. While the V1 pilot only
 * ships Turkey, the selector still renders so adding a new YAML in
 * `configs/countries/` is a zero-frontend-change rollout.
 */
export function CountrySelector() {
  const router = useRouter();
  const pathname = usePathname();
  const searchParams = useSearchParams();
  const [, startTransition] = useTransition();

  const [countries, setCountries] = useState<CountryInfo[]>([]);
  const selected = searchParams.get("country") || DEFAULT_COUNTRY;

  useEffect(() => {
    let cancelled = false;
    api
      .countries()
      .then((data) => {
        if (!cancelled) setCountries(data);
      })
      .catch(() => {
        // Fail-soft: keep the default Turkey option so the selector still
        // renders even if /api/countries is briefly unreachable.
        if (!cancelled) {
          setCountries([
            { code: "TR", slug: "turkey", name: "Turkey", language: "tr", status: "active" },
          ]);
        }
      });
    return () => {
      cancelled = true;
    };
  }, []);

  function onChange(e: React.ChangeEvent<HTMLSelectElement>) {
    const next = e.target.value;
    const params = new URLSearchParams(searchParams.toString());
    if (next === DEFAULT_COUNTRY) {
      params.delete("country");
    } else {
      params.set("country", next);
    }
    const qs = params.toString();
    startTransition(() => {
      router.push(qs ? `${pathname}?${qs}` : pathname);
    });
  }

  return (
    <div className="flex items-center gap-1.5 text-xs text-slate-500">
      <Globe className="w-3.5 h-3.5" />
      <select
        value={selected}
        onChange={onChange}
        aria-label="Ülke seç"
        className="bg-transparent border-0 text-xs text-slate-600 font-medium focus:ring-0 focus:outline-none cursor-pointer pr-1"
      >
        {countries.length === 0 ? (
          <option value={DEFAULT_COUNTRY}>Yükleniyor…</option>
        ) : (
          countries.map((c) => (
            <option key={c.slug} value={c.slug}>
              {c.name}
            </option>
          ))
        )}
      </select>
    </div>
  );
}
