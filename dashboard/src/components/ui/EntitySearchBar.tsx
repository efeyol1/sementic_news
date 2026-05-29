"use client";

import { useState } from "react";
import { useRouter, usePathname, useSearchParams } from "next/navigation";
import { Search } from "lucide-react";
import { preserveQueryParams } from "@/lib/url";

interface Props {
  initialQuery: string;
  entityType: string; // "" | PER | ORG | LOC
}

const TYPES = [
  { value: "", label: "Tümü" },
  { value: "PER", label: "Kişi" },
  { value: "ORG", label: "Kurum" },
  { value: "LOC", label: "Yer" },
];

export function EntitySearchBar({ initialQuery, entityType }: Props) {
  const router = useRouter();
  const pathname = usePathname();
  const searchParams = useSearchParams();
  const [q, setQ] = useState(initialQuery);

  function push(overrides: Record<string, string | undefined>) {
    // Keep ?country (and window) while updating the search — same param
    // preservation contract as the DatePicker (Sprint 7.5).
    router.push(preserveQueryParams(pathname, searchParams, overrides));
  }

  function onSubmit(e: React.FormEvent) {
    e.preventDefault();
    push({ q: q.trim() || undefined });
  }

  return (
    <form onSubmit={onSubmit} className="flex items-center gap-2">
      <div className="flex items-center gap-2 card px-3 py-2 flex-1">
        <Search className="w-4 h-4 text-slate-400 shrink-0" />
        <input
          value={q}
          onChange={(e) => setQ(e.target.value)}
          placeholder="Entity ara: Trump, Erdoğan, Rumänien…"
          className="text-sm text-slate-700 bg-transparent border-none outline-none w-full"
        />
      </div>
      <select
        value={entityType}
        onChange={(e) => push({ type: e.target.value || undefined })}
        className="text-sm text-slate-700 bg-white border border-slate-200 rounded-lg px-3 py-2 shadow-sm cursor-pointer font-medium outline-none"
      >
        {TYPES.map((t) => (
          <option key={t.value} value={t.value}>
            {t.label}
          </option>
        ))}
      </select>
      <button
        type="submit"
        className="text-sm font-medium text-white bg-green-600 hover:bg-green-700 transition-colors rounded-lg px-4 py-2 shadow-sm"
      >
        Ara
      </button>
    </form>
  );
}
