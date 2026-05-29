"use client";

import { useRouter, usePathname, useSearchParams } from "next/navigation";
import { Calendar } from "lucide-react";
import { format, parseISO } from "date-fns";
import { tr } from "date-fns/locale";
import { preserveQueryParams } from "@/lib/url";

interface Props {
  availableDates: string[];
  currentDate: string;
}

export function DatePicker({ availableDates, currentDate }: Props) {
  const router = useRouter();
  const pathname = usePathname();
  const searchParams = useSearchParams();

  const sorted = [...availableDates].sort((a, b) => b.localeCompare(a));

  function handleChange(e: React.ChangeEvent<HTMLSelectElement>) {
    // Sprint 7.5: keep ?country (and any other existing param) when
    // switching dates — the old `?date=…` reset dropped country selection.
    router.push(
      preserveQueryParams(pathname, searchParams, { date: e.target.value }),
    );
  }

  function formatDate(iso: string) {
    try {
      return format(parseISO(iso), "d MMM yyyy", { locale: tr });
    } catch {
      return iso;
    }
  }

  if (sorted.length === 0) return null;

  return (
    <div className="flex items-center gap-2 bg-white border border-slate-200 rounded-lg px-3 py-2 shadow-sm">
      <Calendar className="w-4 h-4 text-slate-400 shrink-0" />
      <select
        value={currentDate}
        onChange={handleChange}
        className="text-sm text-slate-700 bg-transparent border-none outline-none cursor-pointer font-medium"
      >
        {sorted.map((d) => (
          <option key={d} value={d}>
            {formatDate(d)}
          </option>
        ))}
      </select>
    </div>
  );
}
