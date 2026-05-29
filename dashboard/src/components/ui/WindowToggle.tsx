"use client";

import { useRouter, usePathname, useSearchParams } from "next/navigation";
import { cn } from "@/lib/utils";
import { preserveQueryParams } from "@/lib/url";

interface Props {
  current: number; // 7 or 30
}

const OPTIONS = [7, 30];

/** 7g / 30g rolling-window switch — drives the `?window=` param that the
 *  profile + compare endpoints read. 30 is the default and drops the param. */
export function WindowToggle({ current }: Props) {
  const router = useRouter();
  const pathname = usePathname();
  const searchParams = useSearchParams();

  function select(w: number) {
    router.push(
      preserveQueryParams(pathname, searchParams, {
        window: w === 30 ? undefined : String(w),
      }),
    );
  }

  return (
    <div className="inline-flex items-center rounded-lg border border-slate-200 bg-white p-0.5 shadow-sm">
      {OPTIONS.map((w) => (
        <button
          key={w}
          onClick={() => select(w)}
          className={cn(
            "px-3 py-1 text-xs font-medium rounded-md transition-colors",
            current === w
              ? "bg-green-600 text-white"
              : "text-slate-500 hover:text-slate-700",
          )}
        >
          {w} gün
        </button>
      ))}
    </div>
  );
}
