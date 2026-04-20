"use client";

import { motion } from "framer-motion";
import type { SourceComparison } from "@/lib/api";

interface Props {
  data: SourceComparison;
}

export function SourceHeatmap({ data }: Props) {
  const sources = Object.entries(data.sources).sort(
    (a, b) => b[1].sentiment_percentages.negative - a[1].sentiment_percentages.negative
  );

  return (
    <div className="space-y-2.5">
      {sources.map(([name, stats], i) => {
        const pos = stats.sentiment_percentages.positive ?? 0;
        const neg = stats.sentiment_percentages.negative ?? 0;

        return (
          <motion.div
            key={name}
            initial={{ opacity: 0, x: -12 }}
            animate={{ opacity: 1, x: 0 }}
            transition={{ delay: i * 0.05, duration: 0.3 }}
            className="flex items-center gap-3"
          >
            <div className="w-24 text-xs text-text-secondary font-medium truncate shrink-0 text-right">
              {name}
            </div>
            <div className="flex-1 h-6 rounded-lg overflow-hidden flex gap-0.5">
              <div
                className="h-full rounded-l-lg flex items-center justify-center text-xs font-bold text-white"
                style={{ width: `${pos}%`, background: "#10B981", minWidth: pos > 0 ? 4 : 0 }}
              >
                {pos > 12 && `${pos.toFixed(0)}%`}
              </div>
              <div
                className="h-full rounded-r-lg flex items-center justify-center text-xs font-bold text-white"
                style={{ width: `${neg}%`, background: "#EF4444", minWidth: neg > 0 ? 4 : 0 }}
              >
                {neg > 12 && `${neg.toFixed(0)}%`}
              </div>
            </div>
            <div className="w-14 text-right">
              <span className="text-xs font-mono text-text-muted">{stats.total}</span>
            </div>
          </motion.div>
        );
      })}

      <div className="flex items-center gap-4 pt-2 border-t border-border-subtle">
        {[{ color: "#10B981", label: "Pozitif" }, { color: "#EF4444", label: "Negatif" }].map(({ color, label }) => (
          <div key={label} className="flex items-center gap-1.5">
            <div className="w-2.5 h-2.5 rounded-sm" style={{ background: color }} />
            <span className="text-xs text-text-muted">{label}</span>
          </div>
        ))}
      </div>
    </div>
  );
}
