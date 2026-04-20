"use client";

import { motion } from "framer-motion";
import type { SentimentCounts } from "@/lib/api";

interface Props {
  counts: SentimentCounts;
}

export function SentimentGauge({ counts }: Props) {
  const pos = counts.positive ?? 0;
  const neg = counts.negative ?? 0;
  const total = pos + neg || 1;
  const positivePct = (pos / total) * 100;
  const negativePct = (neg / total) * 100;

  const segments = [
    { label: "Pozitif", value: positivePct, color: "#10B981", count: pos },
    { label: "Negatif", value: negativePct, color: "#EF4444", count: neg },
  ];

  return (
    <div className="space-y-4">
      <div className="h-3 rounded-full overflow-hidden flex gap-0.5 bg-bg-secondary">
        {segments.map((seg, i) => (
          <motion.div
            key={seg.label}
            initial={{ width: 0 }}
            animate={{ width: `${seg.value}%` }}
            transition={{ duration: 0.8, delay: i * 0.1, ease: "easeOut" }}
            className="h-full rounded-full"
            style={{ backgroundColor: seg.color, minWidth: seg.value > 0 ? 4 : 0 }}
          />
        ))}
      </div>
      <div className="grid grid-cols-2 gap-3">
        {segments.map((seg) => (
          <div key={seg.label} className="text-center">
            <div className="text-2xl font-bold font-mono" style={{ color: seg.color }}>
              {seg.value.toFixed(1)}%
            </div>
            <div className="text-xs text-text-secondary mt-0.5">{seg.label}</div>
            <div className="text-xs text-text-muted">{seg.count.toLocaleString("tr-TR")} haber</div>
          </div>
        ))}
      </div>
    </div>
  );
}
