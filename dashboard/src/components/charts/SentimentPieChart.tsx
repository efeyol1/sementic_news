"use client";

import { PieChart, Pie, Cell, ResponsiveContainer, Tooltip } from "recharts";
import type { SentimentCounts } from "@/lib/api";

interface Props {
  counts: SentimentCounts;
}

export function SentimentPieChart({ counts }: Props) {
  const data = [
    { name: "Pozitif", value: counts.positive ?? 0, color: "#22c55e" },
    { name: "Nötr",    value: counts.neutral ?? 0,  color: "#94a3b8" },
    { name: "Negatif", value: counts.negative ?? 0, color: "#f87171" },
  ].filter((d) => d.value > 0);

  return (
    <ResponsiveContainer width="100%" height={200}>
      <PieChart>
        <Pie
          data={data}
          cx="50%"
          cy="50%"
          innerRadius={55}
          outerRadius={80}
          paddingAngle={3}
          dataKey="value"
          strokeWidth={0}
        >
          {data.map((entry) => (
            <Cell key={entry.name} fill={entry.color} />
          ))}
        </Pie>
        <Tooltip
          contentStyle={{
            background: "#ffffff",
            border: "1px solid #e2e8f0",
            borderRadius: "8px",
            fontSize: 12,
            color: "#0f172a",
          }}
          formatter={(v: number) => [`${v} haber`, ""]}
        />
      </PieChart>
    </ResponsiveContainer>
  );
}
