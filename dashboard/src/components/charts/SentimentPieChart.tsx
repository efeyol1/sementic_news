"use client";

import { PieChart, Pie, Cell, ResponsiveContainer, Tooltip } from "recharts";
import type { SentimentCounts } from "@/lib/api";

interface Props {
  counts: SentimentCounts;
}

export function SentimentPieChart({ counts }: Props) {
  const data = [
    { name: "Pozitif", value: counts.positive ?? 0, color: "#10B981" },
    { name: "Negatif", value: counts.negative ?? 0, color: "#EF4444" },
  ];

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
            background: "#131C35",
            border: "1px solid #1E2D4F",
            borderRadius: "8px",
            fontSize: 12,
            color: "#F1F5F9",
          }}
          formatter={(v: number) => [`${v} haber`, ""]}
        />
      </PieChart>
    </ResponsiveContainer>
  );
}
