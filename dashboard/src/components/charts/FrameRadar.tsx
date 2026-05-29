"use client";

import {
  Radar,
  RadarChart,
  PolarGrid,
  PolarAngleAxis,
  PolarRadiusAxis,
  ResponsiveContainer,
  Tooltip,
} from "recharts";

// Canonical frame order + Turkish labels, kept in sync with
// src/analysis/frame_bridge.py::FRAME_NAMES and configs/frames/*.yaml.
const FRAME_LABELS: { key: string; label: string }[] = [
  { key: "economic", label: "Ekonomi" },
  { key: "security", label: "Güvenlik" },
  { key: "identity", label: "Kimlik" },
  { key: "governance", label: "Yönetim" },
  { key: "humanitarian", label: "İnsani" },
  { key: "conflict", label: "Çatışma" },
];

interface Props {
  // L1-normalized frame intensities, or null when the bridge produced no
  // signal for this (entity, country, window).
  intensities: Record<string, number> | null;
}

function CustomTooltip({ active, payload }: any) {
  if (!active || !payload?.length) return null;
  const d = payload[0].payload as { label: string; value: number };
  return (
    <div className="bg-white border border-slate-200 rounded-lg shadow-sm p-2.5 text-xs">
      <span className="font-semibold text-slate-700">{d.label}: </span>
      <span className="font-mono text-green-600">{(d.value * 100).toFixed(0)}%</span>
    </div>
  );
}

export function FrameRadar({ intensities }: Props) {
  if (!intensities) {
    return (
      <div className="flex items-center justify-center h-48 text-slate-400 text-sm text-center px-4">
        Bu entity'nin collocate'leri çerçeve sözlüğüyle eşleşmedi — frame
        sinyali yok
      </div>
    );
  }

  const data = FRAME_LABELS.map(({ key, label }) => ({
    label,
    value: intensities[key] ?? 0,
  }));

  return (
    <ResponsiveContainer width="100%" height={260}>
      <RadarChart data={data} margin={{ top: 10, right: 20, bottom: 10, left: 20 }}>
        <PolarGrid stroke="#e2e8f0" />
        <PolarAngleAxis
          dataKey="label"
          tick={{ fontSize: 11, fill: "#64748b" }}
        />
        <PolarRadiusAxis
          domain={[0, 1]}
          tick={false}
          axisLine={false}
        />
        <Tooltip content={<CustomTooltip />} />
        <Radar
          dataKey="value"
          stroke="#22c55e"
          fill="#22c55e"
          fillOpacity={0.35}
        />
      </RadarChart>
    </ResponsiveContainer>
  );
}
