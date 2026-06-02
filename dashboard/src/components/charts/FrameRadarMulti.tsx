"use client";

import {
  Radar,
  RadarChart,
  PolarGrid,
  PolarAngleAxis,
  PolarRadiusAxis,
  ResponsiveContainer,
  Tooltip,
  Legend,
} from "recharts";
import { FRAME_LABELS } from "./FrameRadar";

export interface FrameSeries {
  code: string; // ISO code, e.g. "TR"
  name: string; // display name, e.g. "Türkiye"
  // L1-normalized 6-frame intensities, or null when this country has no
  // frame signal for the entity (collocates matched no frame seed).
  intensities: Record<string, number> | null;
}

interface Props {
  series: FrameSeries[];
  activeCode?: string; // the page's active country — drawn green + emphasized
}

// Active country is green to match the single-entity FrameRadar; the rest cycle
// through a fixed palette so a given country keeps the same color across renders.
const ACTIVE_COLOR = "#22c55e";
const PALETTE = ["#3b82f6", "#f59e0b", "#8b5cf6", "#f43f5e", "#14b8a6", "#64748b"];

function MultiTooltip({ active, payload, nameByCode, colorByCode }: any) {
  if (!active || !payload?.length) return null;
  const label = payload[0]?.payload?.label;
  // Sort desc so the dominant framing for this axis reads first.
  const rows = [...payload].sort((a, b) => b.value - a.value);
  return (
    <div className="bg-white border border-slate-200 rounded-lg shadow-sm p-2.5 text-xs space-y-1 min-w-[140px]">
      <div className="font-semibold text-slate-700 mb-1">{label}</div>
      {rows.map((r: any) => (
        <div key={r.dataKey} className="flex items-center gap-2">
          <span
            className="inline-block w-2 h-2 rounded-full shrink-0"
            style={{ background: colorByCode[r.dataKey] }}
          />
          <span className="text-slate-600">{nameByCode[r.dataKey] ?? r.dataKey}</span>
          <span className="font-mono text-slate-500 ml-auto">
            {(r.value * 100).toFixed(0)}%
          </span>
        </div>
      ))}
    </div>
  );
}

/** Cross-country framing overlay: the same entity's 6-frame intensity profile
 *  drawn as one radar series per country. The headline visual of the
 *  entity-narrative track — surfaces how TR/DE/FR frame the same actor. */
export function FrameRadarMulti({ series, activeCode }: Props) {
  const withSignal = series.filter((s) => s.intensities);

  if (withSignal.length < 2) {
    return (
      <div className="flex items-center justify-center h-48 text-slate-400 text-sm text-center px-4">
        Çok ülkeli çerçeve karşılaştırması için en az iki ülkede frame sinyali
        gerekiyor.
      </div>
    );
  }

  // Active country first so it draws on top and reads as the reference.
  const ordered = [...withSignal].sort((a, b) => {
    if (a.code === activeCode) return -1;
    if (b.code === activeCode) return 1;
    return 0;
  });

  let paletteIdx = 0;
  const colorByCode: Record<string, string> = {};
  const nameByCode: Record<string, string> = {};
  for (const s of ordered) {
    colorByCode[s.code] =
      s.code === activeCode ? ACTIVE_COLOR : PALETTE[paletteIdx++ % PALETTE.length];
    nameByCode[s.code] = s.name;
  }

  const data = FRAME_LABELS.map(({ key, label }) => {
    const row: Record<string, number | string> = { label };
    for (const s of ordered) row[s.code] = s.intensities![key] ?? 0;
    return row;
  });

  return (
    <ResponsiveContainer width="100%" height={320}>
      <RadarChart data={data} margin={{ top: 10, right: 30, bottom: 10, left: 30 }}>
        <PolarGrid stroke="#e2e8f0" />
        <PolarAngleAxis dataKey="label" tick={{ fontSize: 11, fill: "#64748b" }} />
        <PolarRadiusAxis domain={[0, 1]} tick={false} axisLine={false} />
        <Tooltip
          content={<MultiTooltip nameByCode={nameByCode} colorByCode={colorByCode} />}
        />
        <Legend
          formatter={(value: string) => nameByCode[value] ?? value}
          wrapperStyle={{ fontSize: 12 }}
        />
        {ordered.map((s) => {
          const isActive = s.code === activeCode;
          return (
            <Radar
              key={s.code}
              name={s.code}
              dataKey={s.code}
              stroke={colorByCode[s.code]}
              fill={colorByCode[s.code]}
              fillOpacity={isActive ? 0.3 : 0.1}
              strokeWidth={isActive ? 2 : 1.5}
            />
          );
        })}
      </RadarChart>
    </ResponsiveContainer>
  );
}
