"use client";

import {
  LineChart,
  Line,
  XAxis,
  YAxis,
  CartesianGrid,
  Tooltip,
  ResponsiveContainer,
  ReferenceLine,
} from "recharts";
import { format, parseISO } from "date-fns";
import { tr } from "date-fns/locale";
import type { TrendPoint } from "@/lib/api";

interface Props {
  points: TrendPoint[];
}

function CustomTooltip({ active, payload, label }: any) {
  if (!active || !payload?.length) return null;
  const d = payload[0].payload as TrendPoint;
  return (
    <div className="bg-white border border-slate-200 rounded-lg shadow-sm p-3 text-xs">
      <p className="font-semibold text-slate-700 mb-1.5">
        {format(parseISO(d.date), "d MMMM yyyy", { locale: tr })}
      </p>
      <div className="space-y-0.5">
        <div className="flex items-center gap-2">
          <span className="w-2 h-2 rounded-full bg-green-500 inline-block" />
          <span className="text-slate-500">Pozitif:</span>
          <span className="font-mono font-medium text-green-600">
            {d.positive} (%{d.positive_pct})
          </span>
        </div>
        {(d.neutral ?? 0) > 0 && (
          <div className="flex items-center gap-2">
            <span className="w-2 h-2 rounded-full bg-slate-400 inline-block" />
            <span className="text-slate-500">Nötr:</span>
            <span className="font-mono font-medium text-slate-500">{d.neutral}</span>
          </div>
        )}
        <div className="flex items-center gap-2">
          <span className="w-2 h-2 rounded-full bg-red-400 inline-block" />
          <span className="text-slate-500">Negatif:</span>
          <span className="font-mono font-medium text-red-500">{d.negative}</span>
        </div>
        <div className="flex items-center gap-2 pt-0.5 border-t border-slate-100 mt-1">
          <span className="text-slate-400">Toplam:</span>
          <span className="font-mono text-slate-600">{d.total} haber</span>
        </div>
      </div>
    </div>
  );
}

export function SentimentTrendChart({ points }: Props) {
  if (points.length === 0) {
    return (
      <div className="flex items-center justify-center h-48 text-slate-400 text-sm">
        Yeterli tarihsel veri yok
      </div>
    );
  }

  const formatted = points.map((p) => ({
    ...p,
    label: format(parseISO(p.date), "d MMM", { locale: tr }),
  }));
  const hasNeutral = points.some((p) => (p.neutral ?? 0) > 0);

  return (
    <ResponsiveContainer width="100%" height={220}>
      <LineChart data={formatted} margin={{ top: 5, right: 10, bottom: 0, left: -10 }}>
        <CartesianGrid strokeDasharray="3 3" stroke="#f1f5f9" />
        <XAxis
          dataKey="label"
          tick={{ fontSize: 11, fill: "#94a3b8" }}
          axisLine={false}
          tickLine={false}
          interval="preserveStartEnd"
        />
        <YAxis
          tick={{ fontSize: 11, fill: "#94a3b8" }}
          axisLine={false}
          tickLine={false}
        />
        <Tooltip content={<CustomTooltip />} />
        <ReferenceLine y={0} stroke="#e2e8f0" />
        <Line
          type="monotone"
          dataKey="positive"
          name="Pozitif"
          stroke="#22c55e"
          strokeWidth={2}
          dot={false}
          activeDot={{ r: 4, fill: "#22c55e" }}
        />
        {hasNeutral && (
          <Line
            type="monotone"
            dataKey="neutral"
            name="Nötr"
            stroke="#94a3b8"
            strokeWidth={1.5}
            strokeDasharray="4 4"
            dot={false}
            activeDot={{ r: 4, fill: "#94a3b8" }}
          />
        )}
        <Line
          type="monotone"
          dataKey="negative"
          name="Negatif"
          stroke="#f87171"
          strokeWidth={2}
          dot={false}
          activeDot={{ r: 4, fill: "#f87171" }}
        />
      </LineChart>
    </ResponsiveContainer>
  );
}
