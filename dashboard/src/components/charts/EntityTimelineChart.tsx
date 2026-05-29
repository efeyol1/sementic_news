"use client";

import {
  LineChart,
  Line,
  XAxis,
  YAxis,
  CartesianGrid,
  Tooltip,
  ResponsiveContainer,
} from "recharts";
import { format, parseISO } from "date-fns";
import { tr } from "date-fns/locale";
import type { EntityTimelinePoint } from "@/lib/api";

interface Props {
  points: EntityTimelinePoint[];
}

function CustomTooltip({ active, payload }: any) {
  if (!active || !payload?.length) return null;
  const d = payload[0].payload as EntityTimelinePoint;
  return (
    <div className="bg-white border border-slate-200 rounded-lg shadow-sm p-3 text-xs">
      <p className="font-semibold text-slate-700 mb-1.5">
        {format(parseISO(d.date), "d MMMM yyyy", { locale: tr })}
      </p>
      <div className="space-y-0.5">
        <div className="flex items-center gap-2">
          <span className="w-2 h-2 rounded-full bg-green-500 inline-block" />
          <span className="text-slate-500">Mention:</span>
          <span className="font-mono font-medium text-green-600">
            {d.mention_count}
          </span>
        </div>
        <div className="flex items-center gap-2">
          <span className="text-slate-400">Cooccurrence:</span>
          <span className="font-mono text-slate-600">{d.total_cooccurrences}</span>
        </div>
        {d.avg_log_likelihood != null && (
          <div className="flex items-center gap-2">
            <span className="text-slate-400">Ø LLR:</span>
            <span className="font-mono text-slate-600">
              {d.avg_log_likelihood.toFixed(1)}
            </span>
          </div>
        )}
      </div>
    </div>
  );
}

export function EntityTimelineChart({ points }: Props) {
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
          allowDecimals={false}
        />
        <Tooltip content={<CustomTooltip />} />
        <Line
          type="monotone"
          dataKey="mention_count"
          name="Mention"
          stroke="#22c55e"
          strokeWidth={2}
          dot={{ r: 2, fill: "#22c55e" }}
          activeDot={{ r: 4, fill: "#22c55e" }}
        />
      </LineChart>
    </ResponsiveContainer>
  );
}
