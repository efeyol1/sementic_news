"use client";

import {
  LineChart,
  Line,
  XAxis,
  YAxis,
  CartesianGrid,
  Tooltip,
  ReferenceLine,
  ResponsiveContainer,
} from "recharts";

interface PsiPoint {
  date: string;
  psi: number;
  severity: string;
}

interface Props {
  points: PsiPoint[];
  locale: string;
}

const SEVERITY_LABELS: Record<string, string> = {
  stable: "Stabil",
  moderate: "Orta",
  significant: "Belirgin",
  insufficient: "Yetersiz",
};

function CustomTooltip({ active, payload, locale }: any) {
  if (!active || !payload?.length) return null;
  const d = payload[0].payload as PsiPoint;
  const formattedDate = new Date(d.date + "T12:00:00").toLocaleDateString(
    locale,
    { month: "short", day: "numeric" }
  );
  return (
    <div className="bg-white border border-slate-200 rounded-lg shadow-sm p-2.5 text-xs">
      <p className="font-semibold text-slate-700 mb-1">{formattedDate}</p>
      <div className="space-y-0.5">
        <div className="flex items-center gap-2">
          <span className="w-2 h-2 rounded-full bg-green-500 inline-block" />
          <span className="text-slate-500">PSI:</span>
          <span className="font-mono font-medium text-green-600">
            {d.psi.toFixed(3)}
          </span>
        </div>
        <div className="flex items-center gap-2">
          <span className="text-slate-400">Durum:</span>
          <span className="font-mono text-slate-600">
            {SEVERITY_LABELS[d.severity] ?? d.severity}
          </span>
        </div>
      </div>
    </div>
  );
}

export function DriftPsiChart({ points, locale }: Props) {
  if (points.length === 0) {
    return (
      <div className="flex items-center justify-center h-48 text-slate-400 text-sm text-center px-4">
        PSI geçmişi için yeterli drift raporu yok
      </div>
    );
  }

  const maxPsi = Math.max(...points.map((p) => p.psi));
  const yMax = Math.max(0.3, maxPsi * 1.1);

  const formatted = points.map((p) => ({
    ...p,
    label: new Date(p.date + "T12:00:00").toLocaleDateString(locale, {
      month: "short",
      day: "numeric",
    }),
  }));

  return (
    <ResponsiveContainer width="100%" height={300}>
      <LineChart
        data={formatted}
        margin={{ top: 10, right: 16, bottom: 0, left: -10 }}
      >
        <CartesianGrid strokeDasharray="3 3" stroke="#e2e8f0" />
        <XAxis
          dataKey="label"
          tick={{ fontSize: 11, fill: "#64748b" }}
          axisLine={false}
          tickLine={false}
          interval="preserveStartEnd"
        />
        <YAxis
          domain={[0, yMax]}
          tick={{ fontSize: 11, fill: "#64748b" }}
          axisLine={false}
          tickLine={false}
        />
        <Tooltip content={<CustomTooltip locale={locale} />} />
        <ReferenceLine
          y={0.1}
          stroke="#f59e0b"
          strokeDasharray="4 3"
          label={{ value: "orta", position: "insideTopRight", fontSize: 10, fill: "#f59e0b" }}
        />
        <ReferenceLine
          y={0.25}
          stroke="#ef4444"
          strokeDasharray="4 3"
          label={{ value: "belirgin", position: "insideTopRight", fontSize: 10, fill: "#ef4444" }}
        />
        <Line
          type="monotone"
          dataKey="psi"
          stroke="#22c55e"
          strokeWidth={2}
          dot={{ r: 2, fill: "#22c55e" }}
          activeDot={{ r: 4, fill: "#22c55e" }}
        />
      </LineChart>
    </ResponsiveContainer>
  );
}
