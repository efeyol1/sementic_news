import { cn } from "@/lib/utils";
import { Newspaper, Globe2, TrendingUp, Layers, TrendingDown, Activity } from "lucide-react";

const ICONS = { Newspaper, Globe2, TrendingUp, Layers, TrendingDown, Activity } as const;
export type IconName = keyof typeof ICONS;

interface Props {
  label: string;
  value: number;
  suffix?: string;
  decimals?: number;
  iconName: IconName;
  accent?: "default" | "positive" | "negative";
}

const accentMap = {
  default: { iconBg: "bg-slate-100", iconColor: "text-slate-500" },
  positive: { iconBg: "bg-green-50", iconColor: "text-green-600" },
  negative: { iconBg: "bg-red-50", iconColor: "text-red-500" },
};

export function StatCard({ label, value, suffix = "", decimals = 0, iconName, accent = "default" }: Props) {
  const colors = accentMap[accent];
  const Icon = ICONS[iconName];

  const formatted = value
    .toFixed(decimals)
    .replace(/\B(?=(\d{3})+(?!\d))/g, ".");

  return (
    <div className="card p-5">
      <div className="flex items-center justify-between mb-3">
        <div className={cn("w-9 h-9 rounded-lg flex items-center justify-center", colors.iconBg)}>
          <Icon className={cn("w-4.5 h-4.5", colors.iconColor)} />
        </div>
      </div>
      <div className="text-2xl font-bold text-slate-900 font-mono tabular-nums">
        {formatted}{suffix}
      </div>
      <div className="text-sm text-slate-500 mt-0.5">{label}</div>
    </div>
  );
}
