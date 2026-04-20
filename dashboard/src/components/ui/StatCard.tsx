"use client";

import { motion } from "framer-motion";
import { AnimatedCounter } from "./AnimatedCounter";
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
  trend?: number;
  accent?: "default" | "positive" | "negative" | "neutral";
  delay?: number;
}

const accentMap = {
  default: { icon: "text-accent-glow", glow: "#6366F1" },
  positive: { icon: "text-sentiment-positive", glow: "#10B981" },
  negative: { icon: "text-sentiment-negative", glow: "#EF4444" },
  neutral: { icon: "text-yellow-400", glow: "#F59E0B" },
};

export function StatCard({ label, value, suffix = "", decimals = 0, iconName, trend, accent = "default", delay = 0 }: Props) {
  const colors = accentMap[accent];
  const Icon = ICONS[iconName];

  return (
    <motion.div
      initial={{ opacity: 0, y: 20 }}
      animate={{ opacity: 1, y: 0 }}
      transition={{ duration: 0.4, delay: delay / 1000, ease: "easeOut" }}
      className="glass-card p-6 group cursor-default"
    >
      <div className="flex items-start justify-between mb-4">
        <div
          className="w-10 h-10 rounded-xl flex items-center justify-center"
          style={{ background: `${colors.glow}18` }}
        >
          <Icon className={cn("w-5 h-5", colors.icon)} />
        </div>
        {trend !== undefined && (
          <span
            className={cn(
              "text-xs font-medium px-2 py-0.5 rounded-full",
              trend >= 0 ? "text-sentiment-positive bg-sentiment-positiveMuted" : "text-sentiment-negative bg-sentiment-negativeMuted"
            )}
          >
            {trend >= 0 ? "+" : ""}
            {trend.toFixed(1)}%
          </span>
        )}
      </div>
      <div className="text-3xl font-bold text-text-primary mb-1 font-mono">
        <AnimatedCounter value={value} suffix={suffix} decimals={decimals} />
      </div>
      <div className="text-sm text-text-secondary">{label}</div>
    </motion.div>
  );
}
