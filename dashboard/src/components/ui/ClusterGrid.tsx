"use client";

import { motion } from "framer-motion";
import Link from "next/link";
import { ArrowRight, Layers } from "lucide-react";

interface Cluster {
  cluster_id: number;
  size: number;
  keywords: string[];
}

interface Props {
  clusters: Cluster[];
  date?: string;
}

export function ClusterGrid({ clusters, date }: Props) {
  return (
    <div className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-3 gap-3">
      {clusters.map((cluster, i) => {
        const query = date ? `?date=${date}` : "";
        return (
          <motion.div
            key={cluster.cluster_id}
            initial={{ opacity: 0, y: 16 }}
            animate={{ opacity: 1, y: 0 }}
            transition={{ delay: i * 0.06, duration: 0.35, ease: "easeOut" }}
          >
            <Link href={`/topic/${cluster.cluster_id}${query}`}>
              <div className="glass-card p-4 group cursor-pointer h-full">
                <div className="flex items-start justify-between mb-3">
                  <div className="flex items-center gap-2">
                    <div className="w-8 h-8 rounded-lg flex items-center justify-center bg-accent-muted border border-border-glow">
                      <Layers className="w-4 h-4 text-accent-glow" />
                    </div>
                    <div>
                      <div className="text-xs text-text-muted">Küme #{cluster.cluster_id}</div>
                      <div className="text-sm font-semibold text-text-primary">{cluster.size} haber</div>
                    </div>
                  </div>
                  <ArrowRight className="w-4 h-4 text-text-muted group-hover:text-accent-glow group-hover:translate-x-0.5 transition-all" />
                </div>
                <div className="flex flex-wrap gap-1.5">
                  {cluster.keywords.slice(0, 4).map((kw) => (
                    <span key={kw} className="keyword-chip">{kw}</span>
                  ))}
                </div>
              </div>
            </Link>
          </motion.div>
        );
      })}
    </div>
  );
}
