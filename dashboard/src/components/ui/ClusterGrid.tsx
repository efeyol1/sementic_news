import Link from "next/link";
import { ArrowRight, Layers } from "lucide-react";

interface Cluster {
  cluster_id: number;
  title: string;
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
      {clusters.map((cluster) => {
        const query = date ? `?date=${date}` : "";
        return (
          <Link key={cluster.cluster_id} href={`/topic/${cluster.cluster_id}${query}`}>
            <div className="card p-4 hover:border-green-200 hover:shadow-md transition-all group cursor-pointer h-full">
              <div className="flex items-start justify-between mb-3">
                <div className="flex items-center gap-2.5">
                  <div className="w-8 h-8 rounded-lg bg-green-50 flex items-center justify-center">
                    <Layers className="w-4 h-4 text-green-600" />
                  </div>
                  <div>
                    <div className="text-sm font-semibold text-slate-800 leading-snug">{cluster.title}</div>
                    <div className="text-xs text-slate-400">{cluster.size} haber</div>
                  </div>
                </div>
                <ArrowRight className="w-4 h-4 text-slate-300 group-hover:text-green-600 group-hover:translate-x-0.5 transition-all shrink-0 mt-0.5" />
              </div>
              <div className="flex flex-wrap gap-1.5">
                {cluster.keywords.slice(0, 4).map((kw) => (
                  <span key={kw} className="keyword-chip">{kw}</span>
                ))}
              </div>
            </div>
          </Link>
        );
      })}
    </div>
  );
}
