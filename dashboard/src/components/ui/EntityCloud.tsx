"use client";

import { motion } from "framer-motion";
import { User, Building2, MapPin } from "lucide-react";

interface Props {
  entities: {
    PER: string[];
    ORG: string[];
    LOC: string[];
  };
}

const types = [
  { key: "PER" as const, label: "Kişiler", icon: User, color: "#818CF8" },
  { key: "ORG" as const, label: "Kurumlar", icon: Building2, color: "#34D399" },
  { key: "LOC" as const, label: "Yerler", icon: MapPin, color: "#FB923C" },
];

export function EntityCloud({ entities }: Props) {
  return (
    <div className="space-y-5">
      {types.map(({ key, label, icon: Icon, color }) => (
        <div key={key}>
          <div className="flex items-center gap-2 mb-2.5">
            <Icon className="w-3.5 h-3.5" style={{ color }} />
            <span className="text-xs font-medium text-text-secondary uppercase tracking-wider">{label}</span>
          </div>
          <div className="flex flex-wrap gap-2">
            {(entities[key] ?? []).slice(0, 8).map((name, i) => (
              <motion.span
                key={name}
                initial={{ opacity: 0, scale: 0.8 }}
                animate={{ opacity: 1, scale: 1 }}
                transition={{ delay: i * 0.04, duration: 0.2 }}
                className="px-2.5 py-1 rounded-lg text-xs font-medium"
                style={{
                  background: `${color}18`,
                  color,
                  border: `1px solid ${color}30`,
                }}
              >
                {name}
              </motion.span>
            ))}
            {(entities[key] ?? []).length === 0 && (
              <span className="text-xs text-text-muted italic">Veri yok</span>
            )}
          </div>
        </div>
      ))}
    </div>
  );
}
