import { User, Building2, MapPin } from "lucide-react";

interface Props {
  entities: {
    PER: string[];
    ORG: string[];
    LOC: string[];
  };
}

const types = [
  { key: "PER" as const, label: "Kişiler", icon: User, chipClass: "bg-violet-50 text-violet-700 border-violet-200" },
  { key: "ORG" as const, label: "Kurumlar", icon: Building2, chipClass: "bg-blue-50 text-blue-700 border-blue-200" },
  { key: "LOC" as const, label: "Yerler", icon: MapPin, chipClass: "bg-amber-50 text-amber-700 border-amber-200" },
];

export function EntityCloud({ entities }: Props) {
  return (
    <div className="space-y-4">
      {types.map(({ key, label, icon: Icon, chipClass }) => (
        <div key={key}>
          <div className="flex items-center gap-1.5 mb-2">
            <Icon className="w-3.5 h-3.5 text-slate-400" />
            <span className="text-xs font-medium text-slate-500 uppercase tracking-wider">{label}</span>
          </div>
          <div className="flex flex-wrap gap-1.5">
            {(entities[key] ?? []).slice(0, 8).map((name) => (
              <span
                key={name}
                className={`px-2 py-0.5 rounded-md text-xs font-medium border ${chipClass}`}
              >
                {name}
              </span>
            ))}
            {(entities[key] ?? []).length === 0 && (
              <span className="text-xs text-slate-400 italic">Veri yok</span>
            )}
          </div>
        </div>
      ))}
    </div>
  );
}
