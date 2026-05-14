"use client";

import Link from "next/link";
import { usePathname } from "next/navigation";
import { LayoutDashboard, BarChart3, Globe2 } from "lucide-react";
import { cn } from "@/lib/utils";
import { CountrySelector } from "@/components/ui/CountrySelector";

const links = [
  { href: "/", label: "Genel Bakış", icon: LayoutDashboard },
  { href: "/sources", label: "Kaynak Analizi", icon: BarChart3 },
];

export function Navbar() {
  const pathname = usePathname();

  return (
    <nav className="fixed top-0 left-0 right-0 z-50 bg-white border-b border-slate-200">
      <div className="max-w-7xl mx-auto px-6 h-16 flex items-center justify-between">
        <Link href="/" className="flex items-center gap-2.5">
          <div className="w-8 h-8 rounded-lg bg-green-600 flex items-center justify-center">
            <Globe2 className="w-4 h-4 text-white" />
          </div>
          <span className="font-semibold text-slate-900 text-sm tracking-tight">
            Semantic News <span className="text-green-600">TR</span>
          </span>
        </Link>

        <div className="flex items-center gap-1">
          {links.map(({ href, label, icon: Icon }) => (
            <Link
              key={href}
              href={href}
              className={cn("nav-item", pathname === href && "active")}
            >
              <Icon className="w-4 h-4" />
              {label}
            </Link>
          ))}
        </div>

        <div className="flex items-center gap-4">
          <CountrySelector />
          <div className="flex items-center gap-2 text-xs text-slate-500">
            <div className="w-1.5 h-1.5 rounded-full bg-green-500" />
            <span>Güncel</span>
          </div>
        </div>
      </div>
    </nav>
  );
}
