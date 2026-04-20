"use client";

import Link from "next/link";
import { usePathname } from "next/navigation";
import { LayoutDashboard, BarChart3, Globe2, Activity } from "lucide-react";
import { cn } from "@/lib/utils";

const links = [
  { href: "/", label: "Genel Bakış", icon: LayoutDashboard },
  { href: "/sources", label: "Kaynak Analizi", icon: BarChart3 },
];

export function Navbar() {
  const pathname = usePathname();

  return (
    <nav className="fixed top-0 left-0 right-0 z-50 border-b border-border-subtle" style={{ background: "rgba(10, 15, 30, 0.85)", backdropFilter: "blur(16px)" }}>
      <div className="max-w-7xl mx-auto px-6 h-16 flex items-center justify-between">
        {/* Logo */}
        <Link href="/" className="flex items-center gap-3 group">
          <div className="w-8 h-8 rounded-lg accent-gradient flex items-center justify-center glow-accent">
            <Globe2 className="w-4 h-4 text-white" />
          </div>
          <div>
            <span className="font-bold text-sm text-text-primary tracking-tight">Semantic News</span>
            <span className="ml-1.5 text-xs text-accent-glow font-medium">TR</span>
          </div>
        </Link>

        {/* Nav links */}
        <div className="flex items-center gap-1">
          {links.map(({ href, label, icon: Icon }) => (
            <Link
              key={href}
              href={href}
              className={cn("nav-item text-sm", pathname === href && "active")}
            >
              <Icon className="w-4 h-4" />
              {label}
            </Link>
          ))}
        </div>

        {/* Live indicator */}
        <div className="flex items-center gap-2 text-xs text-text-secondary">
          <Activity className="w-3.5 h-3.5 text-sentiment-positive" />
          <span>Canlı</span>
          <div className="w-1.5 h-1.5 rounded-full bg-sentiment-positive animate-pulse" />
        </div>
      </div>
    </nav>
  );
}
