import type { Metadata } from "next";
import "./globals.css";
import { Navbar } from "@/components/ui/Navbar";

export const metadata: Metadata = {
  title: "Semantic News",
  description:
    "Avrupa haber kaynaklarının günlük semantik analizi — 6 ülke, sentiment, NER, entity-narrative ve konu kümeleme",
};

export default function RootLayout({ children }: { children: React.ReactNode }) {
  return (
    // Sprint 7.5: html lang stays generic; per-country dates/numbers are
    // formatted via locale helpers that read the active country's language
    // off /api/countries instead of inheriting "tr" from the document.
    <html lang="en">
      <body className="min-h-screen bg-slate-50 antialiased">
        <Navbar />
        <main className="pt-16">{children}</main>
      </body>
    </html>
  );
}
