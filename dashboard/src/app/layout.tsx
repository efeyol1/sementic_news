import type { Metadata } from "next";
import "./globals.css";
import { Navbar } from "@/components/ui/Navbar";

export const metadata: Metadata = {
  title: "Semantic News TR",
  description: "Türk haber kaynaklarının günlük semantik analizi — sentiment, NER ve konu kümeleme",
};

export default function RootLayout({ children }: { children: React.ReactNode }) {
  return (
    <html lang="tr">
      <body className="min-h-screen bg-slate-50 antialiased">
        <Navbar />
        <main className="pt-16">{children}</main>
      </body>
    </html>
  );
}
