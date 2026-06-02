import { api } from "@/lib/api";

export async function Footer() {
  const about = await api.about().catch(() => null);

  if (!about) return null;

  const links: { label: string; href: string }[] = [
    { label: "Repo",              href: about.links.repository },
    { label: "Lisans",            href: about.links.license },
    { label: "Model Kartı",       href: about.links.model_card },
    { label: "Veri Kaynağı",      href: about.links.data_provenance },
    { label: "Veri Lisansı",      href: about.links.data_license },
    { label: "Gizlilik",          href: about.links.privacy },
    { label: "Katkı",             href: about.links.contributing },
    { label: "Davranış Kuralları", href: about.links.code_of_conduct },
    { label: "Güvenlik",          href: about.links.security },
  ];

  return (
    <footer className="border-t border-slate-200 mt-12">
      <div className="max-w-7xl mx-auto px-6 py-8 text-xs text-slate-400 space-y-3">
        <p>{about.disclaimer_tr}</p>
        <div className="flex flex-wrap gap-x-4 gap-y-2">
          {links.map(({ label, href }) => (
            <a
              key={label}
              href={href}
              target="_blank"
              rel="noopener noreferrer"
              className="hover:text-slate-600"
            >
              {label}
            </a>
          ))}
        </div>
        <p className="text-slate-300">
          {about.name} v{about.version}
        </p>
      </div>
    </footer>
  );
}
