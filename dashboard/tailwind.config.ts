import type { Config } from "tailwindcss";

const config: Config = {
  content: [
    "./src/pages/**/*.{js,ts,jsx,tsx,mdx}",
    "./src/components/**/*.{js,ts,jsx,tsx,mdx}",
    "./src/app/**/*.{js,ts,jsx,tsx,mdx}",
  ],
  theme: {
    extend: {
      colors: {
        bg: {
          primary: "#0A0F1E",
          secondary: "#0F1629",
          card: "#131C35",
          hover: "#1A2444",
        },
        accent: {
          primary: "#6366F1",
          glow: "#818CF8",
          muted: "#6366F120",
        },
        sentiment: {
          positive: "#10B981",
          neutral: "#F59E0B",
          negative: "#EF4444",
          positiveMuted: "#10B98120",
          negativeMuted: "#EF444420",
        },
        border: {
          subtle: "#1E2D4F",
          glow: "#6366F140",
        },
        text: {
          primary: "#F1F5F9",
          secondary: "#94A3B8",
          muted: "#475569",
        },
      },
      fontFamily: {
        sans: ["Inter", "system-ui", "sans-serif"],
        mono: ["JetBrains Mono", "monospace"],
      },
      backgroundImage: {
        "gradient-radial": "radial-gradient(var(--tw-gradient-stops))",
        "gradient-conic": "conic-gradient(from 180deg at 50% 50%, var(--tw-gradient-stops))",
        "card-gradient": "linear-gradient(135deg, #131C35 0%, #0F1629 100%)",
        "accent-gradient": "linear-gradient(135deg, #6366F1 0%, #818CF8 100%)",
        "glow-gradient": "radial-gradient(ellipse at center, #6366F120 0%, transparent 70%)",
      },
      boxShadow: {
        card: "0 4px 24px rgba(0, 0, 0, 0.4), 0 1px 0 rgba(255, 255, 255, 0.05) inset",
        "card-hover": "0 8px 32px rgba(0, 0, 0, 0.5), 0 0 0 1px #6366F130",
        glow: "0 0 20px rgba(99, 102, 241, 0.3)",
        "glow-positive": "0 0 20px rgba(16, 185, 129, 0.3)",
        "glow-negative": "0 0 20px rgba(239, 68, 68, 0.3)",
      },
      animation: {
        "fade-up": "fadeUp 0.5s ease-out forwards",
        "fade-in": "fadeIn 0.3s ease-out forwards",
        "pulse-slow": "pulse 3s ease-in-out infinite",
        "shimmer": "shimmer 2s linear infinite",
        "counter": "counter 1.5s ease-out forwards",
      },
      keyframes: {
        fadeUp: {
          from: { opacity: "0", transform: "translateY(16px)" },
          to: { opacity: "1", transform: "translateY(0)" },
        },
        fadeIn: {
          from: { opacity: "0" },
          to: { opacity: "1" },
        },
        shimmer: {
          "0%": { backgroundPosition: "-200% 0" },
          "100%": { backgroundPosition: "200% 0" },
        },
      },
      backdropBlur: {
        xs: "2px",
      },
    },
  },
  plugins: [],
};

export default config;
