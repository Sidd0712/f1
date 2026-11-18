export function Car({ color, accent, className }: { color: string; accent: string; className?: string }) {
  return (
    <svg className={className} viewBox="0 0 62 150" aria-hidden="true">
      <rect x="2" y="2" width="58" height="9" rx="2" fill={accent} />
      <rect x="6" y="11" width="50" height="3" fill={color} />
      <path d="M27 12 L35 12 L36 48 L26 48 Z" fill={color} />
      <rect x="3" y="30" width="11" height="22" rx="4" fill="#111" />
      <rect x="48" y="30" width="11" height="22" rx="4" fill="#111" />
      <path d="M22 48 L40 48 L46 66 L48 100 L42 122 L20 122 L14 100 L16 66 Z" fill={color} />
      <path d="M14 72 L22 64 L22 104 L15 100 Z M48 72 L40 64 L40 104 L47 100 Z" fill={accent} />
      <ellipse cx="31" cy="66" rx="6" ry="8" fill="#0d0d0d" />
      <rect x="1" y="106" width="13" height="26" rx="4" fill="#111" />
      <rect x="48" y="106" width="13" height="26" rx="4" fill="#111" />
      <rect x="10" y="134" width="42" height="11" rx="2" fill={accent} />
    </svg>
  );
}
