const pct = (v: number) => `${Math.round(v * 100)}%`;

export function ValuePair({ model, base, label }: { model: number; base: number | null | undefined; label?: string }) {
  return (
    <span className="vp">
      <b>{pct(model)}</b>
      {base != null && (
        <small>
          {pct(base)}
          {label && <em> {label}</em>}
        </small>
      )}
    </span>
  );
}
