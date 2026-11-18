import { useState } from "react";

const b64 = (buf: ArrayBuffer | null) => (buf ? btoa(String.fromCharCode(...new Uint8Array(buf))).replace(/\+/g, "-").replace(/\//g, "_").replace(/=+$/, "") : "");
const keyBytes = (k: string) => Uint8Array.from(atob(k.replace(/-/g, "+").replace(/_/g, "/") + "=".repeat((4 - (k.length % 4)) % 4)), (c) => c.charCodeAt(0));

export function AlertsSetup() {
  const [laptop, setLaptop] = useState("");
  const [key, setKey] = useState("");
  const [msg, setMsg] = useState<string | null>(null);
  const supported = typeof navigator !== "undefined" && "serviceWorker" in navigator && "PushManager" in window;

  const go = async () => {
    try {
      const reg = await navigator.serviceWorker.register(`${import.meta.env.BASE_URL}sw.js`);
      if ((await Notification.requestPermission()) !== "granted") return setMsg("Notifications are blocked for this site.");
      const sub = await reg.pushManager.subscribe({ userVisibleOnly: true, applicationServerKey: keyBytes(key.trim()) });
      const j = sub.toJSON();
      const enc = b64(new TextEncoder().encode(JSON.stringify({ endpoint: j.endpoint, keys: j.keys })).buffer as ArrayBuffer);
      window.location.href = `http://${laptop.trim()}/f1/live/subscribe?s=${enc}`;
    } catch (e) {
      setMsg(`Couldn't turn on alerts: ${(e as Error).message}`);
    }
  };

  return (
    <div className="panel claim">
      <h2>Race alerts on this phone</h2>
      {!supported ? (
        <p>This browser can't receive web alerts. On iPhone, add this site to your Home Screen first (Share, then Add to Home Screen) and open it from there.</p>
      ) : (
        <>
          <p>One-time setup on the same Wi-Fi as the laptop. The laptop prints its address and alert key when live mode starts.</p>
          <label className="field">Laptop address <input value={laptop} onChange={(e) => setLaptop(e.target.value)} placeholder="192.168.1.20:8765" /></label>
          <label className="field">Alert key <input value={key} onChange={(e) => setKey(e.target.value)} placeholder="B…" /></label>
          <button className="ctl primary" disabled={!laptop || !key} onClick={go}>Get race alerts</button>
          {msg && <p role="alert">{msg}</p>}
        </>
      )}
    </div>
  );
}
