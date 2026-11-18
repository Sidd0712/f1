import { useEffect, useState } from "react";
import { Shell, useHash, useRoute } from "./components/Shell";
import { loadHowWeKnow, loadStrategy, loadIndex, loadPodium, type SiteIndex } from "./data";
import { HowWeKnow, type HowData } from "./views/HowWeKnow";
import { Footer } from "./components/Footer";
import { PodiumModel } from "./model/podium";
import { Weekend } from "./views/Weekend";
import { SeasonBrowser } from "./views/SeasonBrowser";
import { Replay } from "./views/Replay";
import { Race } from "./views/Race";
import { Lab, type Profile } from "./views/Lab";
import { nextRace } from "./data";
import "./app.css";

export function App({ now = new Date() }: { now?: Date }) {
  const [index, setIndex] = useState<SiteIndex | null>(null);
  const [model, setModel] = useState<PodiumModel | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [how, setHow] = useState<HowData | null>(null);
  const [strat, setStrat] = useState<Record<string, Profile> | null>(null);
  const route = useRoute();
  const hash = useHash();
  const slug = route === "replay" ? hash.split("/")[2] ?? "" : "";
  useEffect(() => {
    loadIndex().then(setIndex, (e: Error) => setError(e.message));
    loadPodium().then((b) => setModel(new PodiumModel(b)), () => setModel(null));
    loadHowWeKnow().then(setHow, () => setHow(null));
    loadStrategy().then(setStrat, () => setStrat(null));
  }, []);
  return (
    <Shell asOf={index?.asOf}>
      {error && <p className="sub" role="alert">The data didn't load: {error}. Try reloading.</p>}
      {!index && !error && <p className="sub">Loading…</p>}
      {index && route === "weekend" && <Weekend index={index} model={model} now={now} />}
      {index && route === "replay" && (slug ? <Replay key={slug} slug={slug} index={index} /> : <SeasonBrowser index={index} />)}
      {index && route === "about" && (how ? <HowWeKnow data={how} /> : <p className="sub">Loading…</p>)}
      {index && route === "race" && <Race index={index} />}
      {index && route === "lab" && (strat ? <Lab data={strat} initial={nextRace(index.calendar, now) ?? undefined} locations={Object.fromEntries(index.races.map((r) => [r.circuit, r.location]))} /> : <p className="sub">Loading…</p>)}
      <Footer />
    </Shell>
  );
}
