import * as Tabs from "@radix-ui/react-tabs";
import { useEffect, useState } from "react";
import { AboutPage } from "./components/AboutPage";
import { AskPage } from "./components/AskPage";
import { CheckCodePage } from "./components/CheckCodePage";
import { SidePanel } from "./components/SidePanel";
import { PadlockIcon, SketchButton } from "./components/Sketch";
import { useStore } from "./store/useStore";

function useThemeAttribute() {
  const theme = useStore((s) => s.theme);
  useEffect(() => {
    document.documentElement.dataset.theme = theme;
  }, [theme]);
  return theme;
}

export default function App() {
  const theme = useThemeAttribute();
  const setTheme = useStore((s) => s.setTheme);
  const newSession = useStore((s) => s.newSession);
  const announcement = useStore((s) => s.announcement);
  const [tab, setTab] = useState("ask");
  const [turning, setTurning] = useState(false);

  const turnPage = () => {
    newSession();
    setTab("ask");
    setTurning(true);
    setTimeout(() => setTurning(false), 560);
  };

  return (
    <div className="app">
      <header className="topbar">
        <div className="brand">
          <PadlockIcon />
          <h1>OSlm</h1>
          <span className="tagline">process synchronization notebook</span>
        </div>
        <SketchButton
          seed="theme"
          onClick={() => setTheme(theme === "dark" ? "light" : "dark")}
          aria-label={theme === "dark" ? "Switch to paper (light) theme" : "Switch to chalkboard (dark) theme"}
        >
          <span aria-hidden="true">{theme === "dark" ? "☼" : "☾"}</span>
          <span className="label">{theme === "dark" ? "paper" : "chalkboard"}</span>
        </SketchButton>
        <SketchButton seed="new-page" onClick={turnPage} aria-label="New page (start a new session)">
          <span aria-hidden="true">✎</span>
          <span className="label">New page</span>
        </SketchButton>
      </header>

      <Tabs.Root value={tab} onValueChange={setTab} className="workspace" asChild>
        <div>
          <main className="page">
            <Tabs.List className="dividers" aria-label="Notebook sections">
              <Tabs.Trigger value="ask" className="divider-tab">
                Ask
              </Tabs.Trigger>
              <Tabs.Trigger value="check" className="divider-tab">
                Check my code
              </Tabs.Trigger>
              <Tabs.Trigger value="about" className="divider-tab">
                About
              </Tabs.Trigger>
            </Tabs.List>
            <Tabs.Content value="ask" asChild forceMount>
              <div style={{ display: tab === "ask" ? "contents" : "none" }}>
                <AskPage turning={turning} />
              </div>
            </Tabs.Content>
            <Tabs.Content value="check" asChild>
              <div style={{ display: "contents" }}>
                <CheckCodePage />
              </div>
            </Tabs.Content>
            <Tabs.Content value="about" asChild>
              <div style={{ display: "contents" }}>
                <AboutPage />
              </div>
            </Tabs.Content>
          </main>
          <SidePanel />
        </div>
      </Tabs.Root>

      <div className="sr-only" role="status" aria-live="polite">
        {announcement}
      </div>
    </div>
  );
}
