// The floating pill at the top: the mode switch, the game context (New game, or the recorded game or sequence to
// watch), what kind of record is on screen, About and Details.
import { useStore } from './useStore';
import { EXAMPLE_NOTE, GAME_NAMES, recordType } from '../game/words';

export function TopBar() {
  const S = useStore();
  const g = S.game;
  const kind = g?.meta.provenance.kind;
  const engine = !!S.engine?.ready;
  const simName = g && g.meta.provenance.kind === 'new_simulation' && S.mode === 'watch' ? g.label : null;
  const picked = S.guide ? 'guide:' + S.guide.guide.id : simName ? '__sim' : g?.name ?? '';
  return (
    <header className="glass topbar" id="topbar">
      <div className="seg" role="group" aria-label="Mode">
        <button id="modePlay" aria-pressed={S.mode === 'play'} onClick={() => void S.returnToPlay()}>Play</button>
        <button id="modeWatch" aria-pressed={S.mode === 'watch'} onClick={() => { if (S.mode !== 'watch') { if (S.replays.length) void S.watch(S.replays[0].name); else S.openDialog('newSim'); } }}>Watch</button>
      </div>
      <div className="context" id="context">
        {S.mode === 'play' ? (
          <button className="btn" id="btnNew" disabled={!engine} title={engine ? undefined : 'Play needs the engine'} onClick={() => S.openDialog('newGame')}>New game</button>
        ) : (
          <>
            <label className="sel"><span className="sr-only">Game</span>
              <select id="gamePick" value={picked} onChange={(e) => { const v = e.target.value; if (v.startsWith('guide:')) void S.openGuide(v.slice(6)); else if (v !== '__sim') void S.watch(v); }}>
                <optgroup label="Recorded research games">
                  {S.replays.map((r) => <option key={r.name} value={r.name}>{GAME_NAMES[r.name] || r.label}</option>)}
                </optgroup>
                {S.guides.length > 0 && <optgroup label="Sequences">
                  {S.guides.map((x) => <option key={x.id} value={'guide:' + x.id}>{x.title}</option>)}
                </optgroup>}
                {simName && <optgroup label="Generated for this demo"><option value="__sim">{simName}</option></optgroup>}
              </select>
            </label>
            <button className="btn" id="btnSim" disabled={!engine} title={engine ? undefined : 'Needs the engine'} onClick={() => S.openDialog('newSim')}>
              <span className="long">New game between agents</span><span className="short">New game</span>
            </button>
          </>
        )}
      </div>
      <span className="rectype" id="recType" title={EXAMPLE_NOTE}>{g && g.name !== 'idle' ? recordType(kind) : ''}</span>
      <span className="tb-end">
        <button className="linkbtn" id="aboutBtn" onClick={() => S.openDialog('about')}>About</button>
        <button className="pillbtn" id="detailsBtn" aria-expanded={S.detailsOpen} aria-controls="details" onClick={() => (S.detailsOpen ? S.closeDetails() : S.openDetails())}>Details</button>
      </span>
    </header>
  );
}
