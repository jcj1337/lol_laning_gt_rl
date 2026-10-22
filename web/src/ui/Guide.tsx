// The sequence's note panel (M6c), above the playback row in Watch: the introduction or the selected event's
// note, Previous and Next, and at an event the next event's title, so each event leads to the next. While the replay
// walks to the selected event, a status line stands in for its note; once the viewed position has left the event, a
// line says which position the note describes and the way back. The note never changes with the viewed position; the
// store says which checkpoint is selected, whether the replay is walking to it and whether the view is at it.
import { useStore } from './useStore';
import { afterTurn, DOCS_URL } from '../game/words';

export function GuidePanel() {
  const S = useStore();
  const gv = S.guide!, g = gv.guide, cp = S.guideCheckpoint(), n = g.checkpoints.length;
  const kicker = cp ? `Event ${gv.at! + 1} of ${n}` : 'Sequence';
  const title = cp ? cp.title : g.title;
  const close = <button className="linkbtn" id="gdClose" onClick={() => S.closeGuide()}>Close sequence</button>;
  if (!gv.notes) {
    return (
      <div className="guide collapsed" id="guide" aria-label="Sequence">
        <span className="gd-kicker">{kicker}</span><b>{title}</b>
        <span className="gd-tools"><button className="linkbtn" id="gdShow" onClick={() => S.setGuideNotes(true)}>Show notes</button>{close}</span>
      </div>
    );
  }
  const walking = S.guideWalking(), away = !walking && !S.guideAtEvent();
  const next = cp ? g.checkpoints[gv.at! + 1] : undefined;
  return (
    <div className="guide" id="guide" aria-label="Sequence">
      <div className="gd-head">
        <span className="gd-kicker">{kicker}</span><b>{title}</b>{cp && <span className="muted">{afterTurn(cp.position)}</span>}
        <span className="gd-tools"><button className="linkbtn" id="gdHide" onClick={() => S.setGuideNotes(false)}>Hide notes</button>{close}</span>
      </div>
      {walking
        ? <p className="gd-text muted" id="gdWalk" role="status">Playing to turn {S.guidePosition()}: now at turn {S.ctrl!.view}.</p>
        : <p className="gd-text">{cp ? cp.text : g.intro}</p>}
      <div className="gd-nav">
        <button className="pillbtn small" id="gdPrev" disabled={gv.at === null} onClick={() => S.guidePrev()}>Previous event</button>
        <button className="pillbtn small gold" id="gdNext" disabled={gv.at === n - 1} onClick={() => S.guideNext()}>Next event</button>
        {walking ? null
          : away ? <span className="gd-away" id="gdAway" role="status">This note describes the position {afterTurn(S.guidePosition())}. <button className="linkbtn" id="gdReturn" onClick={() => S.guideReturn()}>Return to this event</button></span>
          : cp ? next && <span className="muted" id="gdNextTitle">Next: {next.title}</span>
          : <>
            {g.end !== null && g.end !== undefined && <span className="muted">Covers turns {g.start} to {g.end} of {S.ctrl!.progress}</span>}
            <a className="linkbtn" href={DOCS_URL} target="_blank" rel="noreferrer">Research documentation</a>
          </>}
      </div>
    </div>
  );
}
