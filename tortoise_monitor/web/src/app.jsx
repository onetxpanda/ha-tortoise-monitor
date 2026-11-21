// Web UI for the add-on. Bundled with React by esbuild at image build time (npm run build ->
// dist/app.js) and served by app/web.py with index.html. All URLs are relative, so it works
// under ingress at any path.
import React, { useState, useEffect, useCallback, useRef, useMemo, useContext, createContext } from 'react';
import { createRoot } from 'react-dom/client';

const enc = encodeURIComponent;
const clamp = (v, lo, hi) => Math.max(lo, Math.min(hi, v));

// ── Page visibility ───────────────────────────────────────────────────────────
// Polling pauses while the page is hidden. On return every poll runs at once and the page is
// greyed as "last known state" until each has answered. Starts as visible whatever
// document.hidden says: the HA Companion app's WebView can report hidden at load and only
// correct it later, which would leave the page blank until tapped.
const pageVis = { hidden: false, resuming: new Set(), listeners: new Set() };
const notifyPage = () => pageVis.listeners.forEach(fn => fn());
function setPageHidden(hidden) {
  if (pageVis.hidden === hidden) return;
  pageVis.hidden = hidden;
  notifyPage();
}
document.addEventListener('visibilitychange', () => setPageHidden(document.hidden));
for (const ev of ['focus', 'pageshow', 'pointerdown', 'keydown']) window.addEventListener(ev, () => setPageHidden(false), true);

// ── Connection to the add-on ──────────────────────────────────────────────────
// Every call goes through api(), so the page knows when it has lost contact instead of showing
// its last state as if it were current.
const conn = { failures: 0, lastOkAt: Date.now(), listeners: new Set() };
function noteConn(ok) {
  if (!ok && pageVis.hidden) return;  // a hidden page's requests can be throttled: not a lost add-on
  const wasLost = conn.failures >= 2;
  if (ok) { conn.failures = 0; conn.lastOkAt = Date.now(); } else conn.failures++;
  if (wasLost !== (conn.failures >= 2)) conn.listeners.forEach(fn => fn());
}
async function api(path, body) {
  let r;
  try {
    r = await fetch(path, body === undefined ? {} : { method: 'POST', headers: { 'Content-Type': 'application/json' }, body: JSON.stringify(body) });
  } catch (e) { noteConn(false); throw new Error('The add-on is not answering'); }
  noteConn(r.status !== 502 && r.status !== 504);  // 502/504 come from HA's ingress proxy when the add-on isn't answering
  const j = await r.json().catch(() => ({ error: r.statusText }));
  if (!r.ok) throw new Error(j.error || r.statusText);
  return j;
}

function useSubscribe(set) {
  const [, rerender] = useState(0);
  useEffect(() => {
    const fn = () => rerender(n => n + 1);
    set.add(fn);
    return () => set.delete(fn);
  }, [set]);
}
function useConnectionLost() { useSubscribe(conn.listeners); return conn.failures >= 2; }
function usePageState() { useSubscribe(pageVis.listeners); return { visible: !pageVis.hidden, resuming: pageVis.resuming.size > 0 }; }

// fn returns a promise resolving to true when it got fresh data. Runs now and every ms while the page is visible.
function usePoll(name, fn, ms) {
  const { visible } = usePageState();
  const started = useRef(false);
  useEffect(() => {
    if (!visible) return;
    const resume = started.current;
    started.current = true;
    if (resume) { pageVis.resuming.add(name); notifyPage(); }
    const run = () => Promise.resolve(fn()).then(ok => { if (ok && pageVis.resuming.delete(name)) notifyPage(); });
    run();
    const id = setInterval(run, ms);
    return () => { clearInterval(id); if (pageVis.resuming.delete(name)) notifyPage(); };
  }, [name, fn, ms, visible]);
}

// The width of an element, kept up to date. Returns [ref callback, width].
function useWidth() {
  const [el, setEl] = useState(null);
  const [w, setW] = useState(0);
  useEffect(() => {
    if (!el) return;
    const ro = new ResizeObserver(() => setW(el.clientWidth));
    ro.observe(el);
    return () => ro.disconnect();
  }, [el]);
  return [setEl, w];
}

// ── Toast ─────────────────────────────────────────────────────────────────────
const toastBus = { show: null };
const toast = msg => toastBus.show && toastBus.show(msg);
function Toast() {
  const [msg, setMsg] = useState(null);
  const timer = useRef(null);
  useEffect(() => {
    toastBus.show = m => { setMsg(m); clearTimeout(timer.current); timer.current = setTimeout(() => setMsg(null), 3500); };
    return () => { toastBus.show = null; };
  }, []);
  return msg ? <div className="toast" role="status">{msg}</div> : null;
}

// ── Formatting ────────────────────────────────────────────────────────────────
function dur(s) {
  if (s == null) return '';
  s = Math.round(s);
  if (s < 60) return s + ' s';
  const m = Math.round(s / 60);
  if (m < 60) return m + ' min';
  return Math.floor(m / 60) + ' h ' + String(m % 60).padStart(2, '0') + ' min';
}
const at = ts => ts ? new Date(ts * 1000).toLocaleString([], { weekday: 'short', day: 'numeric', month: 'short', hour: '2-digit', minute: '2-digit' }) : '—';
const ago = ts => ts ? dur(Date.now() / 1000 - ts) + ' ago' : '';
const fmtDay = d => new Date(d + 'T12:00').toLocaleDateString([], { weekday: 'short', day: 'numeric', month: 'short' });
const pct = (a, b) => b ? Math.round(100 * a / b) + '%' : '—';
const R = b => ({ x: b[0], y: b[1], width: b[2] - b[0], height: b[3] - b[1] });

// ── App state shared by the pages ─────────────────────────────────────────────
const Ctx = createContext(null);
const useApp = () => useContext(Ctx);
const camsOf = (ov, role) => ov ? Object.entries(ov.cams).filter(([, v]) => !role || v.role === role).map(([c]) => c) : [];
const camName = (ov, c) => (ov && ov.cams[c] && ov.cams[c].name) || c;
// Review counts: the review page's own (fresh after each answer) until the next overview arrives
const countsOf = (ov, revCounts) => revCounts || Object.fromEntries(Object.entries(ov.cams).map(([c, v]) => [c, v.counts]));

// ── Routes ────────────────────────────────────────────────────────────────────
const PAGES = { review: 'Review', outline: 'Pen outline', training: 'Training', results: 'Results', heatmap: 'Heatmap', live: 'Live', servers: 'Training servers', jobs: 'Background jobs', storage: 'Storage' };
const JOBS = { collect: 'Collection', label: 'First pass', pass: 'Detector pass' };
function parseHash() {
  const [page, arg] = (location.hash || '#review').slice(1).split('/');
  return PAGES[page] ? { page, arg: decodeURIComponent(arg || ''), hash: location.hash || '#review' } : { page: 'review', arg: '', hash: '#review' };
}
function useRoute() {
  const [r, setR] = useState(parseHash);
  useEffect(() => {
    const f = () => setR(parseHash());
    window.addEventListener('hashchange', f);
    return () => window.removeEventListener('hashchange', f);
  }, []);
  return r;
}

// ── Small pieces ──────────────────────────────────────────────────────────────
const Badge = ({ children, cls }) => children === '' || children == null ? null : <span className={'badge ' + (cls || '')}>{children}</span>;
const Dot = ({ cls }) => <i className={'dot ' + cls} />;
const Sub = ({ children }) => <span className="sub">{children}</span>;
function Img({ src, style }) {
  const [bad, setBad] = useState(false);
  useEffect(() => setBad(false), [src]);
  return bad ? null : <img src={src} style={style} alt="" onError={() => setBad(true)} />;
}
const PLACE_SHORT = { house: 'in his house', pen: 'in the pen', outside: 'outside the walls', yard: 'in the yard' };
const PLACE = { house: 'in his house', pen: 'inside the pen walls', outside: 'outside the pen walls', yard: 'in the yard (no pen outline yet)' };

function jobState(name, s) {
  if (!s) return ['', ''];
  if (!s.on) return ['bad', 'off'];
  if (name === 'label') {
    const j = s.job;
    if (j && j.status === 'running') return ['run', 'running'];
    if (j && j.status === 'queued') return ['acc', 'waiting for a server'];
    return ['ok', 'on'];
  }
  if (s.running) return ['run', 'running'];
  if (name === 'collect' && s.paused) return ['bad', 'paused: disk full'];
  if (s.last_end && s.last_ok === false) return ['bad', 'last run failed'];
  if (s.error) return ['bad', 'error'];
  return ['ok', 'on'];
}

// ── Sidebar ───────────────────────────────────────────────────────────────────
function Nav({ route, onPick }) {
  const { ov: o, revCounts } = useApp();
  const [open, setOpen] = useState({});
  const k = countsOf(o, revCounts);
  const all = camsOf(o), tr = c => k[c] ? k[c].to_review || 0 : 0, total = all.reduce((a, c) => a + tr(c), 0);
  const items = [];
  items.push({ key: 'review', href: '#review', label: 'Review', badge: <Badge cls="acc">{total || ''}</Badge>, needs: total > 0,
    kids: [['#review', 'Both cameras', <Badge>{total || ''}</Badge>], ...all.map(c => [`#review/${c}`, camName(o, c), <Badge>{tr(c) || ''}</Badge>])] });
  items.push({ key: 'outline', href: '#outline', label: 'Pen outline', badge: camsOf(o, 'yard').some(c => !o.cams[c].outline) ? <Badge cls="bad">not drawn</Badge> : null, kids: [] });
  const open_ = [...o.runs.running, ...o.runs.proposed, ...o.runs.approved];
  const runLab = r => {
    if (r.status === 'proposed') return 'waiting for you';
    if (r.status === 'approved') return 'waiting for a server';
    const p = r.progress || {}, L = p.lease;
    if (L && L.state === 'waiting') return 'waiting for GPU';
    return p.epoch ? `epoch ${p.epoch}/${p.epochs} · ${r.device === 'gpu' ? 'GPU' : 'CPU'}` : (p.phase || 'starting');
  };
  items.push({ key: 'training', href: '#training', label: 'Training',
    badge: o.runs.proposed.length ? <Badge cls="acc">{o.runs.proposed.length + ' to approve'}</Badge>
      : o.runs.running.length ? <Badge cls="acc">{o.runs.running.length > 1 ? o.runs.running.length + ' running' : 'running'}</Badge> : null,
    needs: open_.length > 0, kids: [['#training', 'All runs', null], ...open_.map(r => [`#training/${r.id}`, `${camName(o, r.camera)} run ${r.id}`, <Sub>{runLab(r)}</Sub>])] });
  items.push({ key: 'results', href: '#results', label: 'Results', kids: all.map(c => [`#results/${c}`, camName(o, c), <Sub>{o.cams[c].model ? 'run ' + o.cams[c].model.run : 'no detector'}</Sub>]) });
  const yardFirst = [...camsOf(o, 'yard'), ...all.filter(c => !camsOf(o, 'yard').includes(c))];
  items.push({ key: 'heatmap', href: '#heatmap', label: 'Heatmap', kids: yardFirst.map(c => [`#heatmap/${c}`, camName(o, c), null]) });
  const ls = (o.detection || {}).last_seen;
  items.push({ key: 'live', href: '#live', label: 'Live', badge: ls ? <Sub>{PLACE_SHORT[ls.place] || ''}</Sub> : null, kids: [] });
  const srv = o.servers || [], up = srv.filter(s => s.online);
  items.push({ key: 'servers', href: '#servers', label: 'Training servers',
    badge: !srv.length ? <Badge cls="bad">none</Badge> : !up.length ? <Badge cls="bad">offline</Badge> : null,
    needs: srv.length > 0 && !up.length,
    kids: srv.map(s => ['#servers', s.name, <><Dot cls={s.online ? 'ok' : 'bad'} /><Sub>{s.online ? (s.detect !== 'off' ? 'detecting' : 'online') : 'offline'}</Sub></>]) });
  const bad = Object.keys(JOBS).some(n => jobState(n, o.jobs[n])[0] === 'bad');
  items.push({ key: 'jobs', href: '#jobs/collect', label: 'Background jobs', badge: bad ? <Badge cls="bad">needs attention</Badge> : null, needs: bad,
    kids: Object.entries(JOBS).map(([n, l]) => { const [c, t] = jobState(n, o.jobs[n]); return [`#jobs/${n}`, l, <><Dot cls={c} /><Sub>{t}</Sub></>]; }) });
  const stg = o.storage || {};
  items.push({ key: 'storage', href: '#storage', label: 'Storage',
    badge: stg.collection_paused ? <Badge cls="bad">full</Badge> : stg.used_percent != null ? <Sub>{stg.used_percent}%</Sub> : null, kids: [] });
  return (
    <nav onClick={e => { if (e.target.closest('a')) onPick(); }}>
      {items.map(it => {
        const active = route.page === it.key;
        const isOpen = it.kids.length > 0 && (open[it.key] ?? (active || !!it.needs));
        return (
          <div key={it.key} className={'nv' + (isOpen ? ' open' : '')}>
            <div className={'nv-i' + (active ? ' on' : '')}>
              {it.kids.length
                ? <button className="chev tg" aria-label="Show or hide" aria-expanded={isOpen} onClick={() => setOpen(s => ({ ...s, [it.key]: !isOpen }))}>▶</button>
                : <span className="chev" />}
              <a href={it.href}>{it.label}</a>{it.badge}
            </div>
            {isOpen && <div className="nv-kids">
              {it.kids.map(([h, l, bd], i) => <a key={i} className={'nv-k' + (route.hash === h ? ' on' : '')} href={h}>{l}{bd}</a>)}
            </div>}
          </div>
        );
      })}
      <p className="mut ver">Tortoise Monitor {o.version}</p>
    </nav>
  );
}

// ── Review ────────────────────────────────────────────────────────────────────
function zoomAt(size, p, L, prev) {
  const [W, H] = size;
  L = Math.round(Math.min(L || (prev ? prev.L : 480), W, H));
  return { x0: Math.round(clamp(p[0] - L / 2, 0, W - L)), y0: Math.round(clamp(p[1] - L / 2, 0, H - L)), L };
}
function zoomBox(size, b) {
  const L = Math.min(Math.max(Math.max(b[2] - b[0], b[3] - b[1]) * 3, 320), size[0], size[1]);
  return zoomAt(size, [(b[0] + b[2]) / 2, (b[1] + b[3]) / 2], L);
}
const norm = (a, b) => [Math.min(a[0], b[0]), Math.min(a[1], b[1]), Math.max(a[0], b[0]), Math.max(a[1], b[1])].map(Math.round);

function Marks({ it, sel, drawn, fs }) {
  const lab = (c, i) => <text x={c[1][0]} y={c[1][1] - 4} fontSize={fs}>{`${i + 1} · ${c[0].toFixed(2)} ${c[2]}`}</text>;
  return <>
    {it.hint && !it.cands.length && <><rect className="hint" {...R(it.hint)} /><text x={it.hint[0]} y={it.hint[1] - 4} fontSize={fs}>nearby frame</text></>}
    {it.cands.map((c, i) => i === sel ? null : <g key={i}><rect className="alt" {...R(c[1])} />{lab(c, i)}</g>)}
    {sel >= 0 && it.cands[sel] && <g><rect className="sel" {...R(it.cands[sel][1])} />{lab(it.cands[sel], sel)}</g>}
    {drawn && <rect className="drawn" {...R(drawn)} />}
  </>;
}

function Review({ cam }) {
  const { ov, revCounts, setRevCounts } = useApp();
  const [st, setSt] = useState(null);
  const [err, setErr] = useState('');
  const [sel, setSel] = useState(-1);
  const [drawn, setDrawn] = useState(null);
  const [zoom, setZoom] = useState(null);
  const [busy, setBusy] = useState(false);
  const [msg, setMsg] = useState('');
  const [frameRef, fw] = useWidth();
  const [zoomRef, zw] = useWidth();
  const imgRef = useRef(null), zimgRef = useRef(null), gest = useRef(null), busyRef = useRef(false);
  const it = st && st.item;

  const show = useCallback(d => {
    setSt(d); setErr(''); setRevCounts(d.counts); setMsg('');
    const x = d.item;
    if (!x) { setZoom(null); return; }
    const dr = x.review && x.review.box ? x.review.box : null;
    const s = dr ? -1 : (x.cands.length ? 0 : -1);
    setDrawn(dr); setSel(s);
    const target = dr || (s >= 0 ? x.cands[s][1] : null) || x.hint;
    setZoom(target ? zoomBox(x.size, target) : null);
    if (d.after) (new Image()).src = `img?f=${enc(d.after)}&w=1600`;  // the next frame, so it shows at once
  }, [setRevCounts]);
  useEffect(() => { api('api/review?cam=' + enc(cam)).then(show).catch(e => setErr(e.message)); }, [cam, show]);

  async function send(path, body) {
    if (busyRef.current) return;
    busyRef.current = true; setBusy(true);
    try { show(await api(path, body)); } catch (e) { toast(e.message); } finally { busyRef.current = false; setBusy(false); }
  }
  function answer(verdict) {
    if (!it) return;
    let box = null;
    if (verdict === 'tortoise') {
      box = drawn || (sel >= 0 ? it.cands[sel][1] : null);
      if (!box) { setMsg('Drag a box around him first.'); return; }
    }
    send('api/review', { file: it.file, verdict, box, cam });
  }
  const undo = () => send('api/undo', { cam });
  function pick(i) {
    if (!it || !it.cands[i]) return;
    setSel(i); setDrawn(null); setZoom(zoomBox(it.size, it.cands[i][1]));
  }
  const keys = useRef(null);
  keys.current = { answer, undo, pick };
  useEffect(() => {
    const h = e => {
      if (e.ctrlKey || e.metaKey || e.altKey || /INPUT|SELECT|TEXTAREA/.test(e.target.tagName)) return;
      const k = e.key.toLowerCase(), f = keys.current;
      if (k === 'y') f.answer('tortoise'); else if (k === 'n') f.answer('none'); else if (k === 's') f.answer('skip'); else if (k === 'z') f.undo();
      else if (/^[1-4]$/.test(k)) f.pick(+k - 1);
    };
    document.addEventListener('keydown', h);
    return () => document.removeEventListener('keydown', h);
  }, []);

  // Drag on either picture to draw a box; a tap picks the smallest box under it, or zooms there (main frame)
  function tapAt(p, isFrame) {
    let best = -1, area = Infinity;
    it.cands.forEach((c, i) => {
      const b = c[1], a = (b[2] - b[0]) * (b[3] - b[1]);
      if (p[0] >= b[0] && p[0] <= b[2] && p[1] >= b[1] && p[1] <= b[3] && a < area) { best = i; area = a; }
    });
    if (best >= 0) { setSel(best); setDrawn(null); if (isFrame) setZoom(zoomBox(it.size, it.cands[best][1])); }
    else if (isFrame) setZoom(zoomAt(it.size, p, null, zoom));
  }
  const mapFrame = e => {
    const r = imgRef.current.getBoundingClientRect(), [W, H] = it.size;
    return [clamp((e.clientX - r.left) / r.width * W, 0, W), clamp((e.clientY - r.top) / r.height * H, 0, H)];
  };
  const mapZoom = e => {
    const r = zimgRef.current.getBoundingClientRect();
    return [zoom.x0 + clamp((e.clientX - r.left) / r.width, 0, 1) * zoom.L, zoom.y0 + clamp((e.clientY - r.top) / r.height, 0, 1) * zoom.L];
  };
  const drawing = (map, isFrame) => ({
    onPointerDown: e => {
      e.preventDefault(); e.currentTarget.setPointerCapture(e.pointerId);
      gest.current = { x: e.clientX, y: e.clientY, p: map(e), moved: false, isFrame };
    },
    onPointerMove: e => {
      const g = gest.current;
      if (!g || g.isFrame !== isFrame) return;
      if (Math.hypot(e.clientX - g.x, e.clientY - g.y) > 6) g.moved = true;
      if (g.moved) { setDrawn(norm(g.p, map(e))); setSel(-1); }
    },
    onPointerUp: e => {
      const g = gest.current;
      if (!g || g.isFrame !== isFrame) return;
      gest.current = null;
      const p = map(e);
      if (!g.moved) { tapAt(p, isFrame); return; }
      const b = norm(g.p, p), ok = b[2] - b[0] >= 4 && b[3] - b[1] >= 4;
      setDrawn(ok ? b : null);
      if (ok && isFrame) setZoom(zoomBox(it.size, b));
    },
    onPointerCancel: () => { gest.current = null; },
  });

  const k = countsOf(ov, revCounts), cams = cam ? [cam] : camsOf(ov), sum = f => cams.reduce((a, c) => a + (k[c] ? f(k[c]) : 0), 0);
  return <>
    <h2>Review · {cam ? camName(ov, cam) : 'both cameras'}</h2>
    <p className="mut" style={{ margin: '0 0 4px' }}>
      {sum(x => x.to_review)} left · checked by you: {sum(x => x.pos_you + x.neg_you + x.skipped)} · labelled by the safety checks: {sum(x => x.pos_check)} with him, {sum(x => x.neg_check)} without
    </p>
    <p className="mut" style={{ fontSize: 13, margin: '0 0 10px' }}>Drag on either picture to draw his box · tap a box to pick it · tap the frame to zoom there.</p>
    {err ? <div className="card"><p className="err">Couldn't load the next frame: {err}</p></div>
      : !st ? <p className="mut">Loading…</p>
      : !it ? <div className="card"><p>Nothing left to review here. New frames come in every hour.</p><div className="row"><button disabled={busy} onClick={undo}>Undo the last answer</button></div></div>
      : <div>
        <p style={{ margin: '0 0 8px' }}><b>{it.camName}</b> · {it.when} · {it.why}{it.review && <> · <span className="pill acc">your earlier answer: {it.review.verdict}</span></>}</p>
        <div className="pics">
          <div className="pic" ref={frameRef} {...drawing(mapFrame, true)}>
            <img ref={imgRef} src={`img?f=${enc(it.file)}&w=1600`} alt="" draggable={false} />
            <svg viewBox={`0 0 ${it.size[0]} ${it.size[1]}`} preserveAspectRatio="none"><Marks it={it} sel={sel} drawn={drawn} fs={13 * it.size[0] / (fw || 800)} /></svg>
          </div>
          <div className="zoomwrap">
            {zoom
              ? <div className="pic" ref={zoomRef} {...drawing(mapZoom, false)}>
                <img ref={zimgRef} src={`crop?f=${enc(it.file)}&x=${zoom.x0}&y=${zoom.y0}&s=${zoom.L}&w=720`} alt="" draggable={false} />
                <svg viewBox={`${zoom.x0} ${zoom.y0} ${zoom.L} ${zoom.L}`} preserveAspectRatio="none"><Marks it={it} sel={sel} drawn={drawn} fs={13 * zoom.L / (zw || 400)} /></svg>
              </div>
              : <span className="mut" style={{ padding: 20 }}>Tap the picture to zoom</span>}
          </div>
        </div>
        <div className="actions">
          <button className="yes" disabled={busy || !(drawn || sel >= 0)} onClick={() => answer('tortoise')}>That's him <kbd>Y</kbd></button>
          <button className="no" disabled={busy} onClick={() => answer('none')}>Not in this frame <kbd>N</kbd></button>
          <button disabled={busy} onClick={() => answer('skip')}>Can't tell <kbd>S</kbd></button>
          <button disabled={busy} onClick={undo}>Undo <kbd>Z</kbd></button>
        </div>
        <p className="mut" style={{ margin: 0 }}>{msg}</p>
      </div>}
  </>;
}

// ── Pen outline ───────────────────────────────────────────────────────────────
function Outline() {
  const { ov } = useApp();
  const cam = camsOf(ov, 'yard')[0];
  const [ol, setOl] = useState(null);
  const [err, setErr] = useState('');
  const [pts, setPts] = useState([]);
  const [hist, setHist] = useState([]);
  const [sel, setSel] = useState(-1);
  const [dirty, setDirty] = useState(false);
  const [frame, setFrame] = useState('');
  const [picRef, pw] = useWidth();
  const imgRef = useRef(null), drag = useRef(null);

  const reset = st => { setOl(st); setPts(st.outline ? st.outline.poly.map(p => p.slice()) : []); setHist([]); setSel(-1); setDirty(false); };
  useEffect(() => {
    if (!cam) return;
    api('api/outline?cam=' + enc(cam)).then(st => {
      reset(st);
      const fr = st.frames || [];
      setFrame(st.outline && st.outline.frame && fr.some(f => f.file === st.outline.frame) ? st.outline.frame : fr[0] ? fr[0].file : '');
    }).catch(e => setErr(e.message));
  }, [cam]);

  const push = () => { setHist(h => [...h, pts.map(p => p.slice())].slice(-100)); setDirty(true); };
  const point = e => {
    const r = imgRef.current.getBoundingClientRect();
    return [clamp((e.clientX - r.left) / r.width, 0, 1), clamp((e.clientY - r.top) / r.height, 0, 1)];
  };
  function del() { if (sel < 0) return; push(); setPts(P => P.filter((_, j) => j !== sel)); setSel(-1); }
  function undo() { if (!hist.length) return; setPts(hist[hist.length - 1]); setHist(h => h.slice(0, -1)); setSel(-1); setDirty(true); }
  function clear() { if (!pts.length) return; push(); setPts([]); setSel(-1); }
  function suggest() { if (!ol.suggestion) return; push(); setPts(ol.suggestion.map(p => p.slice())); setSel(-1); }
  function revert() { setPts(ol.outline ? ol.outline.poly.map(p => p.slice()) : []); setHist([]); setSel(-1); setDirty(false); }
  async function save() {
    if (pts.length && pts.length < 3) { toast('An outline needs at least 3 points.'); return; }
    if (!pts.length && !confirm('Save with no outline (removes the saved one)?')) return;
    try { reset(await api('api/outline', { cam: ol.cam, poly: pts, frame })); toast('Outline saved'); } catch (e) { toast(e.message); }
  }
  const delKey = useRef(null);
  delKey.current = del;
  useEffect(() => {
    const h = e => {
      if (/INPUT|SELECT|TEXTAREA/.test(e.target.tagName)) return;
      if (e.key === 'Delete' || e.key === 'Backspace') { e.preventDefault(); delKey.current(); }
    };
    document.addEventListener('keydown', h);
    return () => document.removeEventListener('keydown', h);
  }, []);

  // Drag a point to move it; drag a small dot between two points to add one; tap empty ground to add one at the end
  const handlers = {
    onPointerDown: e => {
      if (!imgRef.current) return;
      e.preventDefault(); e.currentTarget.setPointerCapture(e.pointerId);
      const d = e.target.dataset || {}, x = e.clientX, y = e.clientY;
      if (d.i !== undefined) drag.current = { kind: 'v', i: +d.i, x, y, moved: false };
      else if (d.m !== undefined) {
        push();
        const i = +d.m + 1, p = point(e);
        setPts(P => { const n = P.slice(); n.splice(i, 0, p); return n; });
        setSel(i); drag.current = { kind: 'v', i, x, y, moved: true, pushed: true };
      } else drag.current = { kind: 'bg', x, y, moved: false };
    },
    onPointerMove: e => {
      const d = drag.current;
      if (!d) return;
      if (Math.hypot(e.clientX - d.x, e.clientY - d.y) > 5) {
        if (d.kind === 'v' && !d.pushed) { push(); d.pushed = true; }
        d.moved = true;
      }
      if (d.kind === 'v' && d.moved) { const p = point(e); setPts(P => P.map((q, j) => j === d.i ? p : q)); setSel(d.i); }
    },
    onPointerUp: e => {
      const d = drag.current;
      if (!d) return;
      drag.current = null;
      if (d.kind === 'bg' && !d.moved) { push(); const p = point(e); setPts(P => [...P, p]); setSel(pts.length); }
      else if (d.kind === 'v' && !d.moved) setSel(d.i);
    },
    onPointerCancel: () => { drag.current = null; },
  };

  const [W, H] = (ol && ol.size) || [1, 1], k = W / (pw || 800);
  const feet = useMemo(() => ((ol && ol.feet_pts) || []).map((p, i) => <circle key={i} className="ft" cx={p[0] * W} cy={p[1] * H} r={2.5 * k} />), [ol, W, H, k]);
  if (!cam) return <><h2>Pen outline</h2><p className="mut">No camera has the yard role.</p></>;
  if (err) return <><h2>Pen outline</h2><div className="card"><p className="err">Couldn't load: {err}</p></div></>;
  if (!ol) return <><h2>Pen outline</h2><p className="mut">Loading…</p></>;
  const P = pts.map(p => [p[0] * W, p[1] * H]);
  const idx = ol.frames.findIndex(f => f.file === frame);
  return <>
    <h2>Pen outline</h2>
    <div className="row" style={{ marginBottom: 8 }}>
      <button aria-label="Older frame" disabled={idx >= ol.frames.length - 1} onClick={() => setFrame(ol.frames[idx + 1].file)}>‹</button>
      <select value={frame} onChange={e => setFrame(e.target.value)}>{ol.frames.map(f => <option key={f.file} value={f.file}>{f.when}</option>)}</select>
      <button aria-label="Newer frame" disabled={idx <= 0} onClick={() => setFrame(ol.frames[idx - 1].file)}>›</button>
      <span className="mut">{dirty ? <span className="pill acc">Unsaved changes</span> : ol.outline ? `Saved ${at(ol.outline.saved)}` : <span className="err">No outline saved</span>}</span>
    </div>
    <p className="mut" style={{ margin: '0 0 8px' }}>
      He counts as inside when the bottom-centre of his box (his feet) is inside this outline. <b>Far and side walls:</b> click along where the wall meets the ground on the inside. <b>Near wall</b> (bottom of the picture, seen from outside): click along where it meets the ground on <i>your</i> side, so the whole board is inside the outline — when he is against it the board hides his feet and his box ends on its top edge. Drag a point to move it; drag a small dot between points to add one. Yellow dots: where his feet have been seen.
    </p>
    <div className="pic" ref={picRef}>
      {frame && <img ref={imgRef} src={`img?f=${enc(frame)}&w=1600`} alt="" draggable={false} />}
      <svg viewBox={`0 0 ${W} ${H}`} preserveAspectRatio="none" {...handlers}>
        {feet}
        {P.length >= 2 && <polygon className="ol" points={P.map(p => p.join(',')).join(' ')} />}
        {P.length >= 2 && P.map((p, i) => { const q = P[(i + 1) % P.length]; return <circle key={'m' + i} className="mid" data-m={i} cx={(p[0] + q[0]) / 2} cy={(p[1] + q[1]) / 2} r={6 * k} />; })}
        {P.map((p, i) => <circle key={'v' + i} className={'vx' + (i === sel ? ' selv' : '')} data-i={i} cx={p[0]} cy={p[1]} r={9 * k} />)}
      </svg>
    </div>
    <div className="row" style={{ marginTop: 10 }}>
      <button className="pri" onClick={save}>Save outline</button>
      <button disabled={!hist.length} onClick={undo}>Undo</button>
      <button disabled={sel < 0} onClick={del}>Delete point</button>
      {ol.suggestion && <button onClick={suggest}>Start from where he's been seen</button>}
      <button disabled={!pts.length} onClick={clear}>Clear</button>
      <button disabled={!dirty} onClick={revert}>Back to saved</button>
    </div>
  </>;
}

// ── Training ──────────────────────────────────────────────────────────────────
const STATUS = { proposed: ['Waiting for your approval', 'acc'], approved: ['Approved, waiting to start', 'acc'], running: ['Training', 'acc'],
  done: ['Finished', 'ok'], failed: ['Failed', 'bad'], stopped: ['Stopped', ''], declined: ['Declined', ''] };

function RunCard({ r, st, focus, logOpen, log, act, toggleLog }) {
  const [label, cls] = STATUS[r.status] || [r.status, ''];
  const k = r.counts || {}, cfg = r.config, res = r.result || {}, p = r.progress || {};
  const est = d => r.estimate && r.estimate[d] ? <>about {dur(r.estimate[d].secs)} <span className="mut">({r.estimate[d].measured ? 'from the last run' : 'rough guess'})</span></> : null;
  const logBtn = <button onClick={() => toggleLog(r.id)}>{logOpen ? 'Hide' : 'Show'} log</button>;
  const title = `${r.camName} · run ${r.id}`;
  let body = null;
  if (r.status === 'proposed') {
    const on = (st.servers || []).filter(s => s.online), gpuOn = on.filter(s => s.gpus > 0);
    body = <>
      <p>On a training server's CPU: {est('cpu')}</p><p>On a GPU: {est('gpu')} once it has a card</p>
      <p className="mut">{!on.length ? 'No training server is online: an approved run waits for one (add one under Training servers).'
        : `Online: ${on.map(s => s.name + (s.gpus ? ` (${s.gpus} GPU${s.gpus > 1 ? 's' : ''})` : ' (CPU only)')).join(', ')}. A GPU run waits for the card's holder to hand it over (Basalt does after its current request), keeps it until it's done — minutes — then gives it back. Force on the Arbiter page stops it; it carries on from its last epoch.`}</p>
      <div className="row">
        <button className="yes" onClick={() => act(r.id, 'approve', { device: 'cpu' })}>Approve — train on CPU</button>
        <button className="yes" disabled={!gpuOn.length} onClick={() => act(r.id, 'approve', { device: 'gpu' })}>Approve — train on GPU</button>
        <button onClick={() => act(r.id, 'decline')}>Decline</button>
      </div>
    </>;
  } else if (r.status === 'approved') {
    body = <>
      <p className="mut">Waiting for a training server with a free {cfg.device === 'gpu' ? 'GPU' : 'CPU'}.</p>
      {r.estimate && <p>Estimated time: {est(cfg.device || 'cpu')}</p>}
      <div className="row"><button onClick={() => act(r.id, 'stop')}>Cancel</button></div>
    </>;
  } else if (r.status === 'running') {
    const es = p.epoch_secs || [], per = es.length ? es[es.length - 1] : null, L = p.lease;
    body = <>
      <p>{r.server_name ? `On ${r.server_name} · ` : ''}{p.epoch ? <>Epoch <b>{p.epoch}</b> of {p.epochs}</> : (p.phase || 'Starting')}{p.map50 != null ? ` · mAP50 ${(+p.map50).toFixed(2)}` : ''}</p>
      {L && <p>{L.state === 'waiting' ? <><span className="pill acc">Waiting for {L.card}</span> held by {L.holder || 'another program'}; Arbiter asks it to hand the card over once its current work is done (since {at(L.since)})</>
        : L.state === 'holding' ? <><span className="pill ok">On {L.card}</span> since {at(L.since)}{L.asked_by ? ` · ${L.asked_by} asked for it at ${at(L.asked_at)}; finishing first` : ''}</>
        : L.state === 'no arbiter' ? <><span className="pill bad">On {L.card} without a lease</span> the arbiter isn't running</> : null}</p>}
      {per && p.epoch ? <p className="mut">Last epoch took {dur(per)} · about {dur(per * (p.epochs - p.epoch))} left if it runs every epoch</p> : null}
      {p.at ? <p className="mut">Updated {at(p.at)}</p> : null}
      <div className="row"><button onClick={() => act(r.id, 'stop')}>Stop</button>{logBtn}</div>
    </>;
  } else if (r.status === 'done') {
    body = <>
      <p>Held-out frames with him: found <b>{res.found} of {res.with_him}</b> ({pct(res.found, res.with_him)})</p>
      <p>Held-out frames without him: false alarm in <b>{res.false_alarms} of {res.without_him}</b> ({pct(res.false_alarms, res.without_him)})</p>
      <p className="mut">At confidence {res.conf} · mAP50 {(+res.map50).toFixed(2)} · {res.epochs_run} epochs in {dur(res.secs)} on {res.device === 'gpu' ? 'a GPU' : 'the CPU'}</p>
      <div className="row">{r.in_use ? <button onClick={() => act(r.id, 'unuse')}>Stop using</button> : <button className="pri" onClick={() => act(r.id, 'use')}>Use this detector</button>}{logBtn}</div>
    </>;
  } else if (r.status === 'failed') {
    body = <><p className="err">{res.error || 'Failed'}</p><div className="row">{logBtn}</div></>;
  }
  return (
    <div className="card">
      <h3>{focus ? title : <a href={`#training/${r.id}`} style={{ color: 'inherit' }}>{title}</a>} <span className={'pill ' + cls}>{label}</span>{r.in_use && <> <span className="pill ok">In use</span></>}</h3>
      <p className="mut">Proposed {at(r.created)}{r.decided ? ' · decided ' + at(r.decided) : ''}{r.started ? ' · started ' + at(r.started) : ''}{r.finished ? ' · ended ' + at(r.finished) : ''}</p>
      {k.pos !== undefined && <>
        <p>Frames with him: <b>{k.pos}</b> ({k.pos_you} checked by you) · without him: <b>{k.neg}</b> ({k.neg_you} checked by you)</p>
        {(k.pos_all > k.pos || k.neg_all > k.neg) && <p className="mut">Made diverse from {k.pos_all} with him and {k.neg_all ?? k.neg} without: with him, one per hour for each part of the picture (none where he hasn't moved in 2 hours); without him, an empty scene once per 20 minutes and each look-alike once per hour. Your answers count first.</p>}
        {k.val != null && <p className="mut">{k.split === 'days'
          ? `Tested on ${k.val_days.length} whole day${k.val_days.length === 1 ? '' : 's'} it never trains on: ${k.val} frames (${k.val_pos} with him) from ${k.val_days.map(fmtDay).join(', ')}`
          : k.split === 'hours' ? `Held out for testing: ${k.val} (${k.val_pos} with him), whole hours — only one day with him so far, so it can't be tested on another day`
          : `Held out for testing: ${k.val} (${k.val_pos} with him)`}{k.left_out != null ? ` · uncertain, not used: ${k.left_out}` : ''}</p>}
        {k.train_days ? <p className="mut">Trains on frames with him from {k.train_days} day{k.train_days === 1 ? '' : 's'}{k.capped && Object.keys(k.capped).length ? ` · capped so no day outweighs the rest: ${Object.entries(k.capped).map(([d, ab]) => `${fmtDay(d)} ${ab[0]} → ${ab[1]}`).join(', ')}` : ''}</p> : null}
      </>}
      <p className="mut">YOLO11n starting from COCO weights · {cfg.imgsz} px · up to {cfg.epochs} epochs (stops after {cfg.patience} without gain){r.status !== 'proposed' ? ` · on ${cfg.device === 'gpu' ? 'a GPU' : 'a CPU'}${r.server_name ? ' of ' + r.server_name : ''}` : ''}</p>
      {body}
      {logOpen && <pre>{log || res.log || p.log || '(empty)'}</pre>}
    </div>
  );
}

function Training({ focus }) {
  const { ov, loadOv } = useApp();
  const [st, setSt] = useState(null);
  const [err, setErr] = useState('');
  const [open, setOpen] = useState(() => new Set(focus ? [focus] : []));
  const [logs, setLogs] = useState({});
  const openRef = useRef(open);
  openRef.current = open;
  const load = useCallback(async () => {
    try {
      const d = await api('api/runs');
      const got = {};
      for (const id of openRef.current) { try { got[id] = (await api(`api/runs/${id}/log`)).log; } catch (e) { /* keep the last copy */ } }
      setSt(d); setErr(''); setLogs(l => ({ ...l, ...got }));
      return true;
    } catch (e) { setErr(e.message); return false; }
  }, []);
  usePoll('runs', load, 10000);
  async function toggleLog(id) {
    if (open.has(id)) { setOpen(s => { const n = new Set(s); n.delete(id); return n; }); return; }
    setOpen(s => new Set(s).add(id));
    try { const t = (await api(`api/runs/${id}/log`)).log; setLogs(l => ({ ...l, [id]: t })); } catch (e) { toast(e.message); }
  }
  async function act(id, action, body) {
    if (action === 'stop' && !confirm('Stop run ' + id + '? Training so far is thrown away.')) return;
    try { setSt(await api(`api/runs/${id}/${action}`, body || {})); loadOv(); } catch (e) { toast(e.message); }
  }
  async function propose(cam) { try { setSt(await api('api/runs/propose', { cam })); loadOv(); } catch (e) { toast(e.message); } }

  const runs = st ? (focus ? st.runs.filter(r => r.id === focus) : st.runs) : [];
  return <>
    <h2>{focus ? (runs.length ? `Training · ${runs[0].camName} run ${focus}` : 'Training') : 'Training · all runs'}</h2>
    {!focus && <div className="row" style={{ marginBottom: 10 }}>
      {camsOf(ov).map(c => <button key={c} onClick={() => propose(c)}>Propose a {camName(ov, c).toLowerCase()} run</button>)}
      <span className="mut">A run is also proposed for a camera once its review queue is clear and there's 25% more data (or 100 more answers) than last time. Nothing trains until you approve it.</span>
    </div>}
    {err && !st ? <div className="card"><p className="err">Couldn't load the runs: {err}</p></div>
      : !st ? <p className="mut">Loading…</p>
      : !runs.length ? <div className="card"><p>{focus ? 'No such run.' : 'No runs here. A first run is proposed for a camera once its review queue is clear and it has 150 frames with him, or propose one yourself.'}</p></div>
      : runs.map(r => <RunCard key={r.id} r={r} st={st} focus={focus} logOpen={open.has(r.id)} log={logs[r.id]} act={act} toggleLog={toggleLog} />)}
  </>;
}

// ── Results ───────────────────────────────────────────────────────────────────
function Results({ cam }) {
  const { ov } = useApp();
  const c = cam || camsOf(ov, 'yard')[0] || camsOf(ov)[0];
  const [st, setSt] = useState(null);
  const [err, setErr] = useState('');
  const [t, setT] = useState(0);
  useEffect(() => {
    api('api/results?cam=' + enc(c || '')).then(d => { setSt(d); setT(Date.now()); }).catch(e => setErr(e.message));
  }, [c]);
  if (err) return <><h2>Results</h2><div className="card"><p className="err">Couldn't load: {err}</p></div></>;
  if (!st) return <><h2>Results</h2><p className="mut">Loading…</p></>;
  if (!st.model) return <><h2>Results · {st.camName}</h2>
    <div className="card"><p>No detector in use for the {st.camName.toLowerCase()} camera yet. Approve a training run; when it finishes, press “Use this detector”.</p></div></>;
  const yard = st.role === 'yard';
  return <>
    <h2>Results · {st.camName}</h2>
    <p className="mut" style={{ margin: '0 0 10px' }}>Detector: run {st.model.run}, in use since {at(st.model.since)}{yard && <> · pen outline {st.outline ? 'drawn' : <span className="err">not drawn — <a href="#outline">draw it</a></span>}</>}</p>
    <div className="card"><h3>Last 24 hours</h3>
      <div className="legend">
        <span><i style={{ background: 'var(--ok)' }} />seen {yard ? 'inside the pen walls' : 'in his house'}</span>
        {yard && <span><i style={{ background: 'var(--bad)' }} />seen outside them</span>}
        <span><i style={{ background: '#5b6270' }} />not seen</span><span><i style={{ background: 'var(--btn)' }} />not running</span>
      </div>
      <div className="strip">{st.hours.map((h, i) => {
        const inside = h.seen - h.outside, cls = h.outside ? 'out' : inside ? 'in' : h.n ? 'none' : '';
        return <div key={i} className={cls} title={`${h.label}:00 · ${h.n} checks · seen in ${inside}${yard ? ` · outside the walls in ${h.outside}` : ''}`} />;
      })}</div>
      <div className="striplab">{st.hours.map((h, i) => <span key={i}>{i % 3 === 0 ? h.label : ''}</span>)}</div>
    </div>
    {st.has_snapshot && <div className="card"><h3>Last seen on this camera</h3><div style={{ maxWidth: 800 }}><Img src={`live?name=${enc(st.cam)}.jpg&t=${t}`} style={{ width: '100%', borderRadius: 6 }} /></div></div>}
    {yard && <div className="card"><h3>Seen outside the pen walls ({st.outside.length})</h3>
      {st.outside.length
        ? <div className="figs">{st.outside.map(f => <figure key={f.name}><img loading="lazy" src={`live?name=outside/${enc(f.name)}`} style={{ width: '100%' }} alt="" /><figcaption>{f.when}</figcaption></figure>)}</div>
        : <p className="mut">{st.outline ? 'None kept.' : 'Draw the pen outline first.'}</p>}
    </div>}
  </>;
}

// ── Heatmap ───────────────────────────────────────────────────────────────────
// Where he was on one day: his foot points as heat (time spent) or as dots
// coloured by time of day, over a frame of that day.
const HEAT_STOPS = [[0, [40, 60, 255]], [0.35, [0, 220, 255]], [0.6, [70, 255, 60]], [0.8, [255, 225, 0]], [1, [255, 40, 0]]];
function heatColor(v) {
  for (let i = 1; i < HEAT_STOPS.length; i++) {
    if (v <= HEAT_STOPS[i][0]) {
      const [a, ca] = HEAT_STOPS[i - 1], [b, cb] = HEAT_STOPS[i], t = (v - a) / (b - a);
      return ca.map((x, j) => Math.round(x + (cb[j] - x) * t));
    }
  }
  return HEAT_STOPS[HEAT_STOPS.length - 1][1];
}
const HEAT_CSS = 'linear-gradient(90deg,' + HEAT_STOPS.map(([v, c]) => `rgb(${c.join(',')}) ${v * 100}%`).join(',') + ')';
// Each sighting adds a soft spot where his feet were; sightings come at a steady
// pace while he's in view, so the heat is time spent. Scaled to the busiest
// spots (99th percentile), so one long stay doesn't wash out the rest.
function drawHeat(cv, pts, size) {
  if (!cv || !size) return;
  const GW = 320, GH = Math.max(1, Math.round(GW * size[1] / size[0]));
  cv.width = GW; cv.height = GH;
  const ctx = cv.getContext && cv.getContext('2d');
  if (!ctx) return;
  ctx.clearRect(0, 0, GW, GH);
  if (!pts.length) return;
  const g = new Float32Array(GW * GH), s = GW * 0.012, R = Math.ceil(3 * s), ker = [];
  for (let dy = -R; dy <= R; dy++) for (let dx = -R; dx <= R; dx++) ker.push([dx, dy, Math.exp(-(dx * dx + dy * dy) / (2 * s * s))]);
  for (const p of pts) {
    const cx = Math.round(p[0] * (GW - 1)), cy = Math.round(p[1] * (GH - 1));
    for (const [dx, dy, w] of ker) {
      const x = cx + dx, y = cy + dy;
      if (x >= 0 && y >= 0 && x < GW && y < GH) g[y * GW + x] += w;
    }
  }
  const nz = Array.from(g).filter(v => v > 0.05).sort((a, b) => a - b);
  const top = nz.length ? nz[Math.min(nz.length - 1, Math.floor(nz.length * 0.99))] : 1;
  const img = ctx.createImageData(GW, GH);
  for (let i = 0; i < g.length; i++) {
    const v = Math.min(1, g[i] / top);
    if (v < 0.03) continue;
    const [r, gg, b] = heatColor(v);
    img.data[i * 4] = r; img.data[i * 4 + 1] = gg; img.data[i * 4 + 2] = b; img.data[i * 4 + 3] = Math.round(255 * (0.25 + 0.55 * Math.sqrt(v)));
  }
  ctx.putImageData(img, 0, 0);
}
const DOT_FROM_H = 6, DOT_TO_H = 20;     // dot colours: blue in the morning to red in the evening
const dotHue = sec => 240 * (1 - clamp((sec / 3600 - DOT_FROM_H) / (DOT_TO_H - DOT_FROM_H), 0, 1));
const hhmm = ts => new Date(ts * 1000).toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' });

function Heatmap({ cam }) {
  const { ov } = useApp();
  const c = cam || camsOf(ov, 'yard')[0] || camsOf(ov)[0];
  const [day, setDay] = useState('');
  const [st, setSt] = useState(null);
  const [err, setErr] = useState('');
  const [mode, setMode] = useState('heat');
  const [noPic, setNoPic] = useState('');
  const last = useRef('');
  const cvRef = useRef(null);
  const load = useCallback(() => api('api/heatmap?cam=' + enc(c || '') + (day ? '&day=' + day : '')).then(d => {
    const s = JSON.stringify(d);
    if (s !== last.current) { last.current = s; setSt(d); }    // unchanged data: nothing redrawn
    setErr('');
    return true;
  }).catch(e => { setErr(e.message); return false; }), [c, day]);
  usePoll('heatmap:' + c + ':' + day, load, 60000);
  useEffect(() => { if (mode === 'heat' && st) drawHeat(cvRef.current, st.points, st.size); }, [st, mode]);
  const [W, H] = (st && st.size) || [16, 9];
  const dots = useMemo(() => {
    if (!st || mode !== 'dots') return null;
    const step = Math.max(1, Math.ceil(st.points.length / 3000)), r = W / 220;
    return st.points.filter((_, i) => i % step === 0).map((p, i) =>
      <circle key={i} cx={p[0] * W} cy={p[1] * H} r={r} style={{ fill: `hsla(${dotHue(p[2])},95%,55%,.75)` }} />);
  }, [st, mode, W, H]);

  if (!c) return <><h2>Heatmap</h2><p className="mut">No cameras.</p></>;
  if (err && !st) return <><h2>Heatmap</h2><div className="card"><p className="err">Couldn't load: {err}</p></div></>;
  if (!st) return <><h2>Heatmap</h2><p className="mut">Loading…</p></>;
  const days = st.days.some(d => d.day === st.day) ? st.days : [{ day: st.day, n: 0 }, ...st.days];
  const idx = days.findIndex(d => d.day === st.day);
  const n = st.points.length;
  const yard = st.role === 'yard', O = st.outline;
  return <>
    <h2>Heatmap · {st.camName}</h2>
    <div className="row" style={{ marginBottom: 8 }}>
      <button aria-label="Day before" disabled={idx >= days.length - 1} onClick={() => setDay(days[idx + 1].day)}>‹</button>
      <select aria-label="Day" value={st.day} onChange={e => setDay(e.target.value)}>
        {days.map(d => <option key={d.day} value={d.day}>{fmtDay(d.day)}{d.n ? ` · ${d.n} sightings` : ' · none'}</option>)}
      </select>
      <button aria-label="Day after" disabled={idx <= 0} onClick={() => setDay(days[idx - 1].day)}>›</button>
      <span className="seg">
        <button className={mode === 'heat' ? 'on' : ''} onClick={() => setMode('heat')}>Heat</button>
        <button className={mode === 'dots' ? 'on' : ''} onClick={() => setMode('dots')}>Dots by time</button>
      </span>
    </div>
    <p className="mut" style={{ margin: '0 0 8px' }}>
      {n ? <>Seen {n} times, about {dur(st.seen_s)} in view, from {hhmm(st.first)} to {hhmm(st.last)}.</>
        : st.days.length ? 'Not seen on this camera that day.' : 'No sightings recorded on this camera yet.'}
    </p>
    <div className="pic heat" style={{ aspectRatio: `${W} / ${H}` }}>
      {st.frame && noPic !== st.frame.file && <img src={`img?f=${enc(st.frame.file)}&w=1600`} alt="" draggable={false}
        onError={() => setNoPic(st.frame.file)} />}
      {mode === 'heat' && <canvas ref={cvRef} />}
      <svg viewBox={`0 0 ${W} ${H}`} preserveAspectRatio="none" style={{ pointerEvents: 'none' }}>
        {yard && O && <polygon className="ol" points={O.map(p => `${p[0] * W},${p[1] * H}`).join(' ')} />}
        {dots}
      </svg>
    </div>
    <div className="legend" style={{ marginTop: 8 }}>
      {mode === 'heat'
        ? <span>less time <i style={{ width: 120, background: HEAT_CSS }} /> more time</span>
        : <span>{DOT_FROM_H}:00 <i style={{ width: 120, background: 'linear-gradient(90deg,hsl(240,95%,55%),hsl(120,95%,55%),hsl(0,95%,55%))' }} /> {DOT_TO_H}:00</span>}
      {yard && O && <span><i style={{ background: '#ff4fd8' }} />pen walls</span>}
    </div>
  </>;
}

// ── Live ──────────────────────────────────────────────────────────────────────
function Live() {
  const { ov } = useApp();
  const [st, setSt] = useState(null);
  const load = useCallback(() => api('api/live').then(d => { setSt(d); return true; }).catch(() => false), []);
  usePoll('live', load, 10000);
  if (!st) return <><h2>Live</h2><p className="mut">Loading…</p></>;
  const ls = st.last_seen;
  return <>
    <h2>Live</h2>
    <div className="card"><h3>Last seen</h3>
      {ls ? <>
        <p><b>{PLACE[ls.place] || ls.place}</b> · {at(ls.ts)} ({ago(ls.ts)}) · {camName(ov, ls.camera)} · {(+ls.score).toFixed(2)} · detected by {ls.source}</p>
        <div style={{ maxWidth: 960 }}><Img src={`live?name=last.jpg&t=${Math.round(ls.ts)}`} style={{ width: '100%', borderRadius: 6 }} /></div>
      </> : <p className="mut">Not seen since detection started.</p>}
    </div>
    <div className="cards" style={{ marginTop: 12 }}>
      {Object.entries(st.cameras).map(([cam, c]) => (
        <div className="card" key={cam}>
          <h3>{camName(ov, cam)}</h3>
          <p>{c.fresh ? <><span className="pill ok">Running</span> on {c.device || c.source}</> : c.model ? <span className="pill bad">Not running</span> : <span className="pill">No detector in use</span>}</p>
          {c.model ? <p className="mut">Detector: run {c.model.run}</p> : <p className="mut">Put a finished run in use under Training.</p>}
          {c.ts ? <p className="mut">Last result {ago(c.ts)}{c.ms != null ? ` · ${(+c.ms).toFixed(0)} ms` : ''} · {c.seen ? `seen, ${PLACE[c.place] || c.place}` : 'not seen'}</p> : null}
          {c.local_running && <p className="mut">No training server is reporting, so this add-on is detecting.</p>}
          {c.local_error && <p className="err">{c.local_error}</p>}
        </div>
      ))}
    </div>
    <p className="mut" style={{ marginTop: 12 }}>The main detector is a training server (see Training servers). When it stops reporting, this add-on runs the same detector itself until it’s back.</p>
  </>;
}

// ── Training servers ──────────────────────────────────────────────────────────
function Servers() {
  const { ov, loadOv } = useApp();
  const [st, setSt] = useState(null);
  const [pending, setPending] = useState({});  // settings sent but not yet confirmed by a reload
  const load = useCallback(() => api('api/servers').then(d => { setSt(d); return true; }).catch(() => false), []);
  usePoll('servers', load, 10000);
  async function setting(id, body) {
    setPending(p => ({ ...p, [id]: { ...p[id], ...body } }));
    try { await api(`api/servers/${id}/settings`, body); await load(); loadOv(); } catch (e) { toast(e.message); }
    setPending(p => { const n = { ...p }; delete n[id]; return n; });
  }
  async function remove(id, name) {
    if (!confirm(`Remove ${name}? It stops getting work. To uninstall it there: systemctl disable --now tortoise-worker`)) return;
    try { await api(`api/servers/${id}/remove`, {}); load(); loadOv(); } catch (e) { toast(e.message); }
  }
  async function pair() { try { await api('api/servers/pair', {}); load(); } catch (e) { toast(e.message); } }
  const copy = () => navigator.clipboard.writeText(st.install).then(() => toast('Copied'), () => toast('Select the line and copy it'));
  if (!st) return <><h2>Training servers</h2><p className="mut">Loading…</p></>;
  return <>
    <h2>Training servers</h2>
    <div className="card"><h3>Add a training server</h3>
      <p className="mut">On a Linux machine with systemd and Python 3.10+ that can reach this Home Assistant (an NVIDIA GPU is used if it has one), run as root:</p>
      <pre style={{ userSelect: 'all' }}>{st.install}</pre>
      <div className="row"><button onClick={copy}>Copy</button><button onClick={pair}>New pairing code</button><span className="mut">This code works once, until {at(st.pair_expires)}.</span></div>
    </div>
    {st.servers.map(s => {
      const i = s.info || {}, set = { ...(s.settings || {}), ...(pending[s.id] || {}) }, gp = i.gpus || [], off = s.update_offer, det = i.detect || {};
      const ver = i.version === st.version ? 'up to date'
        : set.auto_update === false ? `${i.version} (updates off)`
        : i.update_error ? <span className="err">{i.version}; the update failed: {i.update_error}</span>
        : !s.online && off && off.at >= (s.last_seen || 0) - 60 ? <span className="err">sent {off.version} at {at(off.at)} and hasn't checked in since: see <code>journalctl -u tortoise-worker</code> on that machine</span>
        : `${i.version || '?'}: updating to ${st.version} when idle`;
      return (
        <div className="card" key={s.id}>
          <h3><Dot cls={s.online ? 'ok' : 'bad'} /> {s.name} {!s.online && <span className="pill bad">offline</span>}{st.main === s.id && <> <span className="pill ok">main detector</span></>}</h3>
          <p className="mut">{i.host || ''} · {i.cpus || '?'} CPUs · {gp.length ? gp.map(g => g.name).join(', ') : 'no GPU'}{i.arbiter ? ' · GPU arbiter' : ''} · last check-in {at(s.last_seen)}</p>
          <p>Version: {ver}</p>
          <p>Detection: <select value={set.detect || 'cpu'} onChange={e => setting(s.id, { detect: e.target.value })}>
            <option value="off">Off</option>
            <option value="cpu">CPU (OpenVINO)</option>
            <option value="gpu" disabled={!gp.length}>GPU (a free card; the CPU while someone else needs it)</option>
          </select></p>
          {Object.keys(det).length > 0 && <p className="mut">{Object.entries(det).map(([c, d], n) => <React.Fragment key={c}>{n > 0 && ' · '}{camName(ov, c)}: {d.device ? d.device + (d.ms != null ? `, ${(+d.ms).toFixed(0)} ms` : '') : 'starting'}{d.error && <> <span className="err">{d.error}</span></>}</React.Fragment>)}</p>}
          <p className="mut">Jobs running: {(i.jobs || []).length ? i.jobs.join(', ') : 'none'}</p>
          <div className="row">
            <label><input type="checkbox" checked={set.auto_update !== false} onChange={e => setting(s.id, { auto_update: e.target.checked })} /> Update from this add-on</label>
            <button onClick={() => remove(s.id, s.name)}>Remove</button>
          </div>
        </div>
      );
    })}
    {!st.servers.length && <p className="mut" style={{ marginTop: 12 }}>No training servers yet. Until one is added, this add-on detects by itself and nothing can train.</p>}
  </>;
}

// ── Background jobs ───────────────────────────────────────────────────────────
const JOB_INFO = {
  collect: 'Copies frames from Frigate every hour: frames every few seconds from each camera’s recordings (night-time yard frames only one per 5 min), plus frames and snapshots from Frigate events. Frigate keeps its recordings about a day, so anything not collected within a day is gone.',
  label: 'Hands batches of frames to a training server, which puts first boxes on them with OWLv2 (on a GPU only while nobody else is using one): the frame at each safety check’s moment, and Frigate’s event snapshots. A tiled second look follows for yard frames the first look was unsure about. The safety check’s answer and these boxes label each frame; where they disagree, it goes to Review.',
  pass: 'Runs the detector in use over the collected frames here in the add-on (Intel GPU, else CPU), hourly. Frames where it and the first pass disagree go to Review.',
};
function Jobs({ name }) {
  const { ov: o, setOv } = useApp();
  const [log, setLog] = useState(null);
  const loadLog = useCallback(() => api('api/jobs/' + name + '/log').then(j => { setLog(j.log || ''); return true; }).catch(() => false), [name]);
  usePoll('log:' + name, loadLog, 10000);
  const s = o.jobs[name];
  if (!s) return null;
  async function act(action) {
    if (action === 'off' && name === 'collect' && !confirm('Frigate keeps recordings about a day. Frames not collected in that time are gone. Turn collection off?')) return;
    try { setOv(await api(`api/jobs/${name}/${action}`, {})); setTimeout(loadLog, 1500); } catch (e) { toast(e.message); }
  }
  const [cls, t] = jobState(name, s), C = o.cams, all = camsOf(o);
  const rows = {
    collect: [['Frames kept', c => c.collected], ['Collected in the last 24 h', c => c.collected_24h], ['Newest frame', c => at(c.newest)]],
    label: [['Frames with first boxes', c => c.counts.boxed || 0], ['Waiting for first boxes', c => c.waiting_first_pass], ['Sent to Review, not answered', c => c.counts.to_review || 0]],
    pass: [['Detector in use', c => c.model ? 'run ' + c.model.run : 'none'], ['Frames not yet checked', (c, cam) => c.model ? ((s.waiting || {})[cam] ?? '—') : '—']],
  }[name];
  let state;
  if (name === 'label') {
    const j = s.job, p = s.progress || {};
    state = !j ? <p className="mut">No batch yet.</p>
      : j.status === 'running' ? <p><span className="pill acc">Running</span> on a training server{p.total ? `: ${p.done} of ${p.total} frames, ${p.device === 'cpu' ? 'CPU' : 'GPU ' + p.device}, ${p.per_frame_s} s a frame` : ''}</p>
      : j.status === 'queued' ? <p><span className="pill acc">Waiting</span> for a training server to take the batch</p>
      : <p className="mut">Last batch {j.status}{j.finished ? ' ' + at(j.finished) : ''}</p>;
  } else {
    const took = !s.running && s.last_start && s.last_end && s.last_end >= s.last_start ? dur(s.last_end - s.last_start) : '';
    const failed = s.last_ok === false || !!s.error;
    state = <>
      {s.running ? <><p><span className="pill acc">Running now</span></p>{s.last_start ? <p className="mut">Started {at(s.last_start)}</p> : null}</>
        : s.last_end ? <><p>Finished {at(s.last_end)}{took ? ` · took ${took}` : ''}</p><p className={failed ? 'err' : 'mut'}>{failed ? 'Failed: ' + (s.error || 'see the log') : 'Succeeded'}</p></>
        : <p className="mut">Not run since the add-on started</p>}
      {name === 'pass' && s.device ? <p className="mut">On {s.device}</p> : null}
    </>;
  }
  return <>
    <h2>{JOBS[name]} <span className={'pill ' + (cls === 'ok' ? 'ok' : cls === 'bad' ? 'bad' : 'acc')} style={{ verticalAlign: 'middle' }}>{t}</span></h2>
    <p className="mut" style={{ maxWidth: 800, marginTop: 0 }}>{JOB_INFO[name]}</p>
    <div className="row" style={{ margin: '12px 0 16px' }}>
      <button onClick={() => act(s.on ? 'off' : 'on')}>{s.on ? 'Turn off' : 'Turn on'}</button>
      {name !== 'label' && !s.running && <button onClick={() => act('run')}>Run now</button>}
    </div>
    <div className="cards">
      <div className="card"><h3>Schedule</h3>
        <p>{!s.on ? 'Off — not scheduled' : name === 'label' ? 'A batch whenever frames are waiting and a server is free' : 'Hourly'}</p>
        {s.on && s.next ? <p className="mut">Next run {at(s.next)}</p> : null}
      </div>
      <div className="card"><h3>Now</h3>{state}</div>
      <div className="card"><h3>By camera</h3>
        <table className="kv"><tbody>
          <tr><th />{all.map(c => <th key={c}>{camName(o, c)}</th>)}</tr>
          {rows.map(([l, f]) => <tr key={l}><td>{l}</td>{all.map(c => <td key={c}>{f(C[c], c)}</td>)}</tr>)}
        </tbody></table>
      </div>
    </div>
    <h3 style={{ margin: '20px 0 6px' }}>Log</h3>
    <pre style={{ maxHeight: '60vh' }}>{log == null ? 'Loading…' : log || '(nothing logged since the add-on started)'}</pre>
  </>;
}

// ── Storage ───────────────────────────────────────────────────────────────────
function Storage() {
  const [st, setSt] = useState(null);
  const [err, setErr] = useState('');
  useEffect(() => { api('api/storage').then(setSt).catch(e => setErr(e.message)); }, []);
  if (err) return <><h2>Storage</h2><div className="card"><p className="err">Couldn't load: {err}</p></div></>;
  if (!st) return <><h2>Storage</h2><p className="mut">Loading…</p></>;
  const s = st.state || {}, gb = b => b == null ? '—' : (b / 1e9).toFixed(1) + ' GB';
  return <>
    <h2>Storage</h2>
    <div className="cards">
      <div className="card"><h3>Disk</h3>
        <p><b>{s.used_percent ?? '—'}%</b> used · {gb(s.free)} free of {gb(s.total)}</p>
        <div style={{ height: 10, borderRadius: 5, background: 'var(--btn)', overflow: 'hidden' }}>
          <div style={{ height: '100%', width: `${s.used_percent || 0}%`, background: (s.used_percent || 0) >= st.limit ? 'var(--bad)' : 'var(--acc)' }} />
        </div>
        <p className="mut">Frames here: {st.frames} ({gb(s.ours)}) · never deleted: {st.protected} (answered in Review or used by a run)</p>
        {s.collection_paused && <p className="err">Collection is paused: the disk is over the limit even after deleting everything that can be lost.</p>}
      </div>
      <div className="card"><h3>At {st.limit}% full</h3>
        <p className="mut">The oldest of these go first, a kind at a time, until the disk is 1% under:</p>
        <ol style={{ margin: '4px 0 0 18px', padding: 0 }}>
          <li>frames nothing uses: no safety check labels them and Review doesn’t ask about them (frames that never got first boxes, after 7 days)</li>
          <li>safety-check frames without him, unused by any run</li>
          <li>frames Review asks about that you haven’t answered</li>
          <li>safety-check frames with him, unused by any run</li>
          <li>detection history older than 90 days</li>
        </ol>
      </div>
    </div>
    <div className="card" style={{ marginTop: 12 }}><h3>Deleted</h3>
      {st.deletions.length
        ? <table className="kv"><tbody><tr><th>When</th><th>What</th><th>Frames</th><th>Space</th></tr>
          {st.deletions.map((d, i) => <tr key={i}><td>{at(d.at)}</td><td>{d.tier}</td><td>{d.files}</td><td>{gb(d.bytes)}</td></tr>)}</tbody></table>
        : <p className="mut">Nothing so far.</p>}
    </div>
  </>;
}

// ── App ───────────────────────────────────────────────────────────────────────
function Page({ route }) {
  const { page, arg } = route;
  if (page === 'review') return <Review key={arg} cam={arg} />;
  if (page === 'outline') return <Outline />;
  if (page === 'training') return <Training key={arg} focus={arg ? +arg : null} />;
  if (page === 'results') return <Results key={arg} cam={arg} />;
  if (page === 'heatmap') return <Heatmap key={arg} cam={arg} />;
  if (page === 'live') return <Live />;
  if (page === 'servers') return <Servers />;
  if (page === 'jobs') return JOBS[arg] ? <Jobs key={arg} name={arg} /> : null;
  if (page === 'storage') return <Storage />;
  return null;
}

function ConnectionBanner() {
  return <div className="conn-banner">Lost contact with the add-on — retrying. Showing what it last reported at {new Date(conn.lastOkAt).toLocaleTimeString([], { hour: '2-digit', minute: '2-digit', second: '2-digit' })}.</div>;
}

function App() {
  const route = useRoute();
  const [ov, setOv] = useState(null);
  const [revCounts, setRevCounts] = useState(null);
  const [drawer, setDrawer] = useState(false);
  const loadOv = useCallback(() => api('api/overview').then(o => { setOv(o); setRevCounts(null); return true; }).catch(() => false), []);
  usePoll('overview', loadOv, 10000);
  useEffect(() => { if (route.page === 'jobs' && !JOBS[route.arg]) location.replace('#jobs/collect'); }, [route]);
  useEffect(() => setDrawer(false), [route]);
  const lost = useConnectionLost();
  const { resuming } = usePageState();
  const ctx = useMemo(() => ({ ov, setOv, loadOv, revCounts, setRevCounts }), [ov, loadOv, revCounts]);
  return (
    <Ctx.Provider value={ctx}>
      <div className={'app' + (drawer ? ' drawer' : '')}>
        <aside><div className="brand">Tortoise Monitor</div>{ov && <Nav route={route} onPick={() => setDrawer(false)} />}</aside>
        <div className="scrim" onClick={() => setDrawer(false)} />
        <div className="content">
          <div className="topbar"><button aria-label="Menu" onClick={() => setDrawer(d => !d)}>☰</button><b>{route.page === 'jobs' ? JOBS[route.arg] : PAGES[route.page]}</b></div>
          {lost && <ConnectionBanner />}
          <main className={lost || resuming ? 'stale' : ''}>
            {ov ? <Page route={route} /> : <p className="mut">{lost ? 'Waiting for the add-on…' : 'Loading…'}</p>}
          </main>
        </div>
      </div>
      <Toast />
    </Ctx.Provider>
  );
}

createRoot(document.getElementById('root')).render(<App />);
