'use strict';

// ---------- Chord labels ----------
// On screen: A, A:min, A:maj7, G#7, A:add9, D/F#, N (sharps only).
// On disk (Harte, see Annotate_Chords/README.md): A:maj, A:min, A:maj7, G#:7, A:maj(9), D:maj/3, N.
const NOTES = ['C', 'C#', 'D', 'D#', 'E', 'F', 'F#', 'G', 'G#', 'A', 'A#', 'B'];
// quality -> [on-screen suffix, Harte quality]
const QINFO = {
  maj: ['', 'maj'], min: [':min', 'min'], '7': ['7', '7'], maj7: [':maj7', 'maj7'], min7: [':min7', 'min7'],
  sus2: [':sus2', 'sus2'], sus4: [':sus4', 'sus4'], dim: [':dim', 'dim'], aug: [':aug', 'aug'],
  add9: [':add9', 'maj(9)'], '9': ['9', '9'], min9: [':min9', 'min9'],
  hdim7: [':m7b5', 'hdim7'], dim7: [':dim7', 'dim7'], '7sus4': ['7sus4', 'sus4(b7)'],
  '6': ['6', 'maj6'], min6: [':min6', 'min6'], '13': ['13', '13'],
};
// Explicit order: Object.keys would put numeric keys ('6', '7', '9', '13') first.
const QUALS = ['maj', 'min', '7', 'maj7', 'min7', 'sus2', 'sus4', 'dim', 'aug',
  'add9', '9', 'min9', 'hdim7', 'dim7', '7sus4', '6', 'min6', '13'];
const QBUTTON = { hdim7: 'm7b5' };
const ENHARM = { Db: 'C#', Eb: 'D#', Gb: 'F#', Ab: 'G#', Bb: 'A#', Cb: 'B', Fb: 'E', 'E#': 'F', 'B#': 'C' };
// Accepts both the on-screen spelling and Harte.
const QALIAS = { '': 'maj', maj: 'maj', M: 'maj', min: 'min', m: 'min', '7': '7', dom7: '7', maj7: 'maj7', M7: 'maj7',
  min7: 'min7', m7: 'min7', sus2: 'sus2', sus4: 'sus4', sus: 'sus4', dim: 'dim', aug: 'aug', '+': 'aug',
  add9: 'add9', 'maj(9)': 'add9', '(9)': 'add9', '9': '9', min9: 'min9', m9: 'min9',
  hdim7: 'hdim7', m7b5: 'hdim7', dim7: 'dim7', '7sus4': '7sus4', 'sus4(b7)': '7sus4',
  '6': '6', maj6: '6', min6: 'min6', m6: 'min6', '13': '13' };
// Harte bass interval <-> semitones above the root (same names as labeling/chord_utils.py).
const BASS_IV = ['1', 'b2', '2', 'b3', '3', '4', '#4', '5', 'b6', '6', 'b7', '7'];
const DEGREE = { b2: 1, '2': 2, b3: 3, '3': 4, '4': 5, '#4': 6, b5: 6, '5': 7, '#5': 8, b6: 8, '6': 9, b7: 10, '7': 11 };

function normNote(n) {
  if (!n) return null;
  n = n.replace('♯', '#').replace('♭', 'b');
  // Note names only. Leave Harte degrees like b7 alone (the leading b is an accidental).
  if (/^[A-Ga-g][#b♯♭]?$/.test(n)) n = n[0].toUpperCase() + n.slice(1);
  if (ENHARM[n]) return ENHARM[n];
  return NOTES.includes(n) ? n : null;
}

function parseChord(lab) {
  lab = (lab || '').trim();
  if (!lab || lab === 'N') return { root: null };
  if (lab === 'X') return { raw: 'X' };
  let main = lab, bass = null;
  const sl = lab.lastIndexOf('/');
  if (sl > 0) { main = lab.slice(0, sl); bass = lab.slice(sl + 1); }
  const m = main.match(/^([A-Ga-g][#b♯♭]?):?(.*)$/);
  if (!m || !normNote(m[1])) return { raw: lab };
  const root = normNote(m[1]);
  const qual = QALIAS[m[2]] !== undefined ? QALIAS[m[2]] : m[2];
  if (bass !== null) {
    bass = normNote(bass) || (DEGREE[bass] !== undefined ? NOTES[(NOTES.indexOf(root) + DEGREE[bass]) % 12] : null);
  }
  return { root, qual, bass };
}

function formatChord(c) {
  if (c.raw) return c.raw;
  if (!c.root) return 'N';
  let s = c.root + (QINFO[c.qual] ? QINFO[c.qual][0] : ':' + c.qual);
  if (c.bass && c.bass !== c.root) s += '/' + c.bass;
  return s;
}

const canon = (lab) => formatChord(parseChord(lab));

// Unknown qualities (e.g. maj9 typed in a file) are assumed to already be Harte and pass through.
function toHarte(lab) {
  const c = parseChord(lab);
  if (c.raw) return c.raw;
  if (!c.root) return 'N';
  let s = c.root + ':' + (QINFO[c.qual] ? QINFO[c.qual][1] : c.qual);
  if (c.bass && c.bass !== c.root) s += '/' + BASS_IV[(NOTES.indexOf(c.bass) - NOTES.indexOf(c.root) + 12) % 12];
  return s;
}

function chordHue(lab) {
  const c = parseChord(lab);
  if (!c.root) return null;
  return NOTES.indexOf(c.root) * 30;
}

// ---------- State ----------
const S = {
  session: null,          // {id, videoId, title, duration, tempo, beats}
  grid: { mode: 'detected', scale: 1, bpm: 120, offset: 0, bpb: 4, shift: 0, sub: 1, bpr: 4 },
  starts: [],             // cell start times; cell i = [starts[i], starts[i+1] or duration)
  beatOf: [],             // cell index -> beat index
  marks: [],              // chord label at a change, null = continues previous
  sel: null, anchor: 0,
  arm: null,             // {sharp} while waiting for the bass note after "/"
  undo: [], redo: [],
  nowCell: -1, audioDeleted: false,
  labPath: null,          // draft .lab being edited (relative to the server's LAB_DIR)
  labs: [],               // [{path, url, saved}] from /api/labs
  lastSaved: null,        // Harte text last written to disk
};

const $ = (id) => document.getElementById(id);
const audio = $('audio');

function setStatus(msg, err) { const el = $('status'); el.textContent = msg; el.className = err ? 'err' : ''; }

function fmtTime(t, dec = 2) {
  const m = Math.floor(t / 60), s = t - m * 60;
  return m + ':' + (s < 10 ? '0' : '') + s.toFixed(dec);
}

// ---------- Grid ----------
function baseBeats() {
  const dur = S.session.duration, g = S.grid;
  let b;
  if (g.mode === 'manual') {
    b = [];
    const p = 60 / g.bpm;
    for (let t = g.offset; t < dur; t += p) b.push(t);
  } else {
    b = S.session.beats.slice();
    if (g.scale === 2) b = b.flatMap((t, i) => i + 1 < b.length ? [t, (t + b[i + 1]) / 2] : [t]);
    if (g.scale === 0.5) b = b.filter((_, i) => i % 2 === 0);
  }
  if (b.length < 2) b = [0, dur / 2];
  const iv = b.slice(1).map((t, i) => t - b[i]).sort((x, y) => x - y);
  const med = iv[Math.floor(iv.length / 2)];
  while (b[0] - med > med * 0.3) b.unshift(b[0] - med);
  while (dur - b[b.length - 1] > med * 1.3) b.push(b[b.length - 1] + med);
  b = b.filter((t) => t < dur - 0.05);
  b[0] = 0;
  return b;
}

function buildCells() {
  const beats = baseBeats(), sub = S.grid.sub;
  S.starts = []; S.beatOf = [];
  beats.forEach((t, k) => {
    const next = k + 1 < beats.length ? beats[k + 1] : S.session.duration;
    for (let j = 0; j < sub; j++) { S.starts.push(t + (next - t) * j / sub); S.beatOf.push(k); }
  });
}

const cellEnd = (i) => i + 1 < S.starts.length ? S.starts[i + 1] : S.session.duration;

function effective() {
  const eff = new Array(S.marks.length);
  for (let i = 0; i < S.marks.length; i++) eff[i] = S.marks[i] !== null ? S.marks[i] : eff[i - 1];
  return eff;
}

function normalize() {
  if (S.marks[0] === null) S.marks[0] = 'N';
  let cur = S.marks[0];
  for (let i = 1; i < S.marks.length; i++) {
    if (S.marks[i] === null) continue;
    if (S.marks[i] === cur) S.marks[i] = null; else cur = S.marks[i];
  }
}

function segments() {
  const eff = effective(), out = [];
  eff.forEach((lab, i) => {
    if (out.length && out[out.length - 1].label === lab) out[out.length - 1].e = cellEnd(i);
    else out.push({ s: S.starts[i], e: cellEnd(i), label: lab });
  });
  return out;
}

// Assign each cell the label with the largest overlap in the timeline.
function resample(timeline) {
  const n = S.starts.length, eff = new Array(n);
  let j = 0;
  for (let i = 0; i < n; i++) {
    const a = S.starts[i], b = cellEnd(i), ov = {};
    while (j < timeline.length && timeline[j].e <= a) j++;
    for (let k = j; k < timeline.length && timeline[k].s < b; k++) {
      const o = Math.min(b, timeline[k].e) - Math.max(a, timeline[k].s);
      if (o > 0) ov[timeline[k].label] = (ov[timeline[k].label] || 0) + o;
    }
    let best = null, bv = 0;
    for (const lab in ov) if (ov[lab] > bv) { best = lab; bv = ov[lab]; }
    eff[i] = best || 'N';
  }
  S.marks = eff.map((lab, i) => (i === 0 || lab !== eff[i - 1]) ? lab : null);
}

function regrid() {
  const tl = S.starts.length ? segments() : [{ s: 0, e: S.session.duration, label: 'N' }];
  buildCells();
  resample(tl);
  S.sel = null; S.undo = []; S.redo = [];
  render();
  save();
}

// ---------- Rendering ----------
function render() {
  const g = S.grid, eff = effective(), grid = $('grid');
  const perRow = g.bpb * g.bpr;
  grid.innerHTML = '';
  grid.style.setProperty('--cols', perRow * g.sub);
  let rowEl = null, rowIdx = -1, cellsEl = null, filled = 0;
  const slotOf = (k) => g.shift ? k - g.shift + g.bpb : k;
  S.cellEls = [];
  for (let i = 0; i < S.starts.length; i++) {
    const k = S.beatOf[i], slot = slotOf(k), r = Math.floor(slot / perRow);
    const j = i % g.sub, col = (slot % perRow) * g.sub + j;
    if (r !== rowIdx) {
      rowIdx = r;
      rowEl = document.createElement('div'); rowEl.className = 'grow';
      const tm = document.createElement('div'); tm.className = 'gtime'; tm.textContent = fmtTime(S.starts[i], 1);
      cellsEl = document.createElement('div'); cellsEl.className = 'gcells';
      cellsEl.style.gridTemplateColumns = `repeat(${perRow * g.sub}, 1fr)`;
      rowEl.append(tm, cellsEl); grid.append(rowEl); filled = 0;
    }
    for (; filled < col; filled++) {
      const b = document.createElement('div'); b.className = 'cell blank'; cellsEl.append(b);
    }
    const el = document.createElement('div');
    el.className = 'cell';
    if (slot % g.bpb === 0 && j === 0) el.classList.add('bar0');
    if (j) el.classList.add('sub2');
    const hue = chordHue(eff[i]);
    el.style.background = hue === null ? '' : `hsl(${hue} 55% var(--l))`;
    if (S.marks[i] !== null) {
      el.classList.add('change');
      const sp = document.createElement('span'); sp.textContent = S.marks[i]; el.append(sp);
    }
    el.dataset.i = i;
    el.title = `${fmtTime(S.starts[i])} · ${eff[i]}`;
    cellsEl.append(el); filled++;
    S.cellEls.push(el);
  }
  S.nowCell = -1;
  paintSel();
}

function paintSel() {
  S.cellEls.forEach((el) => el.classList.remove('sel'));
  const info = $('selInfo');
  const input = $('chordInput');
  if (!S.sel) {
    info.textContent = 'Klik kotak untuk memilih';
    if (input) { input.value = ''; input.disabled = true; }
    paintPicker(null);
    return;
  }
  for (let i = S.sel.a; i <= S.sel.b; i++) S.cellEls[i].classList.add('sel');
  const eff = effective(), lab = eff[S.sel.a];
  const n = S.sel.b - S.sel.a + 1;
  const c = parseChord(lab);
  info.textContent = `${fmtTime(S.starts[S.sel.a])}–${fmtTime(cellEnd(S.sel.b))} · ${n} kotak · ${lab}`;
  if (c && c.bass) info.textContent += ` · file ${toHarte(lab)}`;
  if (S.arm) info.textContent += S.arm.sharp ? ' · bass #: tekan A–G' : ' · bass: tekan A–G';
  if (input) {
    input.disabled = false;
    if (document.activeElement !== input) input.value = lab || '';
  }
  paintPicker(lab);
}

function buildPicker() {
  const mk = (rowId, title, items, onClick) => {
    const row = $(rowId); row.innerHTML = `<span class="lbl">${title}</span>`;
    items.forEach(([val, text]) => {
      const b = document.createElement('button'); b.textContent = text; b.dataset.v = val;
      b.onclick = () => onClick(val); row.append(b);
    });
  };
  mk('rootRow', 'Root', [...NOTES.map((n) => [n, n]), ['N', 'N']], pickRoot);
  mk('qualRow', 'Jenis', QUALS.map((q) => [q, QBUTTON[q] || q]), pickQual);
  mk('bassRow', 'Bass', [['', '—'], ...NOTES.map((n) => [n, '/' + n])], pickBass);
}

function paintPicker(lab) {
  const c = lab ? parseChord(lab) : null;
  const set = (rowId, v, enabled) => $(rowId).querySelectorAll('button').forEach((b) => {
    b.classList.toggle('on', c !== null && b.dataset.v === v);
    b.disabled = !enabled;
  });
  set('rootRow', c && (c.root || 'N'), !!S.sel);
  set('qualRow', c && c.root ? c.qual : null, !!(S.sel && c && c.root));
  set('bassRow', c && c.root ? (c.bass || '') : null, !!(S.sel && c && c.root));
}

// ---------- Editing ----------
function snapshot() { return JSON.stringify(S.marks); }
function pushUndo() { S.undo.push(snapshot()); if (S.undo.length > 300) S.undo.shift(); S.redo = []; }

function applyLabel(label) {
  if (!S.sel) return;
  const armed = S.arm;
  S.arm = null;
  const { a, b } = S.sel, eff = effective(), n = S.marks.length;
  let same = true;
  for (let i = a; i <= b; i++) if (eff[i] !== label) { same = false; break; }
  if (same) { if (armed) paintSel(); return; }
  pushUndo();
  if (b + 1 < n && S.marks[b + 1] === null) S.marks[b + 1] = eff[b + 1];
  S.marks[a] = label;
  for (let i = a + 1; i <= b; i++) S.marks[i] = null;
  normalize();
  render(); save();
}

function curChord() { return S.sel ? parseChord(effective()[S.sel.a]) : null; }
const shiftNote = (n, semi) => NOTES[(NOTES.indexOf(n) + semi + 120) % 12];
function pickRoot(r) {
  if (r === 'N') return applyLabel('N');
  const c = curChord();
  let bass = c && c.bass ? c.bass : null;
  if (bass === r) bass = null;
  applyLabel(formatChord({ root: r, qual: c && c.root && !c.raw ? c.qual : 'maj', bass }));
}
function pickQual(q) { const c = curChord(); if (c && c.root) applyLabel(formatChord({ ...c, qual: q })); }
function pickBass(bs) { const c = curChord(); if (c && c.root) applyLabel(formatChord({ ...c, bass: bs || null })); }

// Chord box: B/A, Bm/A, b/a, D/F#. Incomplete "B/" is left alone so the bass can still be typed.
function commitTyped() {
  const input = $('chordInput');
  if (!input || !S.sel) return false;
  const raw = input.value.trim().replace(/\s+/g, '');
  if (!raw) { input.value = effective()[S.sel.a] || ''; return false; }
  if (raw.endsWith('/')) { setStatus('Bass belum lengkap. Contoh: B/A', true); return false; }
  let label;
  if (raw === 'N' || raw === 'X') label = raw;
  else {
    const c = parseChord(raw);
    if (!c.root || c.raw || !QINFO[c.qual]) {
      setStatus('Chord tidak dikenali: ' + raw + '. Contoh: B/A, Bm/A, D/F#', true);
      return false;
    }
    label = formatChord(c);
  }
  applyLabel(label);
  input.value = label;
  if (label.includes('/')) setStatus(`${label} → file ${toHarte(label)}`);
  return true;
}

function undo(redo) {
  const from = redo ? S.redo : S.undo, to = redo ? S.undo : S.redo;
  if (!from.length) return;
  to.push(snapshot()); S.marks = JSON.parse(from.pop());
  render(); save();
}

function select(i, extend) {
  S.arm = null;
  i = Math.max(0, Math.min(S.starts.length - 1, i));
  if (extend) S.sel = { a: Math.min(S.anchor, i), b: Math.max(S.anchor, i) };
  else { S.sel = { a: i, b: i }; S.anchor = i; }
  paintSel();
  S.cellEls[i].scrollIntoView({ block: 'nearest' });
}

function selectSegment(i) {
  S.arm = null;
  let a = i, b = i;
  while (a > 0 && S.marks[a] === null) a--;
  while (b + 1 < S.marks.length && S.marks[b + 1] === null) b++;
  S.sel = { a, b }; S.anchor = a; paintSel();
}

// ---------- Playback ----------
function cellAt(t) {
  let lo = 0, hi = S.starts.length - 1;
  while (lo < hi) { const mid = (lo + hi + 1) >> 1; if (S.starts[mid] <= t) lo = mid; else hi = mid - 1; }
  return lo;
}

function tick() {
  if (S.session && S.starts.length) {
    const t = audio.currentTime;
    if ($('loop').checked && S.sel && !audio.paused && t >= cellEnd(S.sel.b)) audio.currentTime = S.starts[S.sel.a];
    $('time').textContent = fmtTime(t);
    if (!S.seeking) $('seek').value = t;
    const c = cellAt(t);
    if (c !== S.nowCell) {
      if (S.cellEls[S.nowCell]) S.cellEls[S.nowCell].classList.remove('now');
      S.nowCell = c;
      const el = S.cellEls[c];
      el.classList.add('now');
      if ($('follow').checked && !audio.paused) {
        const r = el.parentElement.getBoundingClientRect();
        if (r.top < 100 || r.bottom > window.innerHeight - 240) el.parentElement.scrollIntoView({ block: 'center', behavior: 'smooth' });
      }
    }
  }
  requestAnimationFrame(tick);
}

function togglePlay() { if (audio.paused) audio.play(); else audio.pause(); }
audio.onplay = audio.onpause = () => { $('playBtn').textContent = audio.paused ? '▶' : '❚❚'; };

// ---------- .lab import / export ----------
function parseLab(text) {
  const out = [];
  text.split(/\r?\n/).forEach((line) => {
    const p = line.trim().split(/[\s,]+/);
    if (p.length < 3 || isNaN(+p[0]) || isNaN(+p[1])) return;
    out.push({ s: +p[0], e: +p[1], label: canon(p.slice(2).join('')) });
  });
  return out.sort((x, y) => x.s - y.s);
}

function importLab(text) {
  const tl = parseLab(text);
  if (!tl.length) return setStatus('Tidak ada baris .lab yang valid', true);
  pushUndo();
  resample(tl);
  S.sel = null;
  render(); save();
  setStatus(`Import ${tl.length} segmen → ${segments().length} segmen di grid`);
}

function labText(fmt = (l) => l) {
  return segments().map((s) => `${s.s.toFixed(3)} ${s.e.toFixed(3)} ${fmt(s.label)}`).join('\n') + '\n';
}

async function copyLab() {
  const text = labText();
  try {
    await navigator.clipboard.writeText(text);
  } catch (e) {
    const ta = document.createElement('textarea');
    ta.value = text; document.body.append(ta); ta.select();
    document.execCommand('copy'); ta.remove();
  }
  setStatus(`${segments().length} segmen di-copy ke clipboard.`);
}

function deleteAudio() {
  if (!S.session || S.audioDeleted) return;
  S.audioDeleted = true;
  fetch(`/api/session/${S.session.id}`, { method: 'DELETE', keepalive: true }).catch(() => {});
}

// ---------- Autosave (overwrites the draft .lab, debounced) ----------
const SAVE_DELAY = 1200;
let saveTimer = null;

function setSaveState(state, detail) {
  const el = $('saveState');
  el.className = 'save-' + state;
  el.textContent = {
    pending: '● Ada perubahan…', saving: 'Menyimpan…',
    saved: '✓ Tersimpan' + (detail ? ' ' + new Date(detail).toLocaleTimeString() : ''),
    clean: 'Belum ada perubahan', error: '⚠ Gagal menyimpan: ' + detail + ' (klik untuk coba lagi)',
  }[state];
}

// Called after anything that may change the chords; only writes when the Harte text differs.
function save() {
  if (!S.session || !S.labPath) return;
  clearTimeout(saveTimer);
  if (labText(toHarte) === S.lastSaved) { saveTimer = null; if (!S.saving) setSaveState(S.savedAt ? 'saved' : 'clean', S.savedAt); return; }
  setSaveState('pending');
  saveTimer = setTimeout(flushSave, SAVE_DELAY);
}

async function flushSave(keepalive) {
  clearTimeout(saveTimer); saveTimer = null;
  if (!S.session || !S.labPath) return;
  const text = labText(toHarte);
  if (text === S.lastSaved) return;
  if (S.saving) { S.saveAgain = true; return; }
  S.saving = true; setSaveState('saving');
  try {
    const r = await fetch('/api/lab?path=' + encodeURIComponent(S.labPath), {
      method: 'PUT', headers: { 'Content-Type': 'application/json' }, body: JSON.stringify({ text }), keepalive: !!keepalive,
    });
    const d = await r.json();
    if (!r.ok) throw new Error(d.detail || r.statusText);
    S.lastSaved = text; S.savedAt = d.saved;
    const entry = S.labs.find((l) => l.path === S.labPath);
    if (entry) { entry.saved = d.saved; fillLabList(); }
    setSaveState('saved', d.saved);
  } catch (e) {
    setSaveState('error', e.message);
  } finally {
    S.saving = false;
    if (S.saveAgain) { S.saveAgain = false; flushSave(); }
  }
}

const unsaved = () => !!saveTimer || S.saving || (S.labPath && S.session && labText(toHarte) !== S.lastSaved);

// ---------- Song list ----------
async function loadLabList() {
  try {
    S.labs = await (await fetch('/api/labs')).json();
    fillLabList();
  } catch (e) { setStatus('Gagal membaca daftar .lab: ' + e.message, true); }
}

function fillLabList() {
  $('labList').innerHTML = '';
  S.labs.forEach((l) => {
    const o = document.createElement('option');
    o.value = l.path;
    o.textContent = l.saved ? '✓ dianotasi ' + new Date(l.saved).toLocaleDateString() : '';
    $('labList').append(o);
  });
}

function onLabPicked() {
  const entry = S.labs.find((l) => l.path === $('labPick').value);
  if (entry && entry.url) $('url').value = entry.url;
}

// ---------- History (songs worked on from this machine) ----------
function timeAgo(iso) {
  const sec = (Date.now() - new Date(iso)) / 1000;
  if (sec < 60) return 'baru saja';
  if (sec < 3600) return Math.floor(sec / 60) + ' menit lalu';
  if (sec < 86400) return Math.floor(sec / 3600) + ' jam lalu';
  if (sec < 172800) return 'kemarin';
  return new Date(iso).toLocaleDateString();
}

async function renderHistory(el) {
  let items = [];
  try { items = await (await fetch('/api/history')).json(); } catch (e) {}
  el.innerHTML = '';
  if (!items.length) {
    el.innerHTML = '<p class="muted">Belum ada. Lagu yang dibuka atau diedit akan muncul di sini.</p>';
    return;
  }
  items.slice(0, 50).forEach((h) => {
    const row = document.createElement('button');
    row.type = 'button'; row.className = 'hrow' + (h.path === S.labPath ? ' cur' : '');
    const name = document.createElement('strong'); name.textContent = h.path.replace(/\.lab$/, '');
    const meta = document.createElement('span'); meta.className = 'muted';
    meta.textContent = [
      h.title,
      h.saved ? `disimpan ${timeAgo(h.saved)} (${h.saves}×)` : 'belum diedit',
      h.opened ? `dibuka ${timeAgo(h.opened)}` : '',
    ].filter(Boolean).join(' · ');
    row.append(name, meta);
    row.onclick = () => openFromHistory(h);
    el.append(row);
  });
}

function openFromHistory(h) {
  if ($('historyDlg').open) $('historyDlg').close();
  $('labPick').value = h.path;
  $('url').value = h.url || '';
  if (h.url) loadSong(h.path, h.url);
  else { $('url').focus(); setStatus('Tempel link YouTube untuk lagu ini', true); }
}

// ---------- Load ----------
async function loadSong(path, url) {
  if (!S.labs.some((l) => l.path === path)) return setStatus('Pilih file .lab dari daftar dulu', true);
  setStatus('Download audio & deteksi beat… (bisa 10–60 detik)');
  $('loadForm').querySelector('button').disabled = true;
  try {
    await flushSave();
    const r = await fetch('/api/load', { method: 'POST', headers: { 'Content-Type': 'application/json' }, body: JSON.stringify({ url, lab: path }) });
    const d = await r.json();
    if (!r.ok) throw new Error(d.detail || r.statusText);
    const lr = await fetch('/api/lab?path=' + encodeURIComponent(path));
    const lab = await lr.json();
    if (!lr.ok) throw new Error(lab.detail || lr.statusText);
    if (S.session) deleteAudio();
    const blob = await (await fetch(`/api/audio/${d.id}`)).blob();
    audio.src = URL.createObjectURL(blob);
    S.session = d; S.audioDeleted = false; S.undo = []; S.redo = []; S.starts = [];
    S.labPath = path; S.savedAt = lab.saved || null;
    const entry = S.labs.find((l) => l.path === path);
    if (entry) entry.url = url;
    Object.assign(S.grid, { mode: 'detected', scale: 1, bpm: d.tempo, offset: d.beats[0] || 0, shift: 0, sub: 1 });
    buildCells();
    const tl = parseLab(lab.text);
    if (tl.length) resample(tl); else S.marks = S.starts.map((_, i) => i ? null : 'N');
    // The file on disk is the baseline: snapping it to the beat grid alone is not a change.
    S.lastSaved = labText(toHarte);
    syncGridInputs();
    $('songTitle').textContent = path.replace(/\.lab$/, '');
    $('songMeta').textContent = `${d.title} · ${fmtTime(d.duration, 0)} · ±${d.tempo.toFixed(1)} BPM terdeteksi`;
    $('seek').max = d.duration; $('dur').textContent = fmtTime(d.duration, 0);
    $('workspace').hidden = false; $('home').hidden = true;
    S.sel = null; render();
    setSaveState(S.savedAt ? 'saved' : 'clean', S.savedAt);
    setStatus(`Dimuat ${tl.length} segmen dari file. Perubahan tersimpan otomatis ke file itu.`);
  } catch (e) {
    setStatus(e.message, true);
  } finally {
    $('loadForm').querySelector('button').disabled = false;
  }
}

function syncGridInputs() {
  const g = S.grid;
  $('gMode').value = g.mode; $('gScale').value = g.scale; $('gBpm').value = (+g.bpm).toFixed(2);
  $('gOffset').value = (+g.offset).toFixed(3); $('gBpb').value = g.bpb; $('gSub').value = g.sub; $('gBpr').value = g.bpr;
  $('detOpts').hidden = g.mode !== 'detected'; $('manOpts').hidden = g.mode !== 'manual';
}

// ---------- Wiring ----------
buildPicker();
$('chordInput').addEventListener('keydown', (e) => {
  if (e.key === 'Enter') { e.preventDefault(); if (commitTyped()) e.target.blur(); }
  else if (e.key === 'Escape') {
    e.preventDefault();
    e.target.value = S.sel ? (effective()[S.sel.a] || '') : '';
    e.target.blur();
  }
});
$('chordInput').addEventListener('change', commitTyped);
loadLabList();
renderHistory($('homeHistory'));
requestAnimationFrame(tick);

$('loadForm').onsubmit = (e) => { e.preventDefault(); loadSong($('labPick').value, $('url').value.trim()); };
$('labPick').oninput = onLabPicked;
$('saveState').onclick = () => flushSave();
$('historyBtn').onclick = () => { renderHistory($('dlgHistory')); $('historyDlg').showModal(); };
$('playBtn').onclick = togglePlay;
$('seek').oninput = () => { S.seeking = true; audio.currentTime = +$('seek').value; };
$('seek').onchange = () => { S.seeking = false; };
$('rate').onchange = () => { audio.playbackRate = +$('rate').value; };
$('undoBtn').onclick = () => undo(false);
$('redoBtn').onclick = () => undo(true);
$('copyBtn').onclick = copyLab;
$('labFile').onchange = async (e) => { const f = e.target.files[0]; if (f) importLab(await f.text()); e.target.value = ''; };
$('pasteBtn').onclick = () => { $('pasteText').value = ''; $('pasteDlg').showModal(); };
$('pasteDlg').onclose = () => { if ($('pasteDlg').returnValue === 'ok') importLab($('pasteText').value); };

const gridChange = (fn) => () => { fn(); syncGridInputs(); regrid(); };
$('gMode').onchange = gridChange(() => { S.grid.mode = $('gMode').value; });
$('gScale').onchange = gridChange(() => { S.grid.scale = +$('gScale').value; });
$('gBpm').onchange = gridChange(() => { S.grid.bpm = Math.max(30, +$('gBpm').value || 120); });
$('gOffset').onchange = gridChange(() => { S.grid.offset = Math.max(0, +$('gOffset').value || 0); });
$('gSetFirst').onclick = gridChange(() => { S.grid.offset = audio.currentTime; });
$('gBpb').onchange = gridChange(() => { S.grid.bpb = +$('gBpb').value; S.grid.shift %= S.grid.bpb; });
$('gSub').onchange = gridChange(() => { S.grid.sub = +$('gSub').value; });
// Bar layout only changes display, not cell timing.
const layoutChange = (fn) => () => { fn(); syncGridInputs(); render(); save(); };
$('gBpr').onchange = layoutChange(() => { S.grid.bpr = +$('gBpr').value; });
$('gShiftL').onclick = layoutChange(() => { S.grid.shift = (S.grid.shift + S.grid.bpb - 1) % S.grid.bpb; });
$('gShiftR').onclick = layoutChange(() => { S.grid.shift = (S.grid.shift + 1) % S.grid.bpb; });

// Drag a chord label (a change cell) left/right to move where that chord starts.
// Dragging from any other cell selects a range.
function moveBoundary(base, from, to) {
  let next = from + 1;
  while (next < base.length && base[next] === null) next++;
  to = Math.max(0, Math.min(to, next - 1));   // never drag past the end of its own segment
  const m = base.slice(), label = base[from];
  if (to < from) { for (let k = to; k < from; k++) m[k] = null; m[to] = label; m[from] = null; }
  else if (to > from) { m[from] = null; m[to] = label; }
  S.marks = m;
  normalize();
}

let drag = null;
const cellFromPoint = (x, y) => {
  const el = document.elementFromPoint(x, y);
  const c = el && el.closest('.cell');
  return c && c.dataset.i !== undefined ? +c.dataset.i : null;
};
$('grid').addEventListener('mousedown', (e) => {
  const i = cellFromPoint(e.clientX, e.clientY);
  if (i === null || e.button !== 0 || e.shiftKey) return;
  e.preventDefault();
  drag = S.marks[i] !== null && i > 0
    ? { mode: 'move', from: i, last: i, base: S.marks.slice() }
    : { mode: 'sel', from: i, last: i };
});
document.addEventListener('mousemove', (e) => {
  if (!drag) return;
  const j = cellFromPoint(e.clientX, e.clientY);
  if (j === null || j === drag.last) return;
  drag.last = j; drag.moved = true;
  if (drag.mode === 'move') {
    moveBoundary(drag.base, drag.from, j);
    const keep = S.sel; render(); S.sel = keep; paintSel();
  } else {
    S.anchor = drag.from; S.sel = { a: Math.min(drag.from, j), b: Math.max(drag.from, j) }; paintSel();
  }
});
document.addEventListener('mouseup', () => {
  if (!drag) return;
  if (drag.moved && drag.mode === 'move') {
    S.undo.push(JSON.stringify(drag.base)); S.redo = [];
    save();
    const k = S.marks.findIndex((m, i) => i > 0 && m !== null && drag.base[i] === null);
    if (k >= 0) { S.sel = { a: k, b: k }; S.anchor = k; paintSel(); }
  }
  S.justDragged = drag.moved;
  drag = null;
});

$('grid').addEventListener('click', (e) => {
  if (S.justDragged) { S.justDragged = false; return; }
  const el = e.target.closest('.cell'); if (!el || el.dataset.i === undefined) return;
  const i = +el.dataset.i;
  select(i, e.shiftKey);
  if (!e.shiftKey) audio.currentTime = S.starts[i];
});
$('grid').addEventListener('dblclick', (e) => {
  const el = e.target.closest('.cell'); if (el && el.dataset.i !== undefined) selectSegment(+el.dataset.i);
});

document.addEventListener('keydown', (e) => {
  if (!S.session || (e.target.matches && e.target.matches('input, textarea, select')) || $('pasteDlg').open || $('historyDlg').open) return;
  const k = e.key, mod = e.metaKey || e.ctrlKey;
  if (mod && k.toLowerCase() === 'z') { e.preventDefault(); return undo(e.shiftKey); }
  if (mod && e.shiftKey && k.toLowerCase() === 'c') { e.preventDefault(); return copyLab(); }
  if (mod) return;
  const perRow = S.grid.bpb * S.grid.bpr * S.grid.sub;
  const c = curChord();
  const step = { ArrowLeft: -1, ArrowRight: 1, ArrowUp: -perRow, ArrowDown: perRow }[k];
  if (k === ' ') { e.preventDefault(); togglePlay(); }
  else if (step) {
    e.preventDefault();
    const from = S.sel ? (e.shiftKey ? (S.sel.a === S.anchor ? S.sel.b : S.sel.a) : S.sel.a) : S.nowCell;
    select(from + step, e.shiftKey);
    if (!e.shiftKey) audio.currentTime = S.starts[S.sel.a];
  }
  else if (k === 'Enter' && S.sel) { audio.currentTime = S.starts[S.sel.a]; audio.play(); }
  else if (k === 'Escape') { if (S.arm) S.arm = null; else S.sel = null; paintSel(); }
  else if (k.toLowerCase() === 'l') $('loop').checked = !$('loop').checked;
  else if (!S.sel) return;
  else if (k === '/' && c && c.root) {
    e.preventDefault();
    S.arm = S.arm && !S.arm.sharp ? null : { sharp: false };
    paintSel();
  }
  else if (S.arm && k === '#') { e.preventDefault(); S.arm.sharp = true; paintSel(); }
  else if (S.arm && /^[a-g]$/i.test(k)) {
    e.preventDefault();
    let n = k.toUpperCase();
    if (S.arm.sharp) n = shiftNote(n, 1);
    S.arm = null;
    pickBass(n);
  }
  else if (/^[a-g]$/i.test(k)) { S.arm = null; pickRoot(k.toUpperCase()); }
  else if (k === '#' && c && c.root) {
    const bass = c.bass ? shiftNote(c.bass, 1) : null;
    applyLabel(formatChord({ root: shiftNote(c.root, 1), qual: c.qual, bass }));
  }
  else if (k.toLowerCase() === 'n') pickRoot('N');
  else if (k.toLowerCase() === 'm' && c && c.root) pickQual(c.qual === 'min' ? 'maj' : 'min');
  else if (k === '7' && c && c.root) pickQual(c.qual === 'min' || c.qual === 'min7' ? 'min7' : '7');
  else if (k.toLowerCase() === 'j') pickQual('maj7');
  else if (k === '4') pickQual('sus4');
  else if (k === '2') pickQual('sus2');
  else if (k === 'Backspace' || k === 'Delete') {
    e.preventDefault();
    applyLabel(S.sel.a > 0 ? effective()[S.sel.a - 1] : 'N');
  }
});

// Flush a pending autosave and remove the temp audio when the tab closes.
window.addEventListener('beforeunload', (e) => { if (unsaved()) { flushSave(true); e.preventDefault(); e.returnValue = ''; } });
window.addEventListener('pagehide', () => { flushSave(true); deleteAudio(); });
