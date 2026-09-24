"""
patch_session_frontend.py

One-time patch: makes frontend_dashboard/index.html's Session Engine
page actually render real /api/session-check data, instead of the
hardcoded stub text and the hardcoded "(Server stub active)" label.

Safe by construction: each old_text must match EXACTLY ONCE, or this
aborts with an error and changes nothing.
"""

from pathlib import Path

TARGET = Path("frontend_dashboard/index.html")

text = TARGET.read_text(encoding="utf-8")

edits = [
    (
        '<div class="sessions-strip" id="sessionsStrip"><div class="empty">Session engine check is currently a labeled server stub.</div></div>',
        '<div class="sessions-strip" id="sessionsStrip"><div class="empty">Click "Check now" to load real session status.</div></div>',
    ),
    (
        '''async function runSessionCheck() {
  const timeLabel = new Date().toLocaleTimeString(undefined, {hour12: false});
  try {
    if (DEMO) throw new Error('Session check is disabled in demo mode.');
    const data = await getJSON('/api/session-check', {method: 'POST'});
    $('lastCheckLabel').textContent = 'Last check: ' + timeLabel + ' (Server stub active)';
    addFeedRow(timeLabel, data.note || ('Check completed: ' + (data.status || 'ok')));
  } catch (e) {
    $('lastCheckLabel').textContent = 'Last check failed at ' + timeLabel;
    addFeedRow(timeLabel, 'Check failed: ' + e.message, 'err');
  }
}''',
        '''function renderSessionsStrip(data) {
  const strip = $('sessionsStrip');
  const sessions = data.sessions || [];
  if (!sessions.length) { strip.innerHTML = '<div class="empty">No session occurrences found.</div>'; return; }
  strip.innerHTML = sessions.map((s) => {
    return '<div class="sess-card"><b>' + esc(s.session_name) + '</b>'
      + '<span>' + esc(s.status) + ' &middot; ' + esc(s.decision) + '</span>'
      + '<span>' + esc(fmtTime(s.scheduled_start)) + '</span></div>';
  }).join('');
}

async function runSessionCheck() {
  const timeLabel = new Date().toLocaleTimeString(undefined, {hour12: false});
  try {
    if (DEMO) throw new Error('Session check is disabled in demo mode.');
    const data = await getJSON('/api/session-check', {method: 'POST'});
    $('lastCheckLabel').textContent = 'Last check: ' + timeLabel;
    renderSessionsStrip(data);
    const cur = data.current_session;
    const nxt = data.next_session;
    const msg = cur
      ? cur.session_name + ' session ACTIVE'
      : (nxt ? 'No active session. Next: ' + nxt.session_name + ' in ' + Math.round(data.minutes_until_next_session) + ' min' : 'No active or upcoming session');
    addFeedRow(timeLabel, msg);
  } catch (e) {
    $('lastCheckLabel').textContent = 'Last check failed at ' + timeLabel;
    addFeedRow(timeLabel, 'Check failed: ' + e.message, 'err');
  }
}''',
    ),
]

for i, (old, new) in enumerate(edits, start=1):
    count = text.count(old)
    if count != 1:
        raise SystemExit(
            f"ABORTED, no changes written: edit {i} matched {count} times "
            f"(expected exactly 1). File not touched."
        )
    text = text.replace(old, new, 1)

TARGET.write_text(text, encoding="utf-8")
print("Patched frontend_dashboard/index.html successfully -- 2 edits applied.")