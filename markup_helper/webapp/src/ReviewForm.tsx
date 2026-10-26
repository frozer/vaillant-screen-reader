import type { RefObject } from 'react'
import { burnGasMismatch, changedFields, isDirty, setIcon, temperatureValid } from './draft'
import { ICONS, ICON_LABELS, type Action, type Draft, type Row } from './types'

interface Props {
  row: Row
  draft: Draft
  onDraft: (d: Draft) => void
  onAction: (a: Action) => void
  onReset: () => void
  tempRef: RefObject<HTMLInputElement | null>
  busy: boolean
  error: string | null
}

const fmt = (v: boolean | number | null) => (v === null ? '—' : String(v))

const STATUS_LABELS: Record<Row['status'], string> = {
  pending: 'Pending',
  confirmed: 'Confirmed',
  too_dark: 'Too dark',
  off: 'Off',
  updated: 'Updated',
}

export default function ReviewForm(props: Props) {
  const { row, draft, onDraft, onAction, onReset, tempRef, busy, error } = props
  const rec = row.recognized
  const changed = changedFields(draft, rec)
  const dirty = isDirty(draft, rec)
  const readable = draft.screen_state === 'ok'
  const tempOk = temperatureValid(draft.temperature)
  const t = Number(draft.temperature)
  const outOfRange = readable && draft.temperature.trim() !== '' && tempOk && (t < 42 || t > 81)

  return (
    <aside className="review-form">
      <section className={`saved status-${row.status}`}>
        <span className={`dot status-${row.status}`} /> {STATUS_LABELS[row.status]}
        {row.retrain_needed && <code>{row.retrain_needed}</code>}
      </section>

      <label className={`field temp ${changed.has('temperature') ? 'changed' : ''}`}>
        <span>
          Temperature <kbd>T</kbd>
        </span>
        <input
          ref={tempRef}
          inputMode="numeric"
          maxLength={2}
          value={draft.temperature}
          disabled={!readable}
          aria-invalid={!tempOk}
          onChange={(e) => onDraft({ ...draft, temperature: e.target.value })}
        />
        {changed.has('temperature') && <small>was {fmt(rec.temperature)}</small>}
        {!tempOk && <small className="error">0–99 only</small>}
        {outOfRange && <small className="warn">outside trained range 42–81</small>}
      </label>

      <fieldset disabled={!readable}>
        {ICONS.map((icon, i) => (
          <label key={icon} className={`field check ${changed.has(icon) ? 'changed' : ''}`}>
            <input
              type="checkbox"
              checked={draft[icon]}
              onChange={(e) => onDraft(setIcon(draft, icon, e.target.checked))}
            />
            <span>{ICON_LABELS[icon]}</span>
            {changed.has(icon) && <small>was {fmt(rec[icon])}</small>}
            <kbd>{i + 1}</kbd>
          </label>
        ))}
      </fieldset>

      <label className="field check too-dark">
        <input
          type="checkbox"
          checked={draft.screen_state === 'unreadable'}
          onChange={(e) => onDraft({ ...draft, screen_state: e.target.checked ? 'unreadable' : 'ok' })}
        />
        <span>Too dark</span>
        {draft.screen_state === 'off' && <small>screen off</small>}
      </label>

      {burnGasMismatch(rec) && (
        <p className="hint warn">Burn/GasValve mismatch in recognized values</p>
      )}
      {error && <p className="hint error">{error}</p>}

      <div className="actions">
        <button className="dark" disabled={busy} onClick={() => onAction('too_dark')}>
          Too Dark <kbd>D</kbd>
        </button>
        <button className="off" disabled={busy} onClick={() => onAction('off')}>
          Off <kbd>O</kbd>
        </button>
        <button className="confirm" disabled={busy || dirty} onClick={() => onAction('confirm')}>
          Confirm <kbd>↵</kbd>
        </button>
        <button
          className="update"
          disabled={busy || !dirty || !readable || !tempOk}
          onClick={() => onAction('update')}
        >
          Update <kbd>↵</kbd>
        </button>
        <button className="reset" disabled={busy} onClick={onReset}>
          Reset <kbd>⌫</kbd>
        </button>
      </div>

      <p className="keys">
        <kbd>J</kbd>/<kbd>K</kbd> next/prev · <kbd>Esc</kbd> leave input
      </p>
    </aside>
  )
}
