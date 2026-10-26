import { useCallback, useEffect, useRef, useState } from 'react'
import * as api from './api'
import { draftFromReading, draftToValues, isDirty, setIcon, temperatureValid } from './draft'
import FileList, { matchesFilter, type Filter } from './FileList'
import ImageView from './ImageView'
import ReviewForm from './ReviewForm'
import { ICONS, type Action, type Draft, type Row } from './types'

const PRELOAD = 3

/** Index of the next pending row after `from`, wrapping around; `from` if none. */
function nextPending(rows: Row[], from: number) {
  for (let step = 1; step <= rows.length; step++) {
    const idx = (from + step) % rows.length
    if (rows[idx].status === 'pending') return idx
  }
  return from
}

export default function App() {
  const [rows, setRows] = useState<Row[] | null>(null)
  const [selected, setSelected] = useState(0)
  const [filter, setFilter] = useState<Filter>('all')
  const [draft, setDraft] = useState<Draft | null>(null)
  const [bright, setBright] = useState(false)
  const [zoom, setZoom] = useState(false)
  const [busy, setBusy] = useState(false)
  const [error, setError] = useState<string | null>(null)
  const tempRef = useRef<HTMLInputElement>(null)

  useEffect(() => {
    api
      .fetchRows()
      .then((loaded) => {
        setRows(loaded)
        const first = loaded.findIndex((r) => r.status === 'pending')
        setSelected(first === -1 ? 0 : first)
      })
      .catch((e) => setError(String(e)))
  }, [])

  const row = rows?.[selected] ?? null

  // Reset the form whenever another row is shown or the shown row was saved.
  useEffect(() => {
    if (row) setDraft(draftFromReading(row.current))
    setError(null)
  }, [row])

  useEffect(() => {
    if (!rows) return
    for (let i = 1; i <= PRELOAD; i++) {
      const next = rows[(selected + i) % rows.length]
      if (next.image_exists) new Image().src = api.imageUrl(next.idx)
    }
  }, [rows, selected])

  const replaceRow = useCallback(
    (updated: Row) => {
      const list = rows!.map((r) => (r.idx === updated.idx ? updated : r))
      setRows(list)
      return list
    },
    [rows],
  )

  const act = useCallback(
    async (action: Action) => {
      if (!row || !draft || busy) return
      setBusy(true)
      try {
        const values = action === 'update' ? draftToValues(draft) : undefined
        const updated = await api.review(row.idx, action, values)
        setSelected(nextPending(replaceRow(updated), row.idx))
      } catch (e) {
        setError(e instanceof Error ? e.message : String(e))
      } finally {
        setBusy(false)
      }
    },
    [row, draft, busy, replaceRow],
  )

  const resetRow = useCallback(async () => {
    if (!row) return
    if (row.status === 'pending') {
      setDraft(draftFromReading(row.recognized))
      return
    }
    try {
      replaceRow(await api.reset(row.idx))
    } catch (e) {
      setError(e instanceof Error ? e.message : String(e))
    }
  }, [row, replaceRow])

  /** Enter: the action the form currently expresses. */
  const primary = useCallback(() => {
    if (!row || !draft) return
    if (draft.screen_state === 'unreadable') return act('too_dark')
    if (draft.screen_state === 'off') return act('off')
    if (!isDirty(draft, row.recognized)) return act('confirm')
    if (temperatureValid(draft.temperature)) return act('update')
  }, [row, draft, act])

  const move = useCallback(
    (delta: number) => {
      if (!rows) return
      // Step through the rows visible under the current filter.
      const visible = rows.filter((r) => matchesFilter(r, filter) || r.idx === selected)
      const pos = visible.findIndex((r) => r.idx === selected)
      const next = visible[pos + delta]
      if (next) setSelected(next.idx)
    },
    [rows, filter, selected],
  )

  useEffect(() => {
    const onKey = (e: KeyboardEvent) => {
      if (e.ctrlKey || e.metaKey || e.altKey || !draft) return
      const inInput = e.target instanceof HTMLInputElement && e.target.type !== 'checkbox'
      if (e.key === 'Enter') {
        e.preventDefault()
        primary()
        return
      }
      if (e.key === 'Escape') {
        ;(document.activeElement as HTMLElement | null)?.blur()
        return
      }
      if (inInput) return

      const key = e.key.toLowerCase()
      const iconIdx = Number(e.key) - 1
      if (iconIdx >= 0 && iconIdx < ICONS.length) {
        const icon = ICONS[iconIdx]
        setDraft(setIcon(draft, icon, !draft[icon]))
      } else if (key === 'd') act('too_dark')
      else if (key === 'o') act('off')
      else if (key === 't') {
        e.preventDefault()
        setDraft({ ...draft, screen_state: 'ok' })
        setTimeout(() => tempRef.current?.select())
      } else if (key === 'j' || e.key === 'ArrowDown') move(1)
      else if (key === 'k' || e.key === 'ArrowUp') move(-1)
      else if (e.key === 'Backspace') resetRow()
      else if (key === 'b') setBright((v) => !v)
      else if (key === 'z') setZoom((v) => !v)
      else return
      e.preventDefault()
    }
    window.addEventListener('keydown', onKey)
    return () => window.removeEventListener('keydown', onKey)
  }, [draft, primary, act, move, resetRow])

  if (!rows) return <div className="loading">{error ?? 'Loading…'}</div>
  if (!row || !draft) return <div className="loading">No rows in the labels CSV</div>

  return (
    <div className="app">
      <FileList
        rows={rows}
        selected={selected}
        filter={filter}
        onFilter={setFilter}
        onSelect={setSelected}
      />
      <ImageView row={row} bright={bright} zoom={zoom} />
      <ReviewForm
        row={row}
        draft={draft}
        onDraft={setDraft}
        onAction={act}
        onReset={resetRow}
        tempRef={tempRef}
        busy={busy}
        error={error}
      />
    </div>
  )
}
