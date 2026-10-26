import { useEffect, useRef } from 'react'
import type { Row } from './types'

export type Filter = 'all' | 'pending' | 'done' | 'flagged'

const FILTERS: Filter[] = ['all', 'pending', 'done', 'flagged']

export function matchesFilter(row: Row, filter: Filter) {
  switch (filter) {
    case 'pending':
      return row.status === 'pending'
    case 'done':
      return row.status !== 'pending'
    case 'flagged':
      return row.retrain_needed !== ''
    default:
      return true
  }
}

interface Props {
  rows: Row[]
  selected: number
  filter: Filter
  onFilter: (f: Filter) => void
  onSelect: (idx: number) => void
}

export default function FileList({ rows, selected, filter, onFilter, onSelect }: Props) {
  const selectedRef = useRef<HTMLLIElement>(null)
  const reviewed = rows.filter((r) => r.status !== 'pending').length
  const visible = rows.filter((r) => matchesFilter(r, filter))

  useEffect(() => {
    selectedRef.current?.scrollIntoView({ block: 'nearest' })
  }, [selected, filter])

  return (
    <nav className="file-list">
      <header>
        <div className="progress">
          <strong>{reviewed}</strong> / {rows.length} reviewed
          <div className="progress-bar">
            <div style={{ width: `${rows.length ? (100 * reviewed) / rows.length : 0}%` }} />
          </div>
        </div>
        <div className="filters" role="tablist">
          {FILTERS.map((f) => (
            <button
              key={f}
              role="tab"
              aria-selected={filter === f}
              className={filter === f ? 'active' : ''}
              onClick={() => onFilter(f)}
            >
              {f}
            </button>
          ))}
        </div>
      </header>
      <ul>
        {visible.map((row) => (
          <li
            key={row.idx}
            ref={row.idx === selected ? selectedRef : undefined}
            className={row.idx === selected ? 'selected' : ''}
            onClick={() => onSelect(row.idx)}
            title={row.retrain_needed || row.filename}
          >
            <span className={`dot status-${row.status}`} aria-label={row.status} />
            <span className="idx">{row.idx + 1}</span>
            <span className="name">{row.filename.replace(/^cam\d+-snapshot-/, '')}</span>
            {!row.image_exists && <span className="missing">no image</span>}
          </li>
        ))}
        {visible.length === 0 && <li className="empty">Nothing here</li>}
      </ul>
    </nav>
  )
}
