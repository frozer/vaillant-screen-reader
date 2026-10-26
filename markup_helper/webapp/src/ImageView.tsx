import { imageUrl } from './api'
import type { Row } from './types'

interface Props {
  row: Row
  bright: boolean
  zoom: boolean
}

export default function ImageView({ row, bright, zoom }: Props) {
  return (
    <main className={`image-view ${zoom ? 'zoom' : 'fit'}`}>
      {row.image_exists ? (
        <img
          src={imageUrl(row.idx)}
          alt={row.filename}
          style={bright ? { filter: 'brightness(2.5) contrast(1.2)' } : undefined}
        />
      ) : (
        <div className="placeholder">Image not found in source dir</div>
      )}
      <footer>
        <span>{row.filename}</span>
        <span className="hint">
          B brightness {bright ? 'on' : 'off'} · Z {zoom ? '1:1' : 'fit'}
        </span>
      </footer>
    </main>
  )
}
