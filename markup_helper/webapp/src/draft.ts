import { ICONS, type Draft, type Icon, type Reading, type Values } from './types'

export function draftFromReading(r: Reading): Draft {
  const draft = {
    temperature: r.temperature === null ? '' : String(r.temperature),
    screen_state: r.screen_state,
  } as Draft
  for (const icon of ICONS) draft[icon] = r[icon] ?? false
  return draft
}

export const temperatureValid = (t: string) => /^\d{0,2}$/.test(t.trim())

export function draftToValues(d: Draft): Values {
  const t = d.temperature.trim()
  const values = { temperature: t === '' ? null : Number(t) } as Values
  for (const icon of ICONS) values[icon] = d[icon]
  return values
}

/** Fields of an 'ok' draft that differ from the recognized reading. */
export function changedFields(d: Draft, rec: Reading): Set<'temperature' | Icon> {
  const changed = new Set<'temperature' | Icon>()
  if (d.screen_state !== 'ok') return changed
  const t = d.temperature.trim()
  if ((t === '' ? null : Number(t)) !== rec.temperature) changed.add('temperature')
  for (const icon of ICONS) if (d[icon] !== rec[icon]) changed.add(icon)
  return changed
}

export function isDirty(d: Draft, rec: Reading): boolean {
  return d.screen_state !== rec.screen_state || changedFields(d, rec).size > 0
}

/**
 * Toggle an icon. Burn and GasValve are coupled: switching either one on switches
 * the other on too. Editing a value means the screen is readable.
 */
export function setIcon(d: Draft, icon: Icon, on: boolean): Draft {
  const next = { ...d, [icon]: on, screen_state: 'ok' as const }
  if (on && icon === 'burn') next.gasvalve = true
  if (on && icon === 'gasvalve') next.burn = true
  return next
}

export const burnGasMismatch = (r: Reading) =>
  r.screen_state === 'ok' && r.burn !== null && r.gasvalve !== null && r.burn !== r.gasvalve
