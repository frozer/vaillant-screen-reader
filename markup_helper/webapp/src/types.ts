export type ScreenState = 'ok' | 'off' | 'unreadable'
export type Status = 'pending' | 'confirmed' | 'too_dark' | 'off' | 'updated'
export type Action = 'confirm' | 'too_dark' | 'off' | 'update'

export const ICONS = ['burn', 'heating', 'hotwater', 'pump', 'gasvalve'] as const
export type Icon = (typeof ICONS)[number]

export const ICON_LABELS: Record<Icon, string> = {
  burn: 'Burn',
  heating: 'Heating',
  hotwater: 'Hot water',
  pump: 'Pump',
  gasvalve: 'Gas valve',
}

export type Reading = { temperature: number | null; screen_state: ScreenState } & Record<
  Icon,
  boolean | null
>

export interface Row {
  idx: number
  filename: string
  source_dir: string
  status: Status
  recognized: Reading
  current: Reading
  too_dark: boolean
  confirmed: boolean
  retrain_needed: string
  image_exists: boolean
}

export type Values = { temperature: number | null } & Record<Icon, boolean>

/** Form state: temperature kept as text so partial input is possible. */
export type Draft = { temperature: string; screen_state: ScreenState } & Record<Icon, boolean>
