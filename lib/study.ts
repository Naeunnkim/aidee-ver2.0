export type StageKey =
  | 'step_1_idea'
  | 'step_2_persona'
  | 'step_2_research'
  | 'step_3_direction'
  | 'step_4_style'
  | 'step_5_design'
  | 'step_6_rfp'
  | 'step_6_company'
  // Legacy keys kept so existing in-progress sessions do not break.
  | 'step_4_definition'
  | 'step_5_rfp'

export const STAGE_DEFINITIONS: Array<{
  key: StageKey
  label: string
  sidebarLabel: string
  sidebarIndex: number
}> = [
  {
    key: 'step_1_idea',
    label: '제품 아이디어 & 개발 조건 정리',
    sidebarLabel: '제품 아이디어&개발 조건 정리',
    sidebarIndex: 0,
  },
  {
    key: 'step_2_persona',
    label: '페르소나 정리',
    sidebarLabel: '사용자 명확화',
    sidebarIndex: 1,
  },
  {
    key: 'step_2_research',
    label: '리서치',
    sidebarLabel: '사용자 명확화',
    sidebarIndex: 1,
  },
  {
    key: 'step_3_direction',
    label: '디자인·기능 방향 정리',
    sidebarLabel: '디자인·개발 방향성 도출',
    sidebarIndex: 2,
  },
  {
    key: 'step_4_definition',
    label: '제품 정의 및 제안 범위 확정',
    sidebarLabel: '스타일 컨셉 도출',
    sidebarIndex: 3,
  },
  {
    key: 'step_4_style',
    label: '스타일 컨셉 도출',
    sidebarLabel: '스타일 컨셉 도출',
    sidebarIndex: 3,
  },
  {
    key: 'step_5_design',
    label: '디자인 제안',
    sidebarLabel: '디자인 제안',
    sidebarIndex: 4,
  },
  {
    key: 'step_5_rfp',
    label: 'RFP 문서 생성',
    sidebarLabel: '평가 및 RFP 문서 생성',
    sidebarIndex: 5,
  },
  {
    key: 'step_6_rfp',
    label: '평가 및 RFP 문서 생성',
    sidebarLabel: '평가 및 RFP 문서 생성',
    sidebarIndex: 5,
  },
  {
    key: 'step_6_company',
    label: '협력업체 연결',
    sidebarLabel: '협력업체 연결',
    sidebarIndex: 6,
  },
]

export const SIDEBAR_STEPS = [
  '제품 아이디어&개발 조건 정리',
  '사용자 명확화',
  '디자인·개발 방향성 도출',
  '스타일 컨셉 도출',
  '디자인 제안',
  '평가 및 RFP 문서 생성',
  '협력업체 연결',
]

const NEXT_STAGE_KEY_MAP: Record<StageKey, StageKey | null> = {
  step_1_idea: 'step_2_persona',
  step_2_persona: 'step_2_research',
  step_2_research: 'step_3_direction',
  step_3_direction: 'step_4_style',
  step_4_style: 'step_5_design',
  step_5_design: 'step_6_rfp',
  step_6_rfp: 'step_6_company',
  step_6_company: null,
  step_4_definition: 'step_4_style',
  step_5_rfp: 'step_6_company',
}

export function getSidebarStepIndex(stageKey: string | null | undefined) {
  return (
    STAGE_DEFINITIONS.find((stage) => stage.key === stageKey)?.sidebarIndex ?? 0
  )
}

export function isKnownStageKey(value: string): value is StageKey {
  return STAGE_DEFINITIONS.some((stage) => stage.key === value)
}

export function getNextStageKey(stageKey: StageKey) {
  return NEXT_STAGE_KEY_MAP[stageKey] ?? null
}

export function isSameOrNextStage(
  currentStageKey: StageKey,
  targetStageKey: StageKey
) {
  return (
    currentStageKey === targetStageKey ||
    getNextStageKey(currentStageKey) === targetStageKey
  )
}

export function canRequestRfpStage(currentStageKey: StageKey) {
  return (
    currentStageKey === 'step_5_design' ||
    currentStageKey === 'step_6_rfp' ||
    currentStageKey === 'step_5_rfp'
  )
}

export function canRequestCompanyStage(currentStageKey: StageKey) {
  return (
    currentStageKey === 'step_6_rfp' ||
    currentStageKey === 'step_6_company' ||
    currentStageKey === 'step_5_rfp'
  )
}

export function getStageKeysForSidebarIndex(sidebarIndex: number) {
  return STAGE_DEFINITIONS.filter(
    (stage) => stage.sidebarIndex === sidebarIndex
  ).map((stage) => stage.key)
}
