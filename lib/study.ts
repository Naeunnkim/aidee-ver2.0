export type StageKey =
  | 'step_0_start'
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
    key: 'step_0_start',
    label: '프로젝트 시작 공통 확인 구간',
    sidebarLabel: '프로젝트 시작 공통 확인 구간',
    sidebarIndex: 0,
  },
  {
    key: 'step_1_idea',
    label: '개발 조건 정리',
    sidebarLabel: '개발 조건 정리',
    sidebarIndex: 1,
  },
  {
    key: 'step_2_persona',
    label: '페르소나 정리',
    sidebarLabel: '사용자 명확화',
    sidebarIndex: 2,
  },
  {
    key: 'step_2_research',
    label: '리서치',
    sidebarLabel: '사용자 명확화',
    sidebarIndex: 2,
  },
  {
    key: 'step_3_direction',
    label: '디자인·기능 방향 정리',
    sidebarLabel: '개발 방향성 도출',
    sidebarIndex: 3,
  },
  {
    key: 'step_4_definition',
    label: '제품 정의 및 제안 범위 확정',
    sidebarLabel: '스타일 컨셉 도출',
    sidebarIndex: 4,
  },
  {
    key: 'step_4_style',
    label: '스타일 컨셉 도출',
    sidebarLabel: '스타일 컨셉 도출',
    sidebarIndex: 4,
  },
  {
    key: 'step_5_design',
    label: '디자인 시안 확정',
    sidebarLabel: '디자인 시안 확정',
    sidebarIndex: 5,
  },
  {
    key: 'step_5_rfp',
    label: '프로젝트 기획안 생성',
    sidebarLabel: '프로젝트 기획안 생성',
    sidebarIndex: 6,
  },
  {
    key: 'step_6_rfp',
    label: '프로젝트 기획안 생성',
    sidebarLabel: '프로젝트 기획안 생성',
    sidebarIndex: 6,
  },
  {
    key: 'step_6_company',
    label: '협력 파트너 매칭',
    sidebarLabel: '협력 파트너 매칭',
    sidebarIndex: 7,
  },
]

export const SIDEBAR_STEPS = [
  '개발 조건 정리',
  '사용자 명확화',
  '개발 방향성 도출',
  '스타일 컨셉 도출',
  '디자인 시안 확정',
  '프로젝트 기획안 생성',
  '협력 파트너 매칭',
]

const NEXT_STAGE_KEY_MAP: Record<StageKey, StageKey | null> = {
  step_0_start: 'step_1_idea',
  step_1_idea: 'step_2_persona',
  step_2_persona: 'step_3_direction',
  step_2_research: 'step_3_direction',
  step_3_direction: 'step_4_style',
  step_4_style: 'step_5_design',
  step_5_design: 'step_6_rfp',
  step_6_rfp: 'step_6_company',
  step_6_company: null,
  step_4_definition: 'step_4_style',
  step_5_rfp: 'step_6_company',
}

export const PROCESS_STEPS: Array<{
  index: number
  title: string
  description: string
  stageKeys: StageKey[]
}> = [
  {
    index: 1,
    title: '개발 조건 정리',
    description: '제품 목표, 조건, 제약을 실행 가능한 기획 기준으로 정리합니다.',
    stageKeys: ['step_1_idea'],
  },
  {
    index: 2,
    title: '사용자 명확화',
    description: '이 제품을 누가, 언제, 왜 사용할지 구체화합니다.',
    stageKeys: ['step_2_persona', 'step_2_research'],
  },
  {
    index: 3,
    title: '개발 방향성 도출',
    description: '가치 우선순위와 기능·개발 방향을 함께 좁힙니다.',
    stageKeys: ['step_3_direction'],
  },
  {
    index: 4,
    title: '스타일 컨셉 도출',
    description: '형태, 색감, 재질의 시각적 방향을 정합니다.',
    stageKeys: ['step_4_style', 'step_4_definition'],
  },
  {
    index: 5,
    title: '디자인 시안 확정',
    description: '선택한 방향을 기준으로 제품 디자인 시안을 제안합니다.',
    stageKeys: ['step_5_design'],
  },
  {
    index: 6,
    title: '프로젝트 기획안 생성',
    description: '확정된 내용을 평가하고 제품개발 기획안으로 문서화합니다.',
    stageKeys: ['step_6_rfp', 'step_5_rfp'],
  },
  {
    index: 7,
    title: '협력 파트너 매칭',
    description: '제품개발 기획안을 기준으로 협력 파트너 매칭 방향을 정리합니다.',
    stageKeys: ['step_6_company'],
  },
]

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

export function getProcessStepForStage(stageKey: StageKey) {
  return (
    PROCESS_STEPS.find((step) => step.stageKeys.includes(stageKey)) ??
    PROCESS_STEPS[0]
  )
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
