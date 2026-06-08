import { createGoogleGenerativeAI } from '@ai-sdk/google'
import { generateObject, generateText, tool, type ModelMessage } from 'ai'
import { createClient } from '@supabase/supabase-js'
import { z } from 'zod'

import {
  appendGeneratedImagesBlock,
  type GeneratedImageBlock,
  generateNanoBananaImages,
} from '@/lib/image-generation'
import {
  getExpertDefinition,
  getExpertPrompt,
  isExpertKey,
  type ExpertKey,
} from '@/lib/experts'
import { SYSTEM_PROMPT_TEMPLATE } from '@/lib/prompts'
import {
  RFP_DOCUMENT_SCHEMA,
  type RfpDocument,
  buildRfpObjectPrompt,
} from '@/lib/rfp'
import {
  canRequestRfpStage,
  getProcessStepForStage,
  PROCESS_STEPS,
  type StageKey,
  getNextStageKey,
  isKnownStageKey,
  isSameOrNextStage,
} from '@/lib/study'

export const maxDuration = 60

type ChatRequestMessagePart = {
  type: string
  text?: string
}

type ChatRequestMessage = {
  role: string
  content?: string
  parts?: ChatRequestMessagePart[]
}

type ChatRequestBody = {
  messages?: ChatRequestMessage[]
  projectId?: string
  currentStageKey?: StageKey
  activeExpert?: ExpertKey
  expertCall?: boolean
  forceImageGeneration?:
    | 'initial_design'
    | 'design_revision'
    | 'problem_statements_visualization'
    | 'experience_keywords_visualization'
    | 'relationship_keywords_visualization'
    | 'market_size_visualization'
    | 'consumption_keywords_visualization'
    | 'brand_positioning_visualization'
    | 'style_reference_options'
    | 'style_moodboard_visualization'
    | 'persona_visualization'
}

type NormalizedMessage = {
  role: 'user' | 'assistant' | 'system'
  content: string
}

type ProjectRecord = {
  title: string | null
  requirements: Record<string, unknown> | null
}

type ReferenceImageRecord = {
  file_name: string | null
  analysis_text: string | null
  analysis_json: unknown
}

type ReferenceAnalysis = {
  summary?: string
  category?: string
  moodKeywords?: string[]
  colorKeywords?: string[]
  materialKeywords?: string[]
  shapeKeywords?: string[]
  detailPoints?: string[]
  designDirection?: string[]
}

type StageMeta = {
  currentStageKey: StageKey
  nextStageKey: StageKey
  transition: boolean
  reason: string
}

type PersonaFlowArtifactKind =
  | 'problem_statements'
  | 'experience_keywords'
  | 'relationship_keywords'

type DirectionResearchKind =
  | 'market_size'
  | 'consumption_keywords'
  | 'brand_positioning'

type VisualizationRevisionTarget =
  | PersonaFlowArtifactKind
  | 'persona'
  | DirectionResearchKind
  | 'style_reference'

const DEFAULT_STAGE_KEY: StageKey = 'step_0_start'

function isStringArray(value: unknown): value is string[] {
  return Array.isArray(value) && value.every((item) => typeof item === 'string')
}

function isImageGenerationRequest(text: string) {
  const normalized = text.toLowerCase()
  return [
    '이미지',
    '시안',
    '렌더',
    '무드보드',
    '비주얼',
    'visual',
    'render',
    'concept image',
    'image',
    '생성',
    '그려',
    '보여줘',
    '만들어줘',
  ].some((keyword) => normalized.includes(keyword))
}

function hasNanoBananaPlaceholder(text: string) {
  return (
    /\[\s*\/?\s*Nano Banana\s*이미지\s*생성\s*요청\s*\]/i.test(text) ||
    /Generating\s+[1-4]\s+images?\s+based\s+on/i.test(text) ||
    /잠시\s*후\s*이미지가\s*생성됩니다/i.test(text) ||
    /잠시(?:만)?\s*(?:기다려|기다려\s*주세요|기다려주세요)/i.test(text) ||
    /곧\s*(?:이미지|시안|렌더).*(?:생성|준비)/i.test(text) ||
    /\(?\s*이미지\s*[1-4]\s*placeholder\s*\)?/i.test(text) ||
    /\(?\s*image\s*[1-4]\s*placeholder\s*\)?/i.test(text) ||
    /placeholder/i.test(text)
  )
}

function getStyleReferenceIntro(imageCount = 3) {
  const countLabel = imageCount === 1 ? '1장' : `${imageCount}장`

  return [
    '이제 원하는 스타일을 구체화하는 시간입니다.',
    '앞서 정리한 가치 키워드와 우선순위를 시각적으로 풀어냈습니다.',
    `아래 스타일 레퍼런스 ${countLabel}을 확인하고, 원하는 방향을 하나 선택해주세요.`,
  ].join('\n')
}

function canGenerateImagesInStage(stageKey: StageKey) {
  return stageKey === 'step_4_style' || stageKey === 'step_5_design'
}

function extractRequestedImageCount(text: string) {
  const digitMatch = text.match(/([1-4])\s*장/)
  if (digitMatch) {
    return Number(digitMatch[1])
  }

  if (text.includes('두 장') || text.includes('2개')) return 2
  if (text.includes('세 장') || text.includes('3개')) return 3
  if (text.includes('네 장') || text.includes('4개')) return 4
  return 1
}

function extractReferenceAnalysis(
  analysisJson: unknown
): ReferenceAnalysis | null {
  if (!analysisJson || typeof analysisJson !== 'object') {
    return null
  }

  const source =
    'analysis' in analysisJson &&
    analysisJson.analysis &&
    typeof analysisJson.analysis === 'object'
      ? analysisJson.analysis
      : analysisJson

  if (!source || typeof source !== 'object') {
    return null
  }

  const candidate = source as Record<string, unknown>

  return {
    summary:
      typeof candidate.summary === 'string' ? candidate.summary : undefined,
    category:
      typeof candidate.category === 'string' ? candidate.category : undefined,
    moodKeywords: isStringArray(candidate.moodKeywords)
      ? candidate.moodKeywords
      : undefined,
    colorKeywords: isStringArray(candidate.colorKeywords)
      ? candidate.colorKeywords
      : undefined,
    materialKeywords: isStringArray(candidate.materialKeywords)
      ? candidate.materialKeywords
      : undefined,
    shapeKeywords: isStringArray(candidate.shapeKeywords)
      ? candidate.shapeKeywords
      : undefined,
    detailPoints: isStringArray(candidate.detailPoints)
      ? candidate.detailPoints
      : undefined,
    designDirection: isStringArray(candidate.designDirection)
      ? candidate.designDirection
      : undefined,
  }
}

function buildReferenceGuidelineBlock(referenceImages: ReferenceImageRecord[]) {
  const parsedAnalyses = referenceImages
    .map((item) => ({
      fileName: item.file_name || '이름 없음',
      analysis: extractReferenceAnalysis(item.analysis_json),
    }))
    .filter(
      (
        item
      ): item is {
        fileName: string
        analysis: ReferenceAnalysis
      } => Boolean(item.analysis)
    )

  if (parsedAnalyses.length === 0) {
    return '레퍼런스 기반 가이드라인 없음'
  }

  return parsedAnalyses
    .map(({ fileName, analysis }, index) => {
      const lines = [
        `가이드라인 ${index + 1} (${fileName})`,
        analysis.summary ? `- 요약: ${analysis.summary}` : null,
        analysis.category ? `- 유형: ${analysis.category}` : null,
        analysis.moodKeywords?.length
          ? `- 무드 키워드: ${analysis.moodKeywords.join(', ')}`
          : null,
        analysis.colorKeywords?.length
          ? `- 색상 키워드: ${analysis.colorKeywords.join(', ')}`
          : null,
        analysis.materialKeywords?.length
          ? `- 재질 키워드: ${analysis.materialKeywords.join(', ')}`
          : null,
        analysis.shapeKeywords?.length
          ? `- 형태 키워드: ${analysis.shapeKeywords.join(', ')}`
          : null,
        analysis.detailPoints?.length
          ? `- 참고 디테일: ${analysis.detailPoints.join(' / ')}`
          : null,
        analysis.designDirection?.length
          ? `- 디자인 방향: ${analysis.designDirection.join(' / ')}`
          : null,
      ]

      return lines.filter(Boolean).join('\n')
    })
    .join('\n\n')
}

function normalizeMessages(rawMessages: ChatRequestMessage[]): NormalizedMessage[] {
  return rawMessages
    .filter((message) => message && typeof message.role === 'string')
    .map((message) => {
      if (typeof message.content === 'string') {
        return {
          role: message.role,
          content: message.content,
        }
      }

      if (Array.isArray(message.parts)) {
        const text = message.parts
          .filter(
            (part): part is ChatRequestMessagePart =>
              Boolean(part) &&
              part.type === 'text' &&
              typeof part.text === 'string'
          )
          .map((part) => part.text)
          .join('\n')

        return {
          role: message.role,
          content: text,
        }
      }

      return {
        role: message.role,
        content: '',
      }
    })
    .filter(
      (message): message is NormalizedMessage =>
        ['user', 'assistant', 'system'].includes(message.role)
    )
}

function buildProjectContext(
  project: ProjectRecord | null,
  referenceImages: ReferenceImageRecord[]
) {
  const title = project?.title || '제목 없음'
  const requirements = JSON.stringify(project?.requirements ?? {}, null, 2)
  const referenceContext = buildReferenceContext(referenceImages)
  const referenceGuidelines = buildReferenceGuidelineBlock(referenceImages)

  return `
[프로젝트 정보]
- 프로젝트명: ${title}
- requirements:
${requirements}

[레퍼런스 이미지 분석]
${referenceContext}

[레퍼런스 기반 가이드라인]
${referenceGuidelines}
`.trim()
}

function buildRequirementsText(project: ProjectRecord | null) {
  return JSON.stringify(project?.requirements ?? {}, null, 2)
}

function getRequirements(project: ProjectRecord | null): Record<string, unknown> {
  return project?.requirements && typeof project.requirements === 'object'
    ? project.requirements
    : {}
}

function getRequirementString(
  requirements: Record<string, unknown>,
  key: string,
  fallback = '미정'
) {
  const value = requirements[key]
  return typeof value === 'string' && value.trim() ? value.trim() : fallback
}

function formatRequirementList(
  requirements: Record<string, unknown>,
  key: string,
  otherKey?: string
) {
  const value = requirements[key]
  const otherValue = otherKey ? getRequirementString(requirements, otherKey, '') : ''
  const values = isStringArray(value)
    ? value
        .map((item) =>
          item === '기타 (직접 입력)' && otherValue ? otherValue : item
        )
        .filter((item) => item !== '기타 (직접 입력)')
    : []

  return values.length > 0 ? values.join(', ') : otherValue || '미정'
}

function formatBudgetValue(value: unknown) {
  if (typeof value !== 'number' || Number.isNaN(value)) {
    return null
  }

  if (value >= 10000) {
    return '1억원'
  }

  return `${value.toLocaleString('ko-KR')}만원`
}

function formatBudgetRange(requirements: Record<string, unknown>) {
  const minBudget = formatBudgetValue(requirements.minBudget)
  const maxBudget = formatBudgetValue(requirements.maxBudget)

  if (minBudget && maxBudget) {
    return `${minBudget} - ${maxBudget}`
  }

  return minBudget || maxBudget || '미정'
}

function formatBudgetMinimum(requirements: Record<string, unknown>) {
  const minBudget = requirements.minBudget

  if (typeof minBudget !== 'number' || Number.isNaN(minBudget)) {
    return '미정'
  }

  return `$${Math.round(minBudget / 100)}K+`
}

function formatTimeline(duration: string) {
  const normalized = duration.replace(/\s+/g, '')
  const yearMatch = normalized.match(/(\d+(?:\.\d+)?)년/)
  const monthMatch = normalized.match(/(\d+(?:\.\d+)?)(?:개월|달|months?|mo)/i)
  const weekMatch = normalized.match(/(\d+(?:\.\d+)?)(?:주|weeks?|w)/i)

  if (yearMatch) {
    const months = Number(yearMatch[1]) * 12
    return `${months}${normalized.includes('+') ? '+' : ''} Months`
  }

  if (monthMatch) {
    const months = Number(monthMatch[1])
    return `${months}${normalized.includes('+') ? '+' : ''} ${
      months === 1 ? 'Month' : 'Months'
    }`
  }

  if (weekMatch) {
    const weeks = Number(weekMatch[1])
    return `${weeks}${normalized.includes('+') ? '+' : ''} ${
      weeks === 1 ? 'Week' : 'Weeks'
    }`
  }

  return duration || '미정'
}

function cleanSingleLineText(text: string) {
  return text.replace(/\s+/g, ' ').trim()
}

function limitKoreanWords(text: string, maxWords: number) {
  const words = text.split(/\s+/).filter(Boolean)

  if (words.length <= maxWords) {
    return text
  }

  return `${words.slice(0, maxWords).join(' ')}...`
}

function formatIdeaSummarySentence(text: string) {
  const summary = limitKoreanWords(cleanSingleLineText(text), 100)

  if (!summary) {
    return ''
  }

  if (/[.!?。！？]$/.test(summary)) {
    return summary
  }

  if (/(다|요|니다|습니다)$/.test(summary)) {
    return `${summary}.`
  }

  return `${summary}가 목표입니다.`
}

function summarizeIdeaText(requirements: Record<string, unknown>) {
  const idea = cleanSingleLineText(getRequirementString(requirements, 'idea', ''))

  if (!idea) {
    return '아직 아이디어 텍스트가 충분히 입력되지 않았습니다.'
  }

  return formatIdeaSummarySentence(idea)
}

function buildProjectStartSnapshot(project: ProjectRecord | null) {
  const requirements = getRequirements(project)
  const goal = getRequirementString(requirements, 'goal')
  const category = formatRequirementList(requirements, 'categories', 'otherCategory')
  const budgetRange = formatBudgetRange(requirements)
  const duration = formatTimeline(getRequirementString(requirements, 'duration'))
  const budgetAndDuration =
    budgetRange === '미정' && duration === '미정'
      ? '미정'
      : `${budgetRange} / ${duration}`

  return {
    title: project?.title || '새 프로젝트',
    goal,
    category,
    budgetMinimum: formatBudgetMinimum(requirements),
    budgetRange,
    duration,
    budgetAndDuration,
    size: getRequirementString(requirements, 'size'),
    features: formatRequirementList(requirements, 'features', 'otherFeature'),
    usage: getRequirementString(requirements, 'usage'),
    ideaSummary: summarizeIdeaText(requirements),
  }
}

type ProjectHintContext = ReturnType<typeof buildProjectHintContext>

function getFirstListValue(value: string) {
  return value
    .split(/[,，/]|(?:\s*·\s*)|(?:\s+및\s+)|(?:\s+그리고\s+)/)
    .map((item) => cleanSingleLineText(item))
    .find((item) => item && item !== '미정')
}

function compactHintText(text: string, fallback: string, maxLength = 34) {
  const cleaned = cleanSingleLineText(text)

  if (!cleaned || cleaned === '미정') {
    return fallback
  }

  if (cleaned.length <= maxLength) {
    return cleaned
  }

  return `${cleaned.slice(0, maxLength).trim()}...`
}

function getHintRequirementText(
  requirements: Record<string, unknown>,
  key: string
) {
  const value = requirements[key]
  return typeof value === 'string' ? value.trim() : ''
}

function inferProjectSpaceLabel(text: string, productLabel: string) {
  if (/책상|데스크|스터디|공부|학습|업무|사무|오피스/i.test(text)) {
    return '책상과 작업 공간'
  }

  if (/주방|요리|식사|음식|식품|카페/i.test(text)) {
    return '주방과 식사 공간'
  }

  if (/침실|수면|휴식|잠|리빙|거실/i.test(text)) {
    return '생활과 휴식 공간'
  }

  if (/운동|피트니스|헬스|러닝|야외|이동|휴대/i.test(text)) {
    return '이동과 활동 공간'
  }

  if (/욕실|세면|샤워|위생|뷰티/i.test(text)) {
    return '관리와 위생 공간'
  }

  return `${productLabel} 사용 공간`
}

function inferProjectSituationLabel(text: string, productLabel: string) {
  if (/공부|학습|시험|과제|자격|집중|몰입/i.test(text)) {
    return '집중이 필요한 순간'
  }

  if (/수면|잠|침실|휴식|회복|긴장|스트레스/i.test(text)) {
    return '휴식과 회복이 필요한 순간'
  }

  if (/요리|식사|주방|음식|보관|정리/i.test(text)) {
    return '준비와 정리가 반복되는 순간'
  }

  if (/운동|피트니스|러닝|이동|야외|휴대/i.test(text)) {
    return '이동하거나 활동하는 순간'
  }

  if (/위생|욕실|세면|샤워|뷰티|관리/i.test(text)) {
    return '개인 관리가 필요한 순간'
  }

  return `${productLabel} 필요성이 생기는 순간`
}

function inferProjectPainLabel(text: string) {
  if (/집중|몰입|방해|알림|산만|흐름/i.test(text)) {
    return '흐름이 끊기는 불편'
  }

  if (/정리|보관|찾기|수납|공간/i.test(text)) {
    return '정리와 보관이 번거로운 불편'
  }

  if (/시간|루틴|습관|반복|일정/i.test(text)) {
    return '루틴을 유지하기 어려운 불편'
  }

  if (/휴대|이동|무게|작다|소형|손바닥/i.test(text)) {
    return '이동 중 사용이 불편한 지점'
  }

  if (/위생|세척|청소|관리/i.test(text)) {
    return '관리와 위생을 유지하기 어려운 지점'
  }

  return '현재 방식에서 반복되는 불편'
}

function buildProjectHintContext(project: ProjectRecord | null) {
  const requirements = getRequirements(project)
  const snapshot = buildProjectStartSnapshot(project)
  const categoryValue = getFirstListValue(snapshot.category)
  const featureValue = getFirstListValue(snapshot.features)
  const rawIdea = getHintRequirementText(requirements, 'idea')
  const productLabel = compactHintText(
    categoryValue || snapshot.title,
    '이 제품',
    18
  )
  const ideaLabel = compactHintText(
    rawIdea || snapshot.ideaSummary,
    `${productLabel} 아이디어`,
    28
  )
  const featureLabel = compactHintText(
    featureValue || snapshot.features,
    '핵심 기능',
    18
  )
  const projectText = [
    snapshot.title,
    snapshot.category,
    snapshot.features,
    rawIdea,
  ].join(' ')

  return {
    productLabel,
    ideaLabel,
    featureLabel,
    spaceLabel: inferProjectSpaceLabel(projectText, productLabel),
    situationLabel: inferProjectSituationLabel(projectText, productLabel),
    painLabel: inferProjectPainLabel(projectText),
  }
}

function buildContextualHintChoices(
  context: ProjectHintContext,
  kind:
    | 'persona_user'
    | 'persona_usage'
    | 'persona_pain'
    | 'persona_decision'
    | 'problem_scene'
    | 'problem_action'
    | 'problem_interruption'
    | 'problem_workaround'
    | 'problem_residue'
    | 'problem_change'
    | 'experience_emotion_entry'
    | 'experience_emotion_relief'
    | 'experience_emotion_after'
    | 'experience_behavior_start'
    | 'experience_behavior_recovery'
    | 'experience_behavior_routine'
    | 'experience_space_impression'
    | 'experience_space_meaning'
    | 'experience_space_relation'
    | 'relationship_interruption'
    | 'relationship_space'
    | 'relationship_time'
    | 'problem_revision'
): [string, string, string] {
  const {
    productLabel,
    featureLabel,
    ideaLabel,
    spaceLabel,
    situationLabel,
    painLabel,
  } = context
  const productTerm = `‘${productLabel}’`
  const featureTerm = `‘${featureLabel}’`

  switch (kind) {
    case 'persona_user':
      return [
        `사용자 단서: ${situationLabel}을 자주 겪는 사람`,
        `행동 단서: ${featureTerm}이 반복적으로 필요한 사람`,
        `맥락 단서: ${spaceLabel}에서 문제를 느끼는 사람`,
      ]
    case 'persona_usage':
      return [
        `상황 단서: ${situationLabel}이 시작되는 때`,
        `기능 단서: ${featureTerm}이 실제로 필요한 때`,
        `공간 단서: ${spaceLabel}에서 불편이 드러나는 때`,
      ]
    case 'persona_pain':
      return [
        `문제 단서: ${painLabel}`,
        `기능 단서: ${featureTerm} 없이 생기는 번거로움`,
        `공간 단서: ${spaceLabel}에서 반복되는 불편`,
      ]
    case 'persona_decision':
      return [
        `효과 단서: ${featureTerm}이 실제로 도움이 되는지`,
        `사용 단서: ${productTerm} 지속 사용이 쉬운지`,
        `공간 단서: ${spaceLabel}에 자연스럽게 맞는지`,
      ]
    case 'problem_scene':
      return [
        `상황 단서: ${situationLabel}이 생기는 장면`,
        `공간 단서: ${spaceLabel}에서 불편이 드러나는 장면`,
        `아이디어 단서: ${ideaLabel}가 떠오른 배경`,
      ]
    case 'problem_action':
      return [
        `행동 단서: ${situationLabel}에 들어가려던 중`,
        `기능 단서: ${featureTerm}을 통해 해결하려던 중`,
        `기존 방식 단서: ${spaceLabel}에서 버티던 중`,
      ]
    case 'problem_interruption':
      return [
        `방해 단서: ${painLabel}`,
        `기능 단서: ${featureTerm}이 없어 생기는 끊김`,
        `공간 단서: ${spaceLabel}에서 반복되는 자극`,
      ]
    case 'problem_workaround':
      return [
        `대응 단서: ${productTerm} 없이 임시로 해결함`,
        `행동 단서: ${featureTerm} 대신 손으로 반복 처리함`,
        `회피 단서: 불편한 상황을 미루거나 우회함`,
      ]
    case 'problem_residue':
      return [
        `문제 단서: ${painLabel}이 계속 남음`,
        `기능 단서: ${featureTerm}이 없어 흐름이 끊김`,
        `공간 단서: ${spaceLabel}에서 불편한 패턴이 반복됨`,
      ]
    case 'problem_change':
      return [
        `변화 단서: ${productTerm} 사용으로 상황이 더 자연스러워짐`,
        `행동 단서: ${featureTerm} 덕분에 행동이 덜 끊김`,
        `공간 단서: ${spaceLabel}에서 원하는 상태가 유지됨`,
      ]
    case 'experience_emotion_entry':
      return [
        `첫인상 단서: ${productTerm}을 믿고 써볼 수 있음`,
        `아이디어 단서: ${ideaLabel}에서 기대감이 생김`,
        `공간 단서: ${spaceLabel}에 놓였을 때 편안함`,
      ]
    case 'experience_emotion_relief':
      return [
        `해소 단서: ${painLabel}이 줄어드는 안도감`,
        `기능 단서: ${featureTerm}을 통해 번거로움이 줄어듦`,
        `상황 단서: ${situationLabel}이 매끄러워짐`,
      ]
    case 'experience_emotion_after':
      return [
        `사용 후 단서: ${productTerm} 사용 뒤의 만족감`,
        `결과 단서: ${painLabel}이 줄었다는 성취감`,
        `공간 단서: ${spaceLabel}이 정돈된 뒤의 여유감`,
      ]
    case 'experience_behavior_start':
      return [
        `시작 단서: ${situationLabel}을 더 쉽게 시작함`,
        `기능 단서: ${featureTerm}을 자연스럽게 사용함`,
        `루틴 단서: ${productTerm} 사용 행동이 반복됨`,
      ]
    case 'experience_behavior_recovery':
      return [
        `회복 단서: ${painLabel}이 생겨도 다시 돌아옴`,
        `기능 단서: ${featureTerm}을 통해 흐름을 회복함`,
        `공간 단서: ${spaceLabel}에서 행동을 다시 정렬함`,
      ]
    case 'experience_behavior_routine':
      return [
        `루틴 단서: ${productTerm} 사용이 생활 흐름에 들어옴`,
        `상황 단서: ${situationLabel} 전후 행동이 일정해짐`,
        `조절 단서: ${featureTerm}을 기준으로 스스로 조절함`,
      ]
    case 'experience_space_impression':
      return [
        `${spaceLabel}에 자연스럽게 놓이는 인상`,
        `${productTerm} 역할이 바로 보이는 인상`,
        `${ideaLabel}가 분위기로 드러나는 인상`,
      ]
    case 'experience_space_meaning':
      return [
        `${spaceLabel}을 더 안정적으로 느끼게 함`,
        `${productTerm}이 행동의 기준점이 됨`,
        `${situationLabel}을 떠올리게 하는 공간 신호가 됨`,
      ]
    case 'experience_space_relation':
      return [
        `${spaceLabel}의 기존 물건과 충돌하지 않음`,
        `${featureTerm}이 생활 흐름에 자연스럽게 연결됨`,
        `${productTerm}이 주변 행동을 조용히 도와줌`,
      ]
    case 'relationship_interruption':
      return [
        `${painLabel}을 만드는 기존 대안과 거리를 둠`,
        `${featureTerm}이 방해 요소를 줄이는 기준이 됨`,
        `${productTerm}이 사용자의 이탈을 다시 붙잡음`,
      ]
    case 'relationship_space':
      return [
        `${spaceLabel}이 ${productTerm}의 주 사용 맥락이 됨`,
        `${productTerm}이 공간의 분위기를 정돈함`,
        `${featureTerm}이 공간 안 행동을 바꿈`,
      ]
    case 'relationship_time':
      return [
        `${situationLabel}의 시작과 끝을 인식함`,
        `${featureTerm}을 기준으로 시간을 조절함`,
        `${painLabel}을 줄이기 위해 생활 리듬을 다시 잡음`,
      ]
    case 'problem_revision':
      return [
        `${spaceLabel}의 실제 상황을 더 구체화`,
        `${painLabel}을 더 선명하게 정리`,
        `${featureTerm}이 해결해야 할 필요를 재정리`,
      ]
  }
}

function buildProjectCardResponse({
  project,
  referenceImages,
}: {
  project: ProjectRecord | null
  referenceImages: ReferenceImageRecord[]
}) {
  const snapshot = buildProjectStartSnapshot(project)
  const referenceSummary =
    referenceImages.length > 0
      ? `${referenceImages.length}개 참고 이미지가 업로드되어 있습니다.`
      : '업로드된 참고 이미지는 아직 없습니다.'

  return [
    "새로운 프로젝트가 시작되었네요! 'Aidee'팀과 함께 아이디어를 구체화해보아요.",
    '',
    '<<AIDEE_PROJECT_DIRECTION>>',
    'Project Direction',
    '',
    `**프로젝트명**  `,
    snapshot.title,
    '',
    `**제품 카테고리**  `,
    snapshot.category,
    '',
    `**아이디어 정리**  `,
    snapshot.ideaSummary,
    '',
    `**Budget Minimum**  `,
    snapshot.budgetMinimum,
    '',
    `**Target Timeline**  `,
    snapshot.duration,
    '',
    `**Project Scope**  `,
    snapshot.budgetRange,
    '',
    `**Key Features**  `,
    snapshot.features,
    '',
    `**참고 자료**  `,
    referenceSummary,
    '<</AIDEE_PROJECT_DIRECTION>>',
    '',
    '제품의 구체적인 모습이나 추가 설명이 있다면 편하게 알려주세요.',
    '형태, 색감, 재질, 사용 장면, 꼭 들어갔으면 하는 디테일처럼 떠오르는 내용만 적어주셔도 좋아요.',
  ].join('\n')
}

function buildProjectStartSummaryResponse({
  project,
  lastUserMessage,
}: {
  project: ProjectRecord | null
  lastUserMessage: string
}) {
  const snapshot = buildProjectStartSnapshot(project)
  const additionalDescription = cleanSingleLineText(lastUserMessage) || '추가 설명 없음'

  return [
    '## 전체 내용 정리',
    '',
    `- 프로젝트 목표: ${snapshot.goal}`,
    `- 제품 카테고리: ${snapshot.category}`,
    `- 예산/기간 범위: ${snapshot.budgetAndDuration}`,
    `- 최종 활용 목적: ${snapshot.usage}`,
    `- 아이디어 방향: ${snapshot.ideaSummary}`,
    `- 제품 모습/추가 설명: ${additionalDescription}`,
    '',
    '이 내용을 기준점으로 두고 다음 흐름을 확인하면 됩니다.',
    '아래의 프로세스 확인하기 버튼을 눌러 1~7단계 진행 순서를 확인해주세요.',
  ].join('\n')
}

function buildProcessGuideResponse() {
  const processLines = PROCESS_STEPS.flatMap((step) => [
    `${step.index}. ${step.title}`,
    step.description,
    '',
  ])

  return [
    '## 전체 프로세스',
    '',
    ...processLines,
    buildStageTransitionPrompt('step_1_idea'),
  ].join('\n')
}

function buildStageTransitionPrompt(nextStageKey: StageKey) {
  const step = getProcessStepForStage(nextStageKey)

  return [
    `다음으로 STEP ${step.index}. ${step.title} 단계로 넘어가겠습니다.`,
    `이 단계에서는 ${step.description}`,
    '진행할까요?',
  ].join('\n')
}

function appendStageTransitionPrompt(text: string, nextStageKey: StageKey) {
  const prompt = buildStageTransitionPrompt(nextStageKey)

  if (text.includes(prompt) || /진행할까요\?\s*$/.test(text.trim())) {
    return text.trim()
  }

  return [text.trim(), prompt].filter(Boolean).join('\n\n')
}

function hasProjectCardMessage(messages: NormalizedMessage[]) {
  return messages.some(
    (message) =>
      message.role === 'assistant' &&
      (/<<AIDEE_PROJECT_DIRECTION>>/i.test(message.content) ||
        /#\s*Project\s*(?:Card|Direction)/i.test(message.content))
  )
}

function hasProjectStartSummaryMessage(messages: NormalizedMessage[]) {
  return messages.some(
    (message) =>
      message.role === 'assistant' && message.content.includes('## 전체 내용 정리')
  )
}

function hasProcessGuideMessage(messages: NormalizedMessage[]) {
  return messages.some(
    (message) =>
      message.role === 'assistant' && message.content.includes('## 전체 프로세스')
  )
}

function isProcessConfirmationRequest(text: string) {
  return /프로세스\s*확인|process\s*confirm|process\s*check/i.test(text)
}

function buildReferenceContext(referenceImages: ReferenceImageRecord[]) {
  return referenceImages.length > 0
    ? referenceImages
        .map((item, index) => {
          const analysisText =
            typeof item.analysis_text === 'string' ? item.analysis_text : ''
          const analysisJson = item.analysis_json
            ? JSON.stringify(item.analysis_json)
            : ''

          return [
            `레퍼런스 이미지 ${index + 1}: ${item.file_name || '이름 없음'}`,
            analysisText ? `- 분석 텍스트: ${analysisText}` : null,
            analysisJson ? `- 분석 JSON: ${analysisJson}` : null,
          ]
            .filter(Boolean)
            .join('\n')
        })
        .join('\n\n')
    : '레퍼런스 이미지 분석 결과 없음'
}

function truncateText(text: string, maxLength: number) {
  if (text.length <= maxLength) {
    return text
  }

  return `${text.slice(0, maxLength)}\n...[truncated]`
}

function buildConversationText(messages: ModelMessage[]) {
  return messages
    .map((message) => {
      if (typeof message.content === 'string') {
        return `[${message.role}] ${message.content}`
      }

      const text = Array.isArray(message.content)
        ? message.content
            .map((part) => ('text' in part && typeof part.text === 'string' ? part.text : ''))
            .filter(Boolean)
            .join('\n')
        : ''

      return `[${message.role}] ${text}`
    })
    .join('\n\n')
}

function hasGeneratedDesignImagesInMessages(messages: ModelMessage[]) {
  const conversation = buildConversationText(messages)

  return (
    /design\s*이미지\s*\d+장/i.test(conversation) ||
    /"purpose"\s*:\s*"design"/i.test(conversation) ||
    /STEP5\s*초기\s*디자인\s*(?:3|4)안\s*세트가\s*이미\s*제시/i.test(conversation)
  )
}

function buildFallbackImagePrompt({
  project,
  referenceImages,
  userRequest,
}: {
  project: ProjectRecord | null
  referenceImages: ReferenceImageRecord[]
  userRequest: string
}) {
  const requirements = JSON.stringify(project?.requirements ?? {}, null, 2)
  const guidelineBlock = truncateText(
    buildReferenceGuidelineBlock(referenceImages),
    2400
  )

  return [
    'Create a polished product design visualization based on the following project context.',
    '',
    `Project title: ${project?.title || 'Untitled project'}`,
    `User request: ${userRequest}`,
    '',
    'Project requirements:',
    requirements,
    '',
    'Reference design guidelines:',
    guidelineBlock,
    '',
    'Output direction:',
    '- preserve the reference image mood, materials, shape language, and detail points when relevant',
    '- generate a high-quality product concept/render image',
    '- no text overlay, no UI, no watermark',
    '- realistic studio or interior product presentation',
  ].join('\n')
}

function buildInitialDesignImagePrompt({
  project,
  referenceImages,
  conversation,
  userSelection,
}: {
  project: ProjectRecord | null
  referenceImages: ReferenceImageRecord[]
  conversation: string
  userSelection: string
}) {
  const requirements = truncateText(
    JSON.stringify(project?.requirements ?? {}, null, 2),
    1800
  )
  const guidelineBlock = truncateText(
    buildReferenceGuidelineBlock(referenceImages),
    2200
  )
  const conversationSnippet = truncateText(conversation, 3000)

  return [
    'Create four initial product design render options for STEP 5.',
    'The user selected one style reference direction in STEP 4. Use that selected direction as the single source style direction.',
    '',
    `Project title: ${project?.title || 'Untitled project'}`,
    `User selection: ${userSelection}`,
    '',
    'Project requirements:',
    requirements,
    '',
    'Conversation context:',
    conversationSnippet,
    '',
    'Reference design guidelines:',
    guidelineBlock,
    '',
    'Image direction:',
    '- create one complete product design render per output image; generate four distinct options total',
    '- keep the selected STEP 4 style direction consistent across all outputs',
    '- vary only sub-details such as proportion, surface treatment, hardware detail, or structural solution',
    '- realistic 3D product render, no text overlay, no UI, no watermark',
    '- do not create collages, boards, grids, labels, or multi-panel images',
  ].join('\n')
}

function buildStyleReferencePrompt({
  project,
  referenceImages,
  conversation,
}: {
  project: ProjectRecord | null
  referenceImages: ReferenceImageRecord[]
  conversation: string
}) {
  const requirements = JSON.stringify(project?.requirements ?? {}, null, 2)
  const guidelineBlock = truncateText(
    buildReferenceGuidelineBlock(referenceImages),
    2400
  )
  const conversationSnippet = truncateText(conversation, 3000)
  const requirementsSnippet = truncateText(requirements, 1800)

  return [
    'Create one standalone style reference image for a product design concept selection step.',
    'This prompt will be called multiple times to create separate options, so each output must contain only one complete reference image.',
    '',
    `Project title: ${project?.title || 'Untitled project'}`,
    '',
    'Project requirements:',
    requirementsSnippet,
    '',
    'Conversation context:',
    conversationSnippet,
    '',
    'Reference design guidelines:',
    guidelineBlock,
    '',
    'Image direction:',
    '- keep the same overall product/category and target user',
    '- create a single product/style concept image, not a board of multiple options',
    '- do not place three images, three panels, or before/after comparisons inside the output',
    '- no text overlay, no UI, no watermark',
    '- high-quality style board or product concept visual suitable for user selection',
  ].join('\n')
}

function buildStyleKeywordReferencePrompt({
  project,
  referenceImages,
  selectedKeywords,
}: {
  project: ProjectRecord | null
  referenceImages: ReferenceImageRecord[]
  selectedKeywords: string
}) {
  const requirements = truncateText(
    JSON.stringify(project?.requirements ?? {}, null, 2),
    1800
  )
  const guidelineBlock = truncateText(
    buildReferenceGuidelineBlock(referenceImages),
    2200
  )

  return [
    'Create one standalone style atmosphere reference image for a product design concept widget.',
    'The image should be useful as one selectable visual style direction, based on selected Korean style keywords.',
    '',
    `Project title: ${project?.title || 'Untitled project'}`,
    '',
    'Selected style keywords:',
    selectedKeywords,
    '',
    'Project requirements:',
    requirements,
    '',
    'Reference design guidelines:',
    guidelineBlock,
    '',
    'Image direction:',
    '- create a single polished product mood/style reference image',
    '- show the product mood through material, color, silhouette, lighting, and environment',
    '- no text, no labels, no watermark, no UI',
    '- no collage, no grid, no multi-panel moodboard',
    '- make each generated variation visibly different in mood while honoring the selected keywords',
  ].join('\n')
}

function buildMoodboardPrompt({
  project,
  referenceImages,
  styleProposal,
}: {
  project: ProjectRecord | null
  referenceImages: ReferenceImageRecord[]
  styleProposal: string
}) {
  const requirements = truncateText(
    JSON.stringify(project?.requirements ?? {}, null, 2),
    1600
  )
  const guidelineBlock = truncateText(
    buildReferenceGuidelineBlock(referenceImages),
    1800
  )

  return [
    'Create one refined product design moodboard image based on the selected style reference proposal.',
    '',
    `Project title: ${project?.title || 'Untitled project'}`,
    '',
    'Selected style proposal:',
    truncateText(styleProposal, 2200),
    '',
    'Project requirements:',
    requirements,
    '',
    'Reference design guidelines:',
    guidelineBlock,
    '',
    'Moodboard direction:',
    '- one cohesive moodboard image with product mood, material, color, shape, and tactile cues',
    '- elegant editorial composition, but no readable text labels',
    '- include abstract material/color/detail samples and one product-context visual direction',
    '- no watermark, no UI chrome',
  ].join('\n')
}

function buildCompactStyleReferencePrompt({
  project,
  referenceImages,
}: {
  project: ProjectRecord | null
  referenceImages: ReferenceImageRecord[]
}) {
  const guidelineBlock = truncateText(
    buildReferenceGuidelineBlock(referenceImages),
    1200
  )

  return [
    'Create one standalone style reference image for product design selection.',
    `Project title: ${project?.title || 'Untitled project'}`,
    '',
    'Reference design guidelines:',
    guidelineBlock,
    '',
    'Return one clear image only. No text, no collage, no UI, no watermark.',
  ].join('\n')
}

function isPersonaCardText(text: string) {
  return (
    text.includes('Persona Card') ||
    (text.includes('User') &&
      text.includes('Behavior Map') &&
      text.includes('Problem') &&
      text.includes('Decision'))
  )
}

function isPersonaImagePlaceholderText(text: string) {
  return (
    currentPersonaTextPattern.test(text) ||
    isPersonaSummaryDraftText(text) ||
    /페르소나.*이미지|이미지.*페르소나|이미지\s*placeholder|프롬프트\s*:|\(이미지\s*생성\s*중/i.test(
      text
    )
  )
}

const currentPersonaTextPattern =
  /이\s*페르소나.*리서치|리서치를\s*진행할까요|페르소나.*수정할까요/i

function isPersonaSummaryDraftText(text: string) {
  return (
    /페르소나.*(?:정보|내용|정리|카드|생성)/i.test(text) &&
    /(?:A\.\s*)?User|인적\s*\/\s*문화적\s*요소|행동\s*\/\s*라이프\s*패턴|사용\s*특성|소비\s*취향/i.test(
      text
    )
  )
}

function extractPersonaSummaryValue(text: string, labelPattern: RegExp) {
  const lines = text
    .replace(/\r\n/g, '\n')
    .split('\n')
    .map((line) =>
      line
        .replace(/^[-•]\s*/, '')
        .replace(/\*\*/g, '')
        .trim()
    )
    .filter(Boolean)

  const matchedLine = lines.find((line) => labelPattern.test(line))
  if (!matchedLine) {
    return ''
  }

  return matchedLine.replace(labelPattern, '').replace(/^[:：]\s*/, '').trim()
}

function summarizePersonaCardItem(text: string, fallback: string) {
  const normalized = (text || fallback)
    .replace(/\.{2,}|…/g, '')
    .replace(/[“”"']/g, '')
    .replace(/(?:입니다|합니다|느낍니다|보입니다|해요|어요|니다)$/g, '')
    .replace(/\s+/g, ' ')
    .trim()

  if (!normalized) {
    return fallback
  }

  const summaryRules: Array<[RegExp, string]> = [
    [/(20대|30대|40대).*직장인/, '$1 직장인'],
    [/고시생|수험생|시험|자격/, '시험 준비생'],
    [/프리랜서/, '프리랜서 사용자'],
    [/집중.*(흐트러|깨|끊).*재몰입|재몰입.*(어려|오래|지연)/, '재몰입 어려움'],
    [/집중.*(필요|유지|몰입)|몰입.*(시간|필요|유지)/, '몰입 시간 필요'],
    [/산만|방해|자극/, '외부 자극 취약'],
    [/효율|생산성/, '효율성 중시'],
    [/디자인|감성|심미/, '디자인 감성 중시'],
    [/성능|효과|기능/, '효과 우선'],
    [/가격|가성비/, '가성비 중시'],
    [/루틴|습관|반복/, '루틴 유지 필요'],
    [/업무|학습|공부/, '업무 학습 몰입'],
  ]

  for (const [pattern, replacement] of summaryRules) {
    if (pattern.test(normalized)) {
      return replacement.includes('$')
        ? normalized.replace(pattern, replacement).replace(/\s+/g, ' ').trim()
        : replacement
    }
  }

  const clause = normalized.split(/[,.，。:：/]|(?:\s+및\s+)|(?:\s+또는\s+)/)[0]?.trim()
  if (clause && clause.length <= 20) {
    return clause
  }

  const words = normalized.split(/\s+/).filter(Boolean)
  const phrase = words.reduce<string[]>((acc, word) => {
    const next = [...acc, word].join(' ')
    return next.length <= 18 ? [...acc, word] : acc
  }, [])

  return phrase.length > 0 ? phrase.join(' ') : fallback
}

function buildPersonaCardTextFromDraft(text: string) {
  const userValue =
    summarizePersonaCardItem(
      extractPersonaSummaryValue(text, /^(?:인적\s*\/\s*문화적\s*요소|User)\s*[:：]?\s*/i),
      '성인 집중 사용자'
    )
  const behaviorValue =
    summarizePersonaCardItem(
      extractPersonaSummaryValue(text, /^행동\s*\/\s*라이프\s*패턴\s*[:：]?\s*/i),
      '몰입 시간 필요'
    )
  const usageValue =
    summarizePersonaCardItem(
      extractPersonaSummaryValue(text, /^사용\s*특성\s*[:：]?\s*/i),
      '재몰입 어려움'
    )
  const preferenceValue =
    summarizePersonaCardItem(
      extractPersonaSummaryValue(text, /^소비\s*취향\s*[:：]?\s*/i),
      '효율성 중시'
    )

  return [
    '## 사용자 명확화 정리',
    '',
    '**사용자 정리**',
    `- ${userValue}`,
    '',
    '**사용 상황**',
    `- ${behaviorValue}`,
    '',
    '**핵심 문제**',
    `- ${usageValue}`,
    '- 회복 시간 지연',
    '',
    '**성공 기준**',
    '- 빠른 몰입 회복',
    '- 재시작 시간 절약',
    '',
    '**선택 기준**',
    `- ${preferenceValue}`,
    '- 실제 집중 효과',
  ].join('\n')
}

function getPersonaClarificationStatus(text: string) {
  const normalized = text.replace(/\s+/g, ' ')
  const hasHumanProfile =
    /(?:\d{2}\s*대|20대|30대|40대|직장인|학생|고시생|프리랜서|자영업|부모|거주|성향|가치관|성격)/i.test(
      normalized
    )
  const hasUsageContext =
    /(?:언제|순간|하루|아침|저녁|업무|학습|공부|집중|사용\s*전|사용\s*중|사용\s*후|공간|카페|서재|침실|사무실|TPO)/i.test(
      normalized
    )
  const hasPainPoint =
    /(?:불편|문제|어려움|결핍|니즈|욕구|흐트러|재몰입|해결|한계|제약)/i.test(
      normalized
    )
  const hasDecisionCriteria =
    /(?:구매|선호|취향|브랜드|가격|성능|디자인|감성|우선순위|소비|효율|생산성|가치)/i.test(
      normalized
    )
  const completedCount = [
    hasHumanProfile,
    hasUsageContext,
    hasPainPoint,
    hasDecisionCriteria,
  ].filter(Boolean).length

  return {
    hasHumanProfile,
    hasUsageContext,
    hasPainPoint,
    hasDecisionCriteria,
    isComplete: completedCount >= 4,
  }
}

function buildPersonaClarificationQuestion(
  text: string,
  project: ProjectRecord | null
) {
  const status = getPersonaClarificationStatus(text)
  const hintContext = buildProjectHintContext(project)

  if (!status.hasHumanProfile) {
    const choices = buildContextualHintChoices(hintContext, 'persona_user')

    return [
      '좋아요. 페르소나 카드를 만들기 전에 사용자를 조금 더 구체화해야 합니다.',
      '',
      '이 제품을 가장 자주 쓸 사람의 나이대와 직업은 무엇에 가깝나요?',
      '',
      `A. ${choices[0]}`,
      `B. ${choices[1]}`,
      `C. ${choices[2]}`,
    ].join('\n')
  }

  if (!status.hasUsageContext) {
    const choices = buildContextualHintChoices(hintContext, 'persona_usage')

    return [
      '좋아요. 사용자의 기본 윤곽은 잡혔고, 이제 실제 사용 장면이 필요합니다.',
      '',
      '그 사람이 하루 중 이 제품을 가장 필요로 하는 순간은 언제인가요?',
      '',
      `A. ${choices[0]}`,
      `B. ${choices[1]}`,
      `C. ${choices[2]}`,
    ].join('\n')
  }

  if (!status.hasPainPoint) {
    const choices = buildContextualHintChoices(hintContext, 'persona_pain')

    return [
      '좋아요. 사용 장면은 잡혔고, 이제 해결해야 할 불편함을 좁히면 됩니다.',
      '',
      '그 순간에 사용자가 가장 크게 불편해하는 점은 무엇인가요?',
      '',
      `A. ${choices[0]}`,
      `B. ${choices[1]}`,
      `C. ${choices[2]}`,
    ].join('\n')
  }

  const choices = buildContextualHintChoices(hintContext, 'persona_decision')

  return [
    '좋아요. 문제 상황까지 잡혔고, 마지막으로 선택 기준을 확인하면 페르소나 카드로 정리할 수 있습니다.',
    '',
    '이 사용자가 제품을 고를 때 가장 먼저 볼 기준은 무엇일까요?',
    '',
    `A. ${choices[0]}`,
    `B. ${choices[1]}`,
    `C. ${choices[2]}`,
  ].join('\n')
}

function buildPersonaImagePrompt({
  project,
  personaText,
}: {
  project: ProjectRecord | null
  personaText: string
}) {
  const projectRequirements = truncateText(
    JSON.stringify(project?.requirements ?? {}, null, 2),
    1200
  )
  const personaSnippet = truncateText(personaText, 2600)

  return [
    'Create one realistic persona profile image for a product design persona card.',
    '',
    `Project title: ${project?.title || 'Untitled project'}`,
    '',
    'Project requirements:',
    projectRequirements,
    '',
    'Persona card text:',
    personaSnippet,
    '',
    'Image direction:',
    '- infer visible persona traits only from the persona card text: gender if explicit, approximate adult age range, occupation/lifestyle, and target-user characteristics',
    '- if gender or exact age is not explicit, use a natural gender-neutral adult representation',
    '- if the persona seems younger than 20, represent the target as a young adult in their 20s; do not depict minors',
    '- create a single waist-up portrait or lifestyle portrait suitable for the left image area of a persona card',
    '- clean modern editorial style, realistic but not a celebrity, natural expression',
    '- simple background related subtly to the persona lifestyle or product context',
    '- vertical composition, subject centered, enough negative space, high quality',
    '- no text, no labels, no UI, no watermark, no collage',
  ].join('\n')
}

const PERSONA_FLOW_CARD_LABELS: Record<PersonaFlowArtifactKind, string> = {
  problem_statements: 'Problem Statements',
  experience_keywords: 'Keywords: Experience',
  relationship_keywords: 'Keywords: Relationship',
}

const PERSONA_PROBLEM_QUESTIONS = {
  triggerScene: /이\s*제품을\s*떠올리게\s*된\s*장면은\s*어떤\s*순간에\s*가까운가요\?/i,
  userAction: /그\s*순간\s*사용자는\s*무엇을\s*하려던\s*중인가요\?/i,
  interruption: /그\s*흐름을\s*가장\s*자주\s*흐트러뜨리는\s*것은\s*무엇인가요\?/i,
  workaround: /지금은\s*그\s*상황을\s*어떻게\s*넘기고\s*있나요\?/i,
  residue: /그\s*방식이\s*반복될\s*때\s*결국\s*어떤\s*문제가\s*남나요\?/i,
  desiredChange: /문제가\s*줄어든다면\s*사용자의\s*하루나\s*행동은\s*어떻게\s*달라질까요\?/i,
}

const PERSONA_EXPERIENCE_QUESTIONS = {
  emotionEntry:
    /(?:감정\s*1:\s*)?제품을\s*처음\s*사용할\s*때\s*사용자가\s*가장\s*먼저\s*어떤\s*정서를\s*느끼길\s*원하나요\?/i,
  emotionRelief:
    /(?:감정\s*2:\s*)?방해나\s*압박이\s*줄어들\s*때\s*어떤\s*감정\s*변화가\s*생기면\s*좋을까요\?/i,
  emotionAfter:
    /(?:감정\s*3:\s*)?사용을\s*마친\s*뒤\s*사용자가\s*어떤\s*감정을\s*남기길\s*원하나요\?/i,
  behaviorStart:
    /(?:행동\s*1:\s*)?제품이\s*사용자의\s*시작\s*행동을\s*어떻게\s*도와야\s*할까요\?/i,
  behaviorRecovery:
    /(?:행동\s*2:\s*)?흐름이\s*끊겼을\s*때\s*어떤\s*회복\s*행동이\s*생기면\s*좋을까요\?/i,
  behaviorRoutine:
    /(?:행동\s*3:\s*)?사용자가\s*시간과\s*루틴을\s*어떻게\s*다루게\s*되길\s*원하나요\?/i,
  spaceImpression:
    /(?:공간\s*1:\s*)?제품이\s*놓인\s*공간에서\s*어떤\s*첫인상을\s*주면\s*좋을까요\?/i,
  spaceMeaning:
    /(?:공간\s*2:\s*)?사용자가\s*그\s*공간을\s*어떻게\s*느끼게\s*만들면\s*좋을까요\?/i,
  spaceRelation:
    /(?:공간\s*3:\s*)?주변\s*물건이나\s*생활\s*흐름과\s*어떤\s*관계를\s*맺으면\s*좋을까요\?/i,
}

const PERSONA_RELATIONSHIP_QUESTIONS = {
  interruption: /사용자와\s*기존\s*방해\s*요소는\s*어떤\s*관계에\s*가깝나요\?/i,
  space: /사용자와\s*제품이\s*놓이는\s*공간은\s*어떤\s*관계에\s*가깝나요\?/i,
  time: /사용자는\s*집중과\s*휴식\s*시간을\s*어떻게\s*인식하길\s*원하나요\?/i,
}

function cleanPersonaFlowAnswer(text: string) {
  return text
    .replace(/^시각화하기\s*/i, '')
    .replace(/<<AIDEE_PERSONA_FLOW_CARD:[\s\S]*?<<\/AIDEE_PERSONA_FLOW_CARD>>/g, '')
    .replace(/^[ABC][.)]\s*/i, '')
    .replace(/\s+/g, ' ')
    .trim()
}

function compactPersonaFlowValue(text: string, fallback: string) {
  const cleaned = cleanPersonaFlowAnswer(text)

  if (!cleaned) {
    return fallback
  }

  if (cleaned.length <= 90) {
    return cleaned
  }

  return `${cleaned.slice(0, 90).trim()}...`
}

const MAX_KEYWORD_CHARS = 10

const SHORT_KEYWORD_RULES: Array<[RegExp, string]> = [
  [/신뢰|믿고/, '신뢰감'],
  [/기대/, '기대감'],
  [/편안|편안함/, '편안함'],
  [/안도/, '안도감'],
  [/개운|번거로움.*줄/, '개운함'],
  [/안정|매끄러/, '안정감'],
  [/만족/, '만족감'],
  [/성취|해냈/, '성취감'],
  [/여유/, '여유감'],
  [/시작과\s*끝|시간/, '시간 인식'],
  [/쉽게\s*시작|시작/, '쉬운 시작'],
  [/회복|다시\s*돌아|재몰입/, '흐름 회복'],
  [/정렬/, '행동 정렬'],
  [/루틴|반복/, '루틴 형성'],
  [/조절/, '자기조절'],
  [/생활\s*흐름|생활\s*연결/, '생활 연결'],
  [/자연스럽게\s*놓|자연.*배치/, '자연 배치'],
  [/역할/, '역할 명확'],
  [/분위기/, '분위기 형성'],
  [/기준점/, '행동 기준'],
  [/공간\s*신호/, '공간 신호'],
  [/충돌하지|조화/, '공간 조화'],
  [/거리|방해/, '방해 완화'],
  [/이탈/, '이탈 방지'],
  [/정돈|정리/, '정돈감'],
  [/보관|수납/, '보관 편의'],
  [/위생|세척|관리/, '관리 용이'],
  [/이동|휴대/, '휴대성'],
  [/불편|문제/, '불편 해소'],
]

function trimKeywordLength(text: string) {
  const cleaned = text.replace(/\s+/g, ' ').trim()
  const chars = Array.from(cleaned)

  if (chars.length <= MAX_KEYWORD_CHARS) {
    return cleaned
  }

  return Array.from(cleaned.replace(/\s+/g, '')).slice(0, MAX_KEYWORD_CHARS).join('')
}

function splitKeywordCandidates(text: string) {
  const cleaned = cleanPersonaFlowAnswer(text)
    .replace(/^[^:：]{1,12}\s*단서\s*[:：]\s*/i, '')
    .trim()
  const keywordText = /[:：]/.test(cleaned)
    ? cleaned.split(/[:：]/).slice(1).join(' ')
    : cleaned

  return keywordText
    .split(
      /[,，、/]|(?:\s+및\s+)|(?:\s+그리고\s+)|(?:(?:과|와)\s+)|(?:\s*·\s*)/g
    )
    .map((keyword) => keyword.replace(/\s+/g, ' ').trim())
    .filter(Boolean)
}

function toShortKeyword(text: string, fallback: string) {
  const cleaned = cleanPersonaFlowAnswer(text)
    .replace(/^[^:：]{1,12}\s*단서\s*[:：]\s*/i, '')
    .replace(/[“”"']/g, '')
    .replace(/[.!?。！？]$/g, '')
    .trim()

  for (const [pattern, keyword] of SHORT_KEYWORD_RULES) {
    if (pattern.test(cleaned)) {
      return keyword
    }
  }

  const quotedMatch = cleaned.match(/[‘`](.{1,10})[’`]/)
  if (quotedMatch?.[1]) {
    return trimKeywordLength(quotedMatch[1])
  }

  const normalized = cleaned
    .replace(/[‘’`]/g, '')
    .replace(
      /(느끼면|생기면|되면|하면|하기|함|됨|중|때|정도|인지|는지|필요한|필요|실제로|자연스럽게|반복적으로|계속|다시|스스로|사용자|제품|기존|방향|단서)\s*/g,
      ''
    )
    .replace(/\s+/g, ' ')
    .trim()

  return trimKeywordLength(normalized || fallback)
}

function buildShortKeywordList(
  text: string,
  fallbackKeywords: string[],
  targetCount: number
) {
  const seen = new Set<string>()
  const keywords: string[] = []

  for (const candidate of [...splitKeywordCandidates(text), ...fallbackKeywords]) {
    const keyword = toShortKeyword(candidate, fallbackKeywords[0] ?? '핵심 키워드')
    const normalized = keyword.replace(/\s+/g, '').toLowerCase()

    if (!normalized || seen.has(normalized)) {
      continue
    }

    seen.add(normalized)
    keywords.push(keyword)

    if (keywords.length === targetCount) {
      break
    }
  }

  return keywords
}

function ensureKoreanSentence(text: string) {
  const cleaned = text.replace(/\s+/g, ' ').trim()

  if (!cleaned) {
    return ''
  }

  if (/[.!?。！？]$/.test(cleaned)) {
    return cleaned
  }

  if (/(다|요|니다|습니다)$/.test(cleaned)) {
    return `${cleaned}.`
  }

  return `${cleaned}을 중심으로 한다.`
}

function normalizeProblemStatementText(text: string) {
  return text.replace(/[\s.,:;!?。！？'"]/g, '').trim()
}

function cleanupProblemStatementTitle(text: string) {
  return text
    .replace(/\.\.\.$/, '')
    .replace(/^(사용자는|사용자가|기존 방식은|실제)\s*/, '')
    .replace(/(?:상황을\s*다룬다|문제가\s*남는다|필요하다|원한다|기반으로\s*한다|이어진다)$/g, '')
    .replace(/\s+/g, ' ')
    .trim()
}

function isGenericProblemStatementTitle(title: string) {
  return /^(?:사용 맥락|문제 흐름|필요 방향|Context|Problems|Needs)(?:\s*요약)?$|^수정된\s*(?:사용 맥락|문제 흐름|필요 방향)$/i.test(
    title.trim()
  )
}

function summarizeProblemStatementTitle(description: string, fallback: string) {
  const firstSentence = description
    .replace(/\s+/g, ' ')
    .trim()
    .split(/[.!?。！？]/)[0]
    ?.trim()
  const colonTitle = firstSentence?.split(/[:：]/)[0]?.trim() || ''
  const title = cleanupProblemStatementTitle(colonTitle)

  if (
    !title ||
    normalizeProblemStatementText(title) === normalizeProblemStatementText(description)
  ) {
    return fallback
  }

  return title
}

function buildProblemStatementSection({
  label,
  title,
  description,
}: {
  label: string
  title: string
  description: string
}) {
  const shouldSummarizeTitle =
    isGenericProblemStatementTitle(title) ||
    normalizeProblemStatementText(title) === normalizeProblemStatementText(description)
  const sectionTitle = shouldSummarizeTitle
    ? summarizeProblemStatementTitle(description, title)
    : cleanupProblemStatementTitle(title)

  return [
    `**${label}**`,
    `**${sectionTitle}**`,
    description,
    '',
  ]
}

function getAnswerAfterQuestion(
  messages: NormalizedMessage[],
  questionPattern: RegExp
) {
  let answer = ''

  messages.forEach((message, index) => {
    if (message.role !== 'assistant' || !questionPattern.test(message.content)) {
      return
    }

    const nextUserMessage = messages
      .slice(index + 1)
      .find((candidate) => candidate.role === 'user')

    if (nextUserMessage) {
      answer = cleanPersonaFlowAnswer(nextUserMessage.content)
    }
  })

  return answer
}

function hasPersonaFlowCard(
  messages: NormalizedMessage[],
  kind: PersonaFlowArtifactKind
) {
  const marker = `<<AIDEE_PERSONA_FLOW_CARD:${kind}>>`

  return messages.some((message) => message.content.includes(marker))
}

function hasPersonaFlowCardConfirmation(
  messages: NormalizedMessage[],
  kind: PersonaFlowArtifactKind
) {
  const label = PERSONA_FLOW_CARD_LABELS[kind].replace(/[.*+?^${}()|[\]\\]/g, '\\$&')
  const confirmationPattern = new RegExp(`${label}\\s*카드[\\s\\S]*확정`, 'i')

  return messages.some(
    (message) => message.role === 'user' && confirmationPattern.test(message.content)
  )
}

function isProblemStatementsConfirmation(text: string) {
  return /Problem\s*Statements\s*카드[\s\S]*확정|문제\s*정리\s*카드[\s\S]*확정/i.test(
    text
  )
}

function isProblemStatementsRevisionRequest(text: string) {
  return /Problem\s*Statements\s*카드[\s\S]*(수정|보완|다시|고치|바꾸)|문제\s*정리[\s\S]*(수정|보완|다시|고치|바꾸)/i.test(
    text
  )
}

function hasRecentProblemStatementsRevisionPrompt(messages: NormalizedMessage[]) {
  const latestAssistant = messages
    .slice()
    .reverse()
    .find((message) => message.role === 'assistant')

  return Boolean(
    latestAssistant &&
      /Problem\s*Statements\s*카드에서\s*수정하고\s*싶은\s*부분/i.test(
        latestAssistant.content
      )
  )
}

function buildProblemStatementsRevisionQuestion(hintContext: ProjectHintContext) {
  return buildPersonaQuestion({
    intro:
      '좋아요. Problem Statements 카드는 확정하지 않고 수정 상태로 둘게요.',
    question:
      'Problem Statements 카드에서 수정하고 싶은 부분을 어떤 방향으로 바꾸면 좋을까요?',
    choices: buildContextualHintChoices(hintContext, 'problem_revision'),
  })
}

function extractSummaryFromVisualizationCommand(text: string) {
  return text
    .replace(/^시각화하기\s*/i, '')
    .replace(/<<AIDEE_PERSONA_FLOW_CARD:[\s\S]*?<<\/AIDEE_PERSONA_FLOW_CARD>>/g, '')
    .split('\n')
    .filter(
      (line) =>
        !/아래의 시각화하기 버튼|내용을\s*확인한\s*뒤\s*확정하기를\s*누르면\s*시각화하기\s*버튼이\s*나타납니다/.test(
          line
        )
    )
    .join('\n')
    .trim()
}

function findLatestPersonaSummary(
  messages: NormalizedMessage[],
  headingPattern: RegExp,
  extraText = ''
) {
  const candidates = [
    ...messages.map((message) => message.content),
    extraText,
  ].filter(Boolean)

  return (
    candidates
      .slice()
      .reverse()
      .find((content) => headingPattern.test(content)) ?? ''
  )
}

function extractSummarySection(
  summary: string,
  labels: string[],
  boundaryLabels = labels
) {
  const normalized = summary.replace(/\r\n/g, '\n')
  const escapedLabels = labels.map((label) =>
    label.replace(/[.*+?^${}()|[\]\\]/g, '\\$&')
  )
  const escapedBoundaryLabels = boundaryLabels.map((label) =>
    label.replace(/[.*+?^${}()|[\]\\]/g, '\\$&')
  )

  for (const label of escapedLabels) {
    const otherLabels = escapedBoundaryLabels
      .filter((item) => item !== label)
      .join('|')
    const lookahead = otherLabels
      ? `(?=\\n\\s*(?:#{1,6}\\s*)?(?:\\*\\*)?(?:${otherLabels})(?:\\*\\*)?\\s*\\n|$)`
      : '$'
    const regex = new RegExp(
      `(?:^|\\n)\\s*(?:#{1,6}\\s*)?(?:\\*\\*)?${label}(?:\\*\\*)?\\s*\\n([\\s\\S]*?)${lookahead}`,
      'i'
    )
    const match = normalized.match(regex)

    if (match) {
      return match[1]
        .split('\n')
        .map((line) =>
          line
            .replace(/^[-•]\s*/, '')
            .replace(/\*\*/g, '')
            .replace(/\s+/g, ' ')
            .trim()
        )
        .filter(Boolean)
        .join(' / ')
    }
  }

  return ''
}

function buildPersonaFlowCardBlock(
  kind: PersonaFlowArtifactKind,
  summary: string
) {
  const cardSummary =
    kind === 'experience_keywords'
      ? buildExperienceKeywordsCardSummary(summary)
      : summary.trim()

  return [
    `<<AIDEE_PERSONA_FLOW_CARD:${kind}>>`,
    cardSummary,
    '<</AIDEE_PERSONA_FLOW_CARD>>',
  ].join('\n')
}

function buildPersonaQuestion({
  intro,
  question,
  choices,
}: {
  intro: string
  question: string
  choices: [string, string, string]
}) {
  return [
    intro,
    '',
    question,
    '',
    `A. ${choices[0]}`,
    `B. ${choices[1]}`,
    `C. ${choices[2]}`,
  ].join('\n')
}

function buildPersonaChoiceQuestion({
  question,
  choices,
}: {
  question: string
  choices: [string, string, string]
}) {
  return [
    question,
    '',
    `A. ${choices[0]}`,
    `B. ${choices[1]}`,
    `C. ${choices[2]}`,
  ].join('\n')
}

function buildProblemSituationQuestion(hintContext: ProjectHintContext) {
  return buildPersonaQuestion({
    intro:
      'STEP 2에서는 먼저 한 사람의 사용 장면을 천천히 따라가며 문제의 맥락을 잡아볼게요.',
    question: '이 제품을 떠올리게 된 장면은 어떤 순간에 가까운가요?',
    choices: buildContextualHintChoices(hintContext, 'problem_scene'),
  })
}

function buildProblemUserActionQuestion(hintContext: ProjectHintContext) {
  return buildPersonaQuestion({
    intro: '좋아요. 그 장면에서 사용자의 행동을 조금 더 가까이 보겠습니다.',
    question: '그 순간 사용자는 무엇을 하려던 중인가요?',
    choices: buildContextualHintChoices(hintContext, 'problem_action'),
  })
}

function buildProblemInterruptionQuestion(hintContext: ProjectHintContext) {
  return buildPersonaQuestion({
    intro: '좋아요. 이제 그 흐름이 어디서 흔들리는지 살펴볼게요.',
    question: '그 흐름을 가장 자주 흐트러뜨리는 것은 무엇인가요?',
    choices: buildContextualHintChoices(hintContext, 'problem_interruption'),
  })
}

function buildProblemWorkaroundQuestion(hintContext: ProjectHintContext) {
  return buildPersonaQuestion({
    intro: '좋아요. 사용자가 지금 어떤 방식으로 버티고 있는지도 보겠습니다.',
    question: '지금은 그 상황을 어떻게 넘기고 있나요?',
    choices: buildContextualHintChoices(hintContext, 'problem_workaround'),
  })
}

function buildProblemResidueQuestion(hintContext: ProjectHintContext) {
  return buildPersonaQuestion({
    intro: '좋아요. 현재 방식이 충분하지 않은 지점을 더 분명히 잡아볼게요.',
    question: '그 방식이 반복될 때 결국 어떤 문제가 남나요?',
    choices: buildContextualHintChoices(hintContext, 'problem_residue'),
  })
}

function buildProblemDesiredChangeQuestion(hintContext: ProjectHintContext) {
  return buildPersonaQuestion({
    intro: '좋아요. 마지막으로 해결 이후의 변화를 그려보겠습니다.',
    question: '문제가 줄어든다면 사용자의 하루나 행동은 어떻게 달라질까요?',
    choices: buildContextualHintChoices(hintContext, 'problem_change'),
  })
}

function buildProblemStatementsSummary(messages: NormalizedMessage[]) {
  const triggerScene = compactPersonaFlowValue(
    getAnswerAfterQuestion(messages, PERSONA_PROBLEM_QUESTIONS.triggerScene),
    '집중이 필요한 순간'
  )
  const userAction = compactPersonaFlowValue(
    getAnswerAfterQuestion(messages, PERSONA_PROBLEM_QUESTIONS.userAction),
    '몰입을 시작하거나 회복하려는 행동'
  )
  const interruption = compactPersonaFlowValue(
    getAnswerAfterQuestion(messages, PERSONA_PROBLEM_QUESTIONS.interruption),
    '반복적인 방해 요소'
  )
  const workaround = compactPersonaFlowValue(
    getAnswerAfterQuestion(messages, PERSONA_PROBLEM_QUESTIONS.workaround),
    '현재 방식으로 임시 대응'
  )
  const residue = compactPersonaFlowValue(
    getAnswerAfterQuestion(messages, PERSONA_PROBLEM_QUESTIONS.residue),
    '재몰입과 루틴 유지의 어려움'
  )
  const needs = compactPersonaFlowValue(
    getAnswerAfterQuestion(messages, PERSONA_PROBLEM_QUESTIONS.desiredChange),
    '몰입 리듬을 스스로 회복하고 유지하는 변화'
  )
  const contextDescription = [
    ensureKoreanSentence(`${triggerScene}에서 사용자가 ${userAction} 상황을 다룬다`),
    '실제 생활 공간 속 반복되는 행동 흐름과 사용 패턴을 기반으로 한다.',
  ].join(' ')
  const problemsDescription = [
    ensureKoreanSentence(
      `${interruption} 때문에 사용자는 ${workaround}로 대응하지만, ${residue} 문제가 남는다`
    ),
    '기존 방식은 문제를 해결하기보다 흐름을 끊거나 관리 부담을 늘리는 상황으로 이어진다.',
  ].join(' ')
  const needsDescription = [
    ensureKoreanSentence(`${needs}가 필요하다`),
    '사용자는 방해 요소를 억지로 참기보다 자연스럽게 행동을 이어가고 다시 돌아올 수 있는 환경을 원한다.',
  ].join(' ')

  return [
    '## Problem Statements',
    '',
    ...buildProblemStatementSection({
      label: '01. Context',
      title: `${triggerScene.replace(/\.\.\.$/, '')}의 사용 맥락`,
      description: contextDescription,
    }),
    ...buildProblemStatementSection({
      label: '02. Problems',
      title: `${interruption.replace(/\.\.\.$/, '')}로 인한 문제 흐름`,
      description: problemsDescription,
    }),
    ...buildProblemStatementSection({
      label: '03. Needs',
      title: `${needs.replace(/\.\.\.$/, '')}를 위한 필요 방향`,
      description: needsDescription,
    }),
  ].join('\n')
}

function buildRevisedProblemStatementsSummary(
  messages: NormalizedMessage[],
  revisionText: string
) {
  const latestSummary = findLatestVisualizationSummary(
    messages,
    /##\s*Problem Statements/i
  )
  const labels = [
    '01. Context',
    '02. Problems',
    '03. Needs',
    'Context',
    'Problems',
    'Needs',
    '문제(현재 상황)',
    '불편함',
  ]
  let context =
    extractSummarySection(latestSummary, ['01. Context', 'Context', '문제(현재 상황)'], labels) ||
    '집중이 필요한 순간 반복적으로 문제 경험'
  let problems =
    extractSummarySection(latestSummary, ['02. Problems', 'Problems', '불편함'], labels) ||
    '기존 방식이 방해 요소를 줄여주지 못함'
  let needs =
    extractSummarySection(latestSummary, ['03. Needs', 'Needs'], labels) ||
    '몰입을 회복하고 루틴을 유지하는 해결 필요'
  const revision = compactPersonaFlowValue(
    revisionText,
    '사용자가 수정 요청한 내용을 반영'
  )

  if (/장면|상황|언제|어디|순간|행동|하려던/i.test(revision)) {
    context = revision
  } else if (/원|필요|해결|달라|변화|기대|목표/i.test(revision)) {
    needs = revision
  } else {
    problems = revision
  }

  return [
    '## Problem Statements',
    '',
    ...buildProblemStatementSection({
      label: '01. Context',
      title: '사용 맥락 요약',
      description: `${ensureKoreanSentence(context)} 이 맥락을 기준으로 사용자가 제품을 필요로 하는 장면과 생활 패턴을 다시 정리한다.`,
    }),
    ...buildProblemStatementSection({
      label: '02. Problems',
      title: '문제 흐름 요약',
      description: `${ensureKoreanSentence(problems)} 반복되는 불편과 현재 대응 방식의 한계를 중심으로 문제를 다시 정리한다.`,
    }),
    ...buildProblemStatementSection({
      label: '03. Needs',
      title: '필요 방향 요약',
      description: `${ensureKoreanSentence(needs)} 사용자가 기대하는 변화와 제품이 제공해야 할 지원 방향을 다시 정리한다.`,
    }),
    '수정 내용을 반영했습니다.',
  ].join('\n')
}

const EXPERIENCE_FALLBACK_KEYWORDS = {
  emotion: [
    '평온함',
    '신뢰감',
    '기대감',
    '안도감',
    '가벼움',
    '개운함',
    '성취감',
    '자기효능감',
    '만족감',
    '안정감',
    '몰입감',
    '여유감',
    '회복감',
    '집중감',
  ],
  behavior: [
    '쉽게 시작',
    '자연스러운 전환',
    '준비 시간 단축',
    '빠른 재몰입',
    '다시 정렬',
    '방해 차단',
    '루틴 형성',
    '시간 인식',
    '자기조절',
    '지속성',
    '집중 유지',
    '행동 전환',
    '습관화',
    '우선순위 정리',
  ],
  space: [
    '책상 위 오브제',
    '정돈된 분위기',
    '차분한 포인트',
    '안전한 몰입 공간',
    '개인화된 자리',
    '조용한 존재감',
    '자연스러운 배치',
    '시각적 정돈',
    '생활 리듬 연결',
    '공간의 안정감',
  ],
}

function splitExperienceKeywords(text: string) {
  return splitKeywordCandidates(text).map((keyword) =>
    keyword.replace(/^(감정|행동|공간)\s*[:：]\s*/i, '').trim()
  )
}

function buildKeywordGroup(
  answerTexts: string[],
  fallbackKeywords: string[],
  targetCount: number
) {
  const seen = new Set<string>()
  const keywords: string[] = []

  for (const keyword of [
    ...answerTexts.flatMap(splitExperienceKeywords),
    ...fallbackKeywords,
  ]) {
    const shortKeyword = toShortKeyword(keyword, fallbackKeywords[0] ?? '키워드')
    const normalized = shortKeyword.replace(/\s+/g, '').toLowerCase()

    if (!normalized || seen.has(normalized)) {
      continue
    }

    seen.add(normalized)
    keywords.push(shortKeyword)

    if (keywords.length === targetCount) {
      break
    }
  }

  return keywords
}

function buildExperienceEmotionEntryQuestion(hintContext: ProjectHintContext) {
  return buildPersonaChoiceQuestion({
    question:
      '제품을 처음 사용할 때 사용자가 가장 먼저 어떤 정서를 느끼길 원하나요?',
    choices: buildContextualHintChoices(
      hintContext,
      'experience_emotion_entry'
    ),
  })
}

function buildExperienceEmotionReliefQuestion(hintContext: ProjectHintContext) {
  return buildPersonaChoiceQuestion({
    question:
      '방해나 압박이 줄어들 때 어떤 감정 변화가 생기면 좋을까요?',
    choices: buildContextualHintChoices(
      hintContext,
      'experience_emotion_relief'
    ),
  })
}

function buildExperienceEmotionAfterQuestion(hintContext: ProjectHintContext) {
  return buildPersonaChoiceQuestion({
    question:
      '사용을 마친 뒤 사용자가 어떤 감정을 남기길 원하나요?',
    choices: buildContextualHintChoices(
      hintContext,
      'experience_emotion_after'
    ),
  })
}

function buildExperienceBehaviorStartQuestion(hintContext: ProjectHintContext) {
  return buildPersonaChoiceQuestion({
    question:
      '제품이 사용자의 시작 행동을 어떻게 도와야 할까요?',
    choices: buildContextualHintChoices(
      hintContext,
      'experience_behavior_start'
    ),
  })
}

function buildExperienceBehaviorRecoveryQuestion(hintContext: ProjectHintContext) {
  return buildPersonaChoiceQuestion({
    question:
      '흐름이 끊겼을 때 어떤 회복 행동이 생기면 좋을까요?',
    choices: buildContextualHintChoices(
      hintContext,
      'experience_behavior_recovery'
    ),
  })
}

function buildExperienceBehaviorRoutineQuestion(hintContext: ProjectHintContext) {
  return buildPersonaChoiceQuestion({
    question:
      '사용자가 시간과 루틴을 어떻게 다루게 되길 원하나요?',
    choices: buildContextualHintChoices(
      hintContext,
      'experience_behavior_routine'
    ),
  })
}

function buildExperienceSpaceImpressionQuestion(hintContext: ProjectHintContext) {
  return buildPersonaChoiceQuestion({
    question:
      '제품이 놓인 공간에서 어떤 첫인상을 주면 좋을까요?',
    choices: buildContextualHintChoices(
      hintContext,
      'experience_space_impression'
    ),
  })
}

function buildExperienceSpaceMeaningQuestion(hintContext: ProjectHintContext) {
  return buildPersonaChoiceQuestion({
    question:
      '사용자가 그 공간을 어떻게 느끼게 만들면 좋을까요?',
    choices: buildContextualHintChoices(
      hintContext,
      'experience_space_meaning'
    ),
  })
}

function buildExperienceSpaceRelationQuestion(hintContext: ProjectHintContext) {
  return buildPersonaChoiceQuestion({
    question:
      '주변 물건이나 생활 흐름과 어떤 관계를 맺으면 좋을까요?',
    choices: buildContextualHintChoices(
      hintContext,
      'experience_space_relation'
    ),
  })
}

function buildExperienceKeywordsSummary(messages: NormalizedMessage[]) {
  const emotionKeywords = buildKeywordGroup(
    [
      getAnswerAfterQuestion(messages, PERSONA_EXPERIENCE_QUESTIONS.emotionEntry),
      getAnswerAfterQuestion(messages, PERSONA_EXPERIENCE_QUESTIONS.emotionRelief),
      getAnswerAfterQuestion(messages, PERSONA_EXPERIENCE_QUESTIONS.emotionAfter),
    ],
    EXPERIENCE_FALLBACK_KEYWORDS.emotion,
    12
  )
  const behaviorKeywords = buildKeywordGroup(
    [
      getAnswerAfterQuestion(messages, PERSONA_EXPERIENCE_QUESTIONS.behaviorStart),
      getAnswerAfterQuestion(
        messages,
        PERSONA_EXPERIENCE_QUESTIONS.behaviorRecovery
      ),
      getAnswerAfterQuestion(messages, PERSONA_EXPERIENCE_QUESTIONS.behaviorRoutine),
    ],
    EXPERIENCE_FALLBACK_KEYWORDS.behavior,
    12
  )
  const spaceKeywords = buildKeywordGroup(
    [
      getAnswerAfterQuestion(messages, PERSONA_EXPERIENCE_QUESTIONS.spaceImpression),
      getAnswerAfterQuestion(messages, PERSONA_EXPERIENCE_QUESTIONS.spaceMeaning),
      getAnswerAfterQuestion(messages, PERSONA_EXPERIENCE_QUESTIONS.spaceRelation),
    ],
    EXPERIENCE_FALLBACK_KEYWORDS.space,
    8
  )

  return [
    '## Keywords: Experience',
    '',
    '감정, 행동, 공간 관점의 질문을 바탕으로 총 32개의 경험 키워드를 정리했습니다.',
    '',
    '**감정 Keywords (12)**',
    `- ${emotionKeywords.join(', ')}`,
    '',
    '**행동 Keywords (12)**',
    `- ${behaviorKeywords.join(', ')}`,
    '',
    '**공간 Keywords (8)**',
    `- ${spaceKeywords.join(', ')}`,
    '',
    'Keywords: Experience 정리가 완료되었습니다.',
  ].join('\n')
}

function buildRelationshipInterruptionQuestion(hintContext: ProjectHintContext) {
  return buildPersonaQuestion({
    intro:
      'Keywords: Experience 카드가 만들어졌습니다. 이제 사용자와 주변 요소 사이의 관계를 키워드로 정리해볼게요.',
    question:
      '기존 방해 요소와의 관계: 사용자와 기존 방해 요소는 어떤 관계에 가깝나요?',
    choices: buildContextualHintChoices(hintContext, 'relationship_interruption'),
  })
}

function buildRelationshipSpaceQuestion(hintContext: ProjectHintContext) {
  return buildPersonaQuestion({
    intro: '좋아요. 이번에는 제품이 놓이는 공간과 사용자의 관계를 보겠습니다.',
    question:
      '제품이 놓이는 공간과의 관계: 사용자와 제품이 놓이는 공간은 어떤 관계에 가깝나요?',
    choices: buildContextualHintChoices(hintContext, 'relationship_space'),
  })
}

function buildRelationshipTimeQuestion(hintContext: ProjectHintContext) {
  return buildPersonaQuestion({
    intro: '좋아요. 마지막으로 집중과 휴식 시간을 어떻게 인식하는지 정리해볼게요.',
    question:
      '시간과의 관계: 사용자는 집중과 휴식 시간을 어떻게 인식하길 원하나요?',
    choices: buildContextualHintChoices(hintContext, 'relationship_time'),
  })
}

function extractKeywordSectionFromSummary(
  summary: string,
  labels: string[],
  boundaryLabels: string[]
) {
  const normalized = summary.replace(/\r\n/g, '\n')
  const escape = (value: string) =>
    value.replace(/[.*+?^${}()|[\]\\]/g, '\\$&')
  const boundary = boundaryLabels.map(escape).join('|')

  for (const label of labels) {
    const regex = new RegExp(
      `(?:^|\\n)\\s*(?:#{1,6}\\s*)?(?:\\*\\*)?${escape(
        label
      )}(?:\\*\\*)?\\s*\\n([\\s\\S]*?)(?=\\n\\s*(?:#{1,6}\\s*)?(?:\\*\\*)?(?:${boundary})(?:\\*\\*)?\\s*\\n|$)`,
      'i'
    )
    const match = normalized.match(regex)

    if (match?.[1]) {
      return splitKeywordCandidates(match[1])
    }
  }

  return []
}

function uniqueKeywords(keywords: string[]) {
  const seen = new Set<string>()
  const unique: string[] = []

  for (const keyword of keywords) {
    const shortKeyword = toShortKeyword(keyword, keyword)
    const normalized = shortKeyword.replace(/\s+/g, '').toLowerCase()

    if (!normalized || seen.has(normalized)) {
      continue
    }

    seen.add(normalized)
    unique.push(shortKeyword)
  }

  return unique
}

function buildExperienceIntroSentence(keywords: string[]) {
  const [context = '직관적인 사용 흐름', valueA = '신뢰감', valueB = '유연함', outcome = '지속 가능한 변화'] =
    keywords

  return `${context} 속에서, ${valueA}와 ${valueB}으로 ${outcome}을 경험하는 제품`
}

function buildExperienceKeywordsCardSummary(summary: string) {
  const sectionLabels = [
    '감정 Keywords (12)',
    '행동 Keywords (12)',
    '공간 Keywords (8)',
    '감정',
    '행동',
    '공간',
  ]
  const keywords = uniqueKeywords([
    ...extractKeywordSectionFromSummary(
      summary,
      ['감정 Keywords (12)', '감정'],
      sectionLabels
    ),
    ...extractKeywordSectionFromSummary(
      summary,
      ['행동 Keywords (12)', '행동'],
      sectionLabels
    ),
    ...extractKeywordSectionFromSummary(
      summary,
      ['공간 Keywords (8)', '공간'],
      sectionLabels
    ),
  ])
  const featured = keywords.slice(0, 6)
  const intro = buildExperienceIntroSentence(featured)
  const description =
    featured.length > 0
      ? `이 제품은 사용자가 자연스럽게 시작하고 흐름을 회복하며, 사용 후에는 안정감과 만족감을 느끼는 경험을 제공합니다. 공간 안에서는 부담 없이 놓이고 생활 흐름과 연결되는 방향을 지향하며, ${featured.join(
          ', '
        )} 같은 감각을 중심으로 제품 경험을 제안합니다.`
      : '이 제품은 사용자가 쉽게 시작하고 자연스럽게 몰입하며, 사용 후에는 긍정적인 감정과 안정된 리듬을 느끼는 경험을 지향합니다. 제품의 기능과 공간 속 존재감이 함께 연결되는 방향을 제안합니다.'

  return [
    '## Keywords: Experience',
    '',
    '**한줄 소개**',
    intro,
    '',
    '**내용**',
    description,
    '',
    '**Keywords**',
    `- ${keywords.join(', ')}`,
  ].join('\n')
}

function buildRelationshipKeywordsSummary(messages: NormalizedMessage[]) {
  const interruption = compactPersonaFlowValue(
    getAnswerAfterQuestion(
      messages,
      PERSONA_RELATIONSHIP_QUESTIONS.interruption
    ),
    '사용자 - 스마트폰: 유혹, 이탈, 알림'
  )
  const space = compactPersonaFlowValue(
    getAnswerAfterQuestion(messages, PERSONA_RELATIONSHIP_QUESTIONS.space),
    '사용자 - 책상: 몰입 공간, 개인화, 안정감'
  )
  const time = compactPersonaFlowValue(
    getAnswerAfterQuestion(messages, PERSONA_RELATIONSHIP_QUESTIONS.time),
    '사용자 - 시간: 흐름, 리듬, 전환'
  )
  const interruptionKeywords = buildShortKeywordList(
    interruption,
    ['거리두기', '이탈 방지', '방해 완화'],
    3
  )
  const spaceKeywords = buildShortKeywordList(
    space,
    ['공간 정돈', '몰입 공간', '안정감'],
    3
  )
  const timeKeywords = buildShortKeywordList(
    time,
    ['시간 인식', '리듬 전환', '예측 가능'],
    3
  )

  return [
    '## Keywords: Relationship',
    '',
    '**기존 방해 요소와의 관계**',
    `- ${interruptionKeywords.join(', ')}`,
    '',
    '**제품이 놓이는 공간과의 관계**',
    `- ${spaceKeywords.join(', ')}`,
    '',
    '**집중/휴식 시간과의 관계**',
    `- ${timeKeywords.join(', ')}`,
  ].join('\n')
}

function buildPersonaCompositeSummary({
  messages,
  relationshipSummary,
}: {
  messages: NormalizedMessage[]
  relationshipSummary: string
}) {
  const problemSummary = findLatestPersonaSummary(
    messages,
    /##\s*Problem Statements/i
  )
  const experienceSummary = findLatestPersonaSummary(
    messages,
    /##\s*Keywords:\s*Experience/i
  )
  const relationshipText =
    relationshipSummary ||
    findLatestPersonaSummary(messages, /##\s*Keywords:\s*Relationship/i)
  const problemLabels = [
    '01. Context',
    '02. Problems',
    '03. Needs',
    'Context',
    'Problems',
    'Needs',
    '문제(현재 상황)',
    '불편함',
  ]
  const experienceLabels = [
    '감정 Keywords (12)',
    '행동 Keywords (12)',
    '공간 Keywords (8)',
    '감정',
    '행동',
    '공간',
  ]
  const relationshipLabels = [
    '기존 방해 요소와의 관계',
    '제품이 놓이는 공간과의 관계',
    '집중/휴식 시간과의 관계',
  ]
  const situation = extractSummarySection(
    problemSummary,
    ['01. Context', 'Context', '문제(현재 상황)'],
    problemLabels
  )
  const discomfort = extractSummarySection(
    problemSummary,
    ['02. Problems', 'Problems', '불편함'],
    problemLabels
  )
  const needs = extractSummarySection(
    problemSummary,
    ['03. Needs', 'Needs'],
    problemLabels
  )
  const emotion = extractSummarySection(
    experienceSummary,
    ['감정 Keywords (12)', '감정'],
    experienceLabels
  )
  const behavior = extractSummarySection(
    experienceSummary,
    ['행동 Keywords (12)', '행동'],
    experienceLabels
  )
  const spaceValue = extractSummarySection(
    experienceSummary,
    ['공간 Keywords (8)', '공간'],
    experienceLabels
  )
  const interruptionRelation = extractSummarySection(relationshipText, [
    '기존 방해 요소와의 관계',
  ], relationshipLabels)
  const spaceRelation = extractSummarySection(relationshipText, [
    '제품이 놓이는 공간과의 관계',
  ], relationshipLabels)
  const timeRelation = extractSummarySection(relationshipText, [
    '집중/휴식 시간과의 관계',
  ], relationshipLabels)

  return [
    '## Persona Summary',
    '',
    '**Demographic Info**',
    '- 집중과 루틴 관리가 필요한 주요 사용자',
    '',
    '**Persona Story**',
    `- ${situation || '일상 속 몰입 전환이 필요한 상황을 반복적으로 경험함'}`,
    `- ${emotion || '평온함과 안정감을 기대함'}`,
    '',
    '**Problem & Needs**',
    `- ${discomfort || '기존 방식에서 반복적 방해 요소를 경험함'}`,
    `- ${needs || '재몰입과 자기조절을 돕는 해결책이 필요함'}`,
    '',
    '**Current Behavior**',
    `- ${behavior || '루틴 형성과 재몰입 행동을 만들고자 함'}`,
    '',
    '**Lifestyle Context**',
    `- ${spaceValue || '책상과 개인 공간 안에서 자연스럽게 사용됨'}`,
    '',
    '**Relationship Keyword**',
    `- ${interruptionRelation || '사용자 - 방해 요소: 유혹, 이탈, 알림'}`,
    `- ${spaceRelation || '사용자 - 공간: 몰입 공간, 안정감'}`,
    `- ${timeRelation || '사용자 - 시간: 흐름, 리듬, 전환'}`,
  ].join('\n')
}

function buildPersonaFlowVisualizationResponse({
  kind,
  summary,
  nextText,
  reason,
}: {
  kind: PersonaFlowArtifactKind
  summary: string
  nextText: string
  reason: string
}) {
  return new Response(
    [
      buildPersonaFlowCardBlock(kind, summary),
      `${PERSONA_FLOW_CARD_LABELS[kind]} 카드를 만들었습니다.`,
      '',
      nextText,
    ].join('\n'),
    {
      headers: {
        'Content-Type': 'text/plain; charset=utf-8',
        'x-aidee-current-stage': 'step_2_persona',
        'x-aidee-next-stage': 'step_2_persona',
        'x-aidee-transition': 'no',
        'x-aidee-reason': reason,
      },
    }
  )
}

function buildPersonaFlowResponse({
  messages,
  forceImageGeneration,
  lastUserMessage,
  project,
}: {
  messages: NormalizedMessage[]
  forceImageGeneration: ChatRequestBody['forceImageGeneration']
  lastUserMessage: string
  project: ProjectRecord | null
}) {
  const hintContext = buildProjectHintContext(project)

  if (forceImageGeneration === 'problem_statements_visualization') {
    const summary = extractSummaryFromVisualizationCommand(lastUserMessage)

    return buildPersonaFlowVisualizationResponse({
      kind: 'problem_statements',
      summary,
      nextText:
        'Problem Statements 카드를 확인해주세요. 수정할 내용이 있으면 수정하기를, 이대로 진행하려면 확정하기를 눌러주세요.',
      reason: 'problem_statements_visualized',
    })
  }

  if (forceImageGeneration === 'experience_keywords_visualization') {
    const summary = extractSummaryFromVisualizationCommand(lastUserMessage)

    return buildPersonaFlowVisualizationResponse({
      kind: 'experience_keywords',
      summary,
      nextText: buildRelationshipInterruptionQuestion(hintContext),
      reason: 'experience_keywords_visualized',
    })
  }

  if (forceImageGeneration === 'relationship_keywords_visualization') {
    const summary = extractSummaryFromVisualizationCommand(lastUserMessage)
    const personaSummary = buildPersonaCompositeSummary({
      messages,
      relationshipSummary: summary,
    })

    return buildPersonaFlowVisualizationResponse({
      kind: 'relationship_keywords',
      summary,
      nextText: personaSummary,
      reason: 'relationship_keywords_visualized',
    })
  }

  const hasProblemSummary = messages.some((message) =>
    /##\s*Problem Statements/i.test(message.content)
  )
  const hasExperienceSummary = messages.some((message) =>
    /##\s*Keywords:\s*Experience/i.test(message.content)
  )
  const hasRelationshipSummary = messages.some((message) =>
    /##\s*Keywords:\s*Relationship/i.test(message.content)
  )
  const hasPersonaSummary = messages.some((message) =>
    /##\s*Persona Summary/i.test(message.content)
  )
  const revisionRequestTarget =
    getVisualizationRevisionTargetFromText(lastUserMessage)
  const recentRevisionTarget = getRecentVisualizationRevisionTarget(messages)

  if (isPersonaVisualizationRevisionTarget(revisionRequestTarget)) {
    return buildStageTextResponse({
      text: buildVisualizationRevisionQuestion(revisionRequestTarget),
      stageKey: 'step_2_persona',
      reason: `${revisionRequestTarget}_revision_question`,
    })
  }

  if (
    isPersonaVisualizationRevisionTarget(recentRevisionTarget) &&
    lastUserMessage.trim()
  ) {
    return buildStageTextResponse({
      text: buildRevisedPersonaVisualizationSummary({
        target: recentRevisionTarget,
        messages,
        revisionText: lastUserMessage,
      }),
      stageKey: 'step_2_persona',
      reason: `${recentRevisionTarget}_revised_summary`,
    })
  }

  if (!hasProblemSummary) {
    if (!getAnswerAfterQuestion(messages, PERSONA_PROBLEM_QUESTIONS.triggerScene)) {
      return new Response(buildProblemSituationQuestion(hintContext), {
        headers: {
          'Content-Type': 'text/plain; charset=utf-8',
          'x-aidee-current-stage': 'step_2_persona',
          'x-aidee-next-stage': 'step_2_persona',
          'x-aidee-transition': 'no',
          'x-aidee-reason': 'problem_situation_question',
        },
      })
    }

    if (!getAnswerAfterQuestion(messages, PERSONA_PROBLEM_QUESTIONS.userAction)) {
      return new Response(buildProblemUserActionQuestion(hintContext), {
        headers: {
          'Content-Type': 'text/plain; charset=utf-8',
          'x-aidee-current-stage': 'step_2_persona',
          'x-aidee-next-stage': 'step_2_persona',
          'x-aidee-transition': 'no',
          'x-aidee-reason': 'problem_user_action_question',
        },
      })
    }

    if (!getAnswerAfterQuestion(messages, PERSONA_PROBLEM_QUESTIONS.interruption)) {
      return new Response(buildProblemInterruptionQuestion(hintContext), {
        headers: {
          'Content-Type': 'text/plain; charset=utf-8',
          'x-aidee-current-stage': 'step_2_persona',
          'x-aidee-next-stage': 'step_2_persona',
          'x-aidee-transition': 'no',
          'x-aidee-reason': 'problem_interruption_question',
        },
      })
    }

    if (!getAnswerAfterQuestion(messages, PERSONA_PROBLEM_QUESTIONS.workaround)) {
      return new Response(buildProblemWorkaroundQuestion(hintContext), {
        headers: {
          'Content-Type': 'text/plain; charset=utf-8',
          'x-aidee-current-stage': 'step_2_persona',
          'x-aidee-next-stage': 'step_2_persona',
          'x-aidee-transition': 'no',
          'x-aidee-reason': 'problem_workaround_question',
        },
      })
    }

    if (!getAnswerAfterQuestion(messages, PERSONA_PROBLEM_QUESTIONS.residue)) {
      return new Response(buildProblemResidueQuestion(hintContext), {
        headers: {
          'Content-Type': 'text/plain; charset=utf-8',
          'x-aidee-current-stage': 'step_2_persona',
          'x-aidee-next-stage': 'step_2_persona',
          'x-aidee-transition': 'no',
          'x-aidee-reason': 'problem_residue_question',
        },
      })
    }

    if (!getAnswerAfterQuestion(messages, PERSONA_PROBLEM_QUESTIONS.desiredChange)) {
      return new Response(buildProblemDesiredChangeQuestion(hintContext), {
        headers: {
          'Content-Type': 'text/plain; charset=utf-8',
          'x-aidee-current-stage': 'step_2_persona',
          'x-aidee-next-stage': 'step_2_persona',
          'x-aidee-transition': 'no',
          'x-aidee-reason': 'problem_desired_change_question',
        },
      })
    }

    return new Response(buildProblemStatementsSummary(messages), {
      headers: {
        'Content-Type': 'text/plain; charset=utf-8',
        'x-aidee-current-stage': 'step_2_persona',
        'x-aidee-next-stage': 'step_2_persona',
        'x-aidee-transition': 'no',
        'x-aidee-reason': 'problem_statements_summary',
      },
    })
  }

  const problemStatementsConfirmed =
    hasPersonaFlowCardConfirmation(messages, 'problem_statements') ||
    isProblemStatementsConfirmation(lastUserMessage)

  if (!hasPersonaFlowCard(messages, 'problem_statements')) {
    return new Response('먼저 Problem Statements 정리를 확인하고 확정한 뒤 시각화하기 버튼을 눌러 카드를 만들어주세요.', {
      headers: {
        'Content-Type': 'text/plain; charset=utf-8',
        'x-aidee-current-stage': 'step_2_persona',
        'x-aidee-next-stage': 'step_2_persona',
        'x-aidee-transition': 'no',
        'x-aidee-reason': 'awaiting_problem_statements_visualization',
      },
    })
  }

  if (!problemStatementsConfirmed) {
    if (isProblemStatementsRevisionRequest(lastUserMessage)) {
      return new Response(buildProblemStatementsRevisionQuestion(hintContext), {
        headers: {
          'Content-Type': 'text/plain; charset=utf-8',
          'x-aidee-current-stage': 'step_2_persona',
          'x-aidee-next-stage': 'step_2_persona',
          'x-aidee-transition': 'no',
          'x-aidee-reason': 'problem_statements_revision_question',
        },
      })
    }

    if (hasRecentProblemStatementsRevisionPrompt(messages) && lastUserMessage.trim()) {
      return new Response(
        buildRevisedProblemStatementsSummary(messages, lastUserMessage),
        {
          headers: {
            'Content-Type': 'text/plain; charset=utf-8',
            'x-aidee-current-stage': 'step_2_persona',
            'x-aidee-next-stage': 'step_2_persona',
            'x-aidee-transition': 'no',
            'x-aidee-reason': 'problem_statements_revised_summary',
          },
        }
      )
    }

    return new Response(
      'Problem Statements 카드를 확인해주세요. 수정할 내용이 있으면 수정하기를, 이대로 진행하려면 확정하기를 눌러주세요.',
      {
        headers: {
          'Content-Type': 'text/plain; charset=utf-8',
          'x-aidee-current-stage': 'step_2_persona',
          'x-aidee-next-stage': 'step_2_persona',
          'x-aidee-transition': 'no',
          'x-aidee-reason': 'awaiting_problem_statements_confirmation',
        },
      }
    )
  }

  if (!hasExperienceSummary) {
    if (
      !getAnswerAfterQuestion(
        messages,
        PERSONA_EXPERIENCE_QUESTIONS.emotionEntry
      )
    ) {
      return new Response(buildExperienceEmotionEntryQuestion(hintContext), {
        headers: {
          'Content-Type': 'text/plain; charset=utf-8',
          'x-aidee-current-stage': 'step_2_persona',
          'x-aidee-next-stage': 'step_2_persona',
          'x-aidee-transition': 'no',
          'x-aidee-reason': 'experience_emotion_entry_question',
        },
      })
    }

    if (
      !getAnswerAfterQuestion(
        messages,
        PERSONA_EXPERIENCE_QUESTIONS.emotionRelief
      )
    ) {
      return new Response(buildExperienceEmotionReliefQuestion(hintContext), {
        headers: {
          'Content-Type': 'text/plain; charset=utf-8',
          'x-aidee-current-stage': 'step_2_persona',
          'x-aidee-next-stage': 'step_2_persona',
          'x-aidee-transition': 'no',
          'x-aidee-reason': 'experience_emotion_relief_question',
        },
      })
    }

    if (
      !getAnswerAfterQuestion(
        messages,
        PERSONA_EXPERIENCE_QUESTIONS.emotionAfter
      )
    ) {
      return new Response(buildExperienceEmotionAfterQuestion(hintContext), {
        headers: {
          'Content-Type': 'text/plain; charset=utf-8',
          'x-aidee-current-stage': 'step_2_persona',
          'x-aidee-next-stage': 'step_2_persona',
          'x-aidee-transition': 'no',
          'x-aidee-reason': 'experience_emotion_after_question',
        },
      })
    }

    if (
      !getAnswerAfterQuestion(
        messages,
        PERSONA_EXPERIENCE_QUESTIONS.behaviorStart
      )
    ) {
      return new Response(buildExperienceBehaviorStartQuestion(hintContext), {
        headers: {
          'Content-Type': 'text/plain; charset=utf-8',
          'x-aidee-current-stage': 'step_2_persona',
          'x-aidee-next-stage': 'step_2_persona',
          'x-aidee-transition': 'no',
          'x-aidee-reason': 'experience_behavior_start_question',
        },
      })
    }

    if (
      !getAnswerAfterQuestion(
        messages,
        PERSONA_EXPERIENCE_QUESTIONS.behaviorRecovery
      )
    ) {
      return new Response(buildExperienceBehaviorRecoveryQuestion(hintContext), {
        headers: {
          'Content-Type': 'text/plain; charset=utf-8',
          'x-aidee-current-stage': 'step_2_persona',
          'x-aidee-next-stage': 'step_2_persona',
          'x-aidee-transition': 'no',
          'x-aidee-reason': 'experience_behavior_recovery_question',
        },
      })
    }

    if (
      !getAnswerAfterQuestion(
        messages,
        PERSONA_EXPERIENCE_QUESTIONS.behaviorRoutine
      )
    ) {
      return new Response(buildExperienceBehaviorRoutineQuestion(hintContext), {
        headers: {
          'Content-Type': 'text/plain; charset=utf-8',
          'x-aidee-current-stage': 'step_2_persona',
          'x-aidee-next-stage': 'step_2_persona',
          'x-aidee-transition': 'no',
          'x-aidee-reason': 'experience_behavior_routine_question',
        },
      })
    }

    if (
      !getAnswerAfterQuestion(
        messages,
        PERSONA_EXPERIENCE_QUESTIONS.spaceImpression
      )
    ) {
      return new Response(buildExperienceSpaceImpressionQuestion(hintContext), {
        headers: {
          'Content-Type': 'text/plain; charset=utf-8',
          'x-aidee-current-stage': 'step_2_persona',
          'x-aidee-next-stage': 'step_2_persona',
          'x-aidee-transition': 'no',
          'x-aidee-reason': 'experience_space_impression_question',
        },
      })
    }

    if (
      !getAnswerAfterQuestion(
        messages,
        PERSONA_EXPERIENCE_QUESTIONS.spaceMeaning
      )
    ) {
      return new Response(buildExperienceSpaceMeaningQuestion(hintContext), {
        headers: {
          'Content-Type': 'text/plain; charset=utf-8',
          'x-aidee-current-stage': 'step_2_persona',
          'x-aidee-next-stage': 'step_2_persona',
          'x-aidee-transition': 'no',
          'x-aidee-reason': 'experience_space_meaning_question',
        },
      })
    }

    if (
      !getAnswerAfterQuestion(
        messages,
        PERSONA_EXPERIENCE_QUESTIONS.spaceRelation
      )
    ) {
      return new Response(buildExperienceSpaceRelationQuestion(hintContext), {
        headers: {
          'Content-Type': 'text/plain; charset=utf-8',
          'x-aidee-current-stage': 'step_2_persona',
          'x-aidee-next-stage': 'step_2_persona',
          'x-aidee-transition': 'no',
          'x-aidee-reason': 'experience_space_relation_question',
        },
      })
    }

    return new Response(buildExperienceKeywordsSummary(messages), {
      headers: {
        'Content-Type': 'text/plain; charset=utf-8',
        'x-aidee-current-stage': 'step_2_persona',
        'x-aidee-next-stage': 'step_2_persona',
        'x-aidee-transition': 'no',
        'x-aidee-reason': 'experience_keywords_summary',
      },
    })
  }

  if (!hasPersonaFlowCard(messages, 'experience_keywords')) {
    return new Response('먼저 Keywords: Experience 정리를 확인하고 확정한 뒤 시각화하기 버튼을 눌러 카드를 만들어주세요.', {
      headers: {
        'Content-Type': 'text/plain; charset=utf-8',
        'x-aidee-current-stage': 'step_2_persona',
        'x-aidee-next-stage': 'step_2_persona',
        'x-aidee-transition': 'no',
        'x-aidee-reason': 'awaiting_experience_keywords_visualization',
      },
    })
  }

  if (!hasRelationshipSummary) {
    if (
      !getAnswerAfterQuestion(
        messages,
        PERSONA_RELATIONSHIP_QUESTIONS.interruption
      )
    ) {
      return new Response(buildRelationshipInterruptionQuestion(hintContext), {
        headers: {
          'Content-Type': 'text/plain; charset=utf-8',
          'x-aidee-current-stage': 'step_2_persona',
          'x-aidee-next-stage': 'step_2_persona',
          'x-aidee-transition': 'no',
          'x-aidee-reason': 'relationship_interruption_question',
        },
      })
    }

    if (!getAnswerAfterQuestion(messages, PERSONA_RELATIONSHIP_QUESTIONS.space)) {
      return new Response(buildRelationshipSpaceQuestion(hintContext), {
        headers: {
          'Content-Type': 'text/plain; charset=utf-8',
          'x-aidee-current-stage': 'step_2_persona',
          'x-aidee-next-stage': 'step_2_persona',
          'x-aidee-transition': 'no',
          'x-aidee-reason': 'relationship_space_question',
        },
      })
    }

    if (!getAnswerAfterQuestion(messages, PERSONA_RELATIONSHIP_QUESTIONS.time)) {
      return new Response(buildRelationshipTimeQuestion(hintContext), {
        headers: {
          'Content-Type': 'text/plain; charset=utf-8',
          'x-aidee-current-stage': 'step_2_persona',
          'x-aidee-next-stage': 'step_2_persona',
          'x-aidee-transition': 'no',
          'x-aidee-reason': 'relationship_time_question',
        },
      })
    }

    return new Response(buildRelationshipKeywordsSummary(messages), {
      headers: {
        'Content-Type': 'text/plain; charset=utf-8',
        'x-aidee-current-stage': 'step_2_persona',
        'x-aidee-next-stage': 'step_2_persona',
        'x-aidee-transition': 'no',
        'x-aidee-reason': 'relationship_keywords_summary',
      },
    })
  }

  if (!hasPersonaFlowCard(messages, 'relationship_keywords')) {
    return new Response('먼저 Keywords: Relationship 정리를 확인하고 확정한 뒤 시각화하기 버튼을 눌러 카드를 만들어주세요.', {
      headers: {
        'Content-Type': 'text/plain; charset=utf-8',
        'x-aidee-current-stage': 'step_2_persona',
        'x-aidee-next-stage': 'step_2_persona',
        'x-aidee-transition': 'no',
        'x-aidee-reason': 'awaiting_relationship_keywords_visualization',
      },
    })
  }

  if (!hasPersonaSummary) {
    return new Response(
      buildPersonaCompositeSummary({
        messages,
        relationshipSummary: '',
      }),
      {
        headers: {
          'Content-Type': 'text/plain; charset=utf-8',
          'x-aidee-current-stage': 'step_2_persona',
          'x-aidee-next-stage': 'step_2_persona',
          'x-aidee-transition': 'no',
          'x-aidee-reason': 'persona_summary_created',
        },
      }
    )
  }

  return null
}

const DIRECTION_RESEARCH_KINDS: DirectionResearchKind[] = [
  'market_size',
  'consumption_keywords',
  'brand_positioning',
]

const DIRECTION_CARD_LABELS: Record<DirectionResearchKind, string> = {
  market_size: 'Tam Sam Som',
  consumption_keywords: 'Keywords:Consumption',
  brand_positioning: 'Positioning Map: Brand',
}

const VISUALIZATION_REVISION_TARGET_LABELS: Record<
  VisualizationRevisionTarget,
  string
> = {
  problem_statements: 'Problem Statements',
  experience_keywords: 'Keywords: Experience',
  relationship_keywords: 'Keywords: Relationship',
  persona: 'Persona Summary',
  market_size: '시장 규모 리서치',
  consumption_keywords: '소비 트렌드 리서치',
  brand_positioning: '경쟁사 리서치',
  style_reference: '스타일 레퍼런스',
}

function isPersonaVisualizationRevisionTarget(
  target: VisualizationRevisionTarget | null
): target is PersonaFlowArtifactKind | 'persona' {
  return (
    target === 'problem_statements' ||
    target === 'experience_keywords' ||
    target === 'relationship_keywords' ||
    target === 'persona'
  )
}

function isDirectionVisualizationRevisionTarget(
  target: VisualizationRevisionTarget | null
): target is DirectionResearchKind {
  return (
    target === 'market_size' ||
    target === 'consumption_keywords' ||
    target === 'brand_positioning'
  )
}

function getVisualizationRevisionTargetFromText(
  text: string
): VisualizationRevisionTarget | null {
  if (!/^수정하기\s*:/i.test(text)) {
    return null
  }

  if (/Problem\s*Statements|문제\s*정리/i.test(text)) {
    return 'problem_statements'
  }

  if (/Keywords:\s*Experience|경험\s*키워드/i.test(text)) {
    return 'experience_keywords'
  }

  if (/Keywords:\s*Relationship|관계\s*키워드/i.test(text)) {
    return 'relationship_keywords'
  }

  if (/Persona\s*Summary|Persona\s*Card|페르소나/i.test(text)) {
    return 'persona'
  }

  if (/시장\s*규모|Tam\s*Sam\s*Som/i.test(text)) {
    return 'market_size'
  }

  if (/소비\s*트렌드|소비트렌드|Consumption/i.test(text)) {
    return 'consumption_keywords'
  }

  if (/경쟁사|Positioning\s*Map|Brand/i.test(text)) {
    return 'brand_positioning'
  }

  if (/스타일|레퍼런스|무드보드/i.test(text)) {
    return 'style_reference'
  }

  return null
}

function getRecentVisualizationRevisionTarget(
  messages: NormalizedMessage[]
): VisualizationRevisionTarget | null {
  const latestAssistant = messages
    .slice()
    .reverse()
    .find((message) => message.role === 'assistant')

  if (!latestAssistant || !/수정하고\s*싶은\s*방향/i.test(latestAssistant.content)) {
    return null
  }

  return (
    (Object.entries(VISUALIZATION_REVISION_TARGET_LABELS).find(([, label]) =>
      latestAssistant.content.includes(`${label} 수정`)
    )?.[0] as VisualizationRevisionTarget | undefined) ?? null
  )
}

function buildVisualizationRevisionQuestion(target: VisualizationRevisionTarget) {
  const label = VISUALIZATION_REVISION_TARGET_LABELS[target]

  return [
    `## ${label} 수정`,
    '',
    `${label}에서 수정하고 싶은 방향을 한 문장으로 입력해주세요.`,
    '입력한 내용을 반영해 다시 정리하고, 확인 후 확정할 수 있게 보여드릴게요.',
  ].join('\n')
}

function buildStageTextResponse({
  text,
  stageKey,
  reason,
}: {
  text: string
  stageKey: StageKey
  reason: string
}) {
  return new Response(text, {
    headers: {
      'Content-Type': 'text/plain; charset=utf-8',
      'x-aidee-current-stage': stageKey,
      'x-aidee-next-stage': stageKey,
      'x-aidee-transition': 'no',
      'x-aidee-reason': reason,
    },
  })
}

function findLatestVisualizationSummary(
  messages: NormalizedMessage[],
  headingPattern: RegExp
) {
  return (
    messages
      .map((message) => message.content)
      .slice()
      .reverse()
      .find(
        (content) =>
          headingPattern.test(content) && !/수정하고\s*싶은\s*방향/i.test(content)
      ) ?? ''
  )
}

function stripVisualizationConfirmationInstruction(text: string) {
  return text
    .split('\n')
    .filter(
      (line) =>
        !/내용을\s*확인한\s*뒤\s*확정하기를\s*누르면\s*시각화하기\s*버튼이\s*나타납니다/i.test(
          line
        )
    )
    .join('\n')
    .trim()
}

function formatRevisionDirection(text: string) {
  const cleaned = cleanSingleLineText(
    text.replace(/^수정하기\s*:\s*/i, '').replace(/^.+?내용을\s*수정하고\s*싶어요\.?/i, '')
  )

  if (!cleaned) {
    return '사용자가 요청한 수정 방향을 반영한다.'
  }

  if (/[.!?。！？]$/.test(cleaned)) {
    return cleaned
  }

  if (/(다|요|니다|습니다)$/.test(cleaned)) {
    return `${cleaned}.`
  }

  return `${cleaned} 방향으로 조정한다.`
}

function buildGenericRevisedVisualizationSummary({
  baseSummary,
  fallbackHeading,
  revisionText,
}: {
  baseSummary: string
  fallbackHeading: string
  revisionText: string
}) {
  const base = stripVisualizationConfirmationInstruction(baseSummary) || fallbackHeading

  return [
    base,
    '',
    '**수정 반영 방향**',
    `- ${formatRevisionDirection(revisionText)}`,
    '',
    '수정 내용을 반영했습니다.',
  ].join('\n')
}

function buildRevisedPersonaVisualizationSummary({
  target,
  messages,
  revisionText,
}: {
  target: PersonaFlowArtifactKind | 'persona'
  messages: NormalizedMessage[]
  revisionText: string
}) {
  if (target === 'problem_statements') {
    return buildRevisedProblemStatementsSummary(messages, revisionText)
  }

  if (target === 'experience_keywords') {
    return buildGenericRevisedVisualizationSummary({
      baseSummary: findLatestVisualizationSummary(
        messages,
        /##\s*Keywords:\s*Experience/i
      ),
      fallbackHeading: '## Keywords: Experience',
      revisionText,
    })
  }

  if (target === 'relationship_keywords') {
    return buildGenericRevisedVisualizationSummary({
      baseSummary: findLatestVisualizationSummary(
        messages,
        /##\s*Keywords:\s*Relationship/i
      ),
      fallbackHeading: '## Keywords: Relationship',
      revisionText,
    })
  }

  return buildGenericRevisedVisualizationSummary({
    baseSummary: findLatestVisualizationSummary(messages, /##\s*Persona Summary/i),
    fallbackHeading: '## Persona Summary',
    revisionText,
  })
}

function buildRevisedDirectionVisualizationSummary({
  target,
  messages,
  revisionText,
}: {
  target: DirectionResearchKind
  messages: NormalizedMessage[]
  revisionText: string
}) {
  const headingPattern =
    target === 'market_size'
      ? /##\s*시장\s*규모\s*리서치/i
      : target === 'consumption_keywords'
        ? /##\s*소비\s*트렌드\s*리서치/i
        : /##\s*경쟁사\s*리서치/i
  const fallbackHeading =
    target === 'market_size'
      ? '## 시장 규모 리서치'
      : target === 'consumption_keywords'
        ? '## 소비 트렌드 리서치'
        : '## 경쟁사 리서치'

  return buildGenericRevisedVisualizationSummary({
    baseSummary: findLatestVisualizationSummary(messages, headingPattern),
    fallbackHeading,
    revisionText,
  })
}

function buildRevisedStyleReferenceSummary({
  messages,
  revisionText,
}: {
  messages: NormalizedMessage[]
  revisionText: string
}) {
  return buildGenericRevisedVisualizationSummary({
    baseSummary: findLatestVisualizationSummary(
      messages,
      /##\s*선택한\s*스타일\s*레퍼런스/i
    ),
    fallbackHeading: '## 선택한 스타일 레퍼런스',
    revisionText,
  })
}

function getDirectionResearchKindFromText(text: string): DirectionResearchKind | null {
  if (/시장\s*규모|market|tam|sam|som/i.test(text)) {
    return 'market_size'
  }

  if (/소비\s*트렌드|소비트렌드|consumption|trend|키워드/i.test(text)) {
    return 'consumption_keywords'
  }

  if (/경쟁사|경쟁\s*브랜드|positioning|포지셔닝|brand/i.test(text)) {
    return 'brand_positioning'
  }

  return null
}

function getDirectionResearchKindFromForce(
  forceImageGeneration: ChatRequestBody['forceImageGeneration']
): DirectionResearchKind | null {
  switch (forceImageGeneration) {
    case 'market_size_visualization':
      return 'market_size'
    case 'consumption_keywords_visualization':
      return 'consumption_keywords'
    case 'brand_positioning_visualization':
      return 'brand_positioning'
    default:
      return null
  }
}

function hasDirectionCard(
  messages: NormalizedMessage[],
  kind: DirectionResearchKind
) {
  return messages.some((message) =>
    message.content.includes(`<<AIDEE_DIRECTION_CARD:${kind}>>`)
  )
}

function extractDirectionSummaryFromVisualizationCommand(text: string) {
  return text
    .replace(/^시각화하기\s*/i, '')
    .replace(/<<AIDEE_DIRECTION_CARD:[\s\S]*?<<\/AIDEE_DIRECTION_CARD>>/g, '')
    .split('\n')
    .filter(
      (line) =>
        !/아래의 시각화하기 버튼|내용을\s*확인한\s*뒤\s*확정하기를\s*누르면\s*시각화하기\s*버튼이\s*나타납니다/.test(
          line
        )
    )
    .join('\n')
    .trim()
}

function buildDirectionWidgetsResponse() {
  return [
    '<<AIDEE_DIRECTION_WIDGETS>>',
    '<</AIDEE_DIRECTION_WIDGETS>>',
    '',
    '## 디자인/개발 방향성 리서치',
    '',
    '이 단계에서는 시장을 세 가지 관점으로 나누어 확인합니다.',
    '아래 위젯 중 하나를 누르면 해당 항목부터 리서치해드릴게요.',
  ].join('\n')
}

function buildDirectionResearchText({
  kind,
  project,
}: {
  kind: DirectionResearchKind
  project: ProjectRecord | null
}) {
  const snapshot = buildProjectStartSnapshot(project)
  const projectLabel = snapshot.title || '현재 제품'
  const category = snapshot.category || '제품 카테고리 미정'
  const goal = snapshot.goal || '프로젝트 목표 미정'
  const usage = snapshot.usage || '활용 목적 미정'

  if (kind === 'market_size') {
    return [
      '## 시장 규모 리서치',
      '',
      `**분석 대상**`,
      `- 프로젝트: ${projectLabel}`,
      `- 카테고리: ${category}`,
      `- 목표: ${goal}`,
      '',
      '**1. TAM**',
      `- ${category}와 인접한 전체 문제 해결 시장을 기준으로 봅니다.`,
      '- 사용자가 겪는 문제를 해결할 수 있는 전체 대체재와 서비스까지 포함합니다.',
      '',
      '**2. SAM**',
      `- ${usage} 목적에 직접 연결되는 사용 상황과 구매 가능 고객군으로 좁힙니다.`,
      '- 실제 제품 형태, 가격대, 유통 가능성을 고려한 접근 가능 시장입니다.',
      '',
      '**3. SOM**',
      '- 초기 제품이 현실적으로 확보할 수 있는 작은 진입 시장입니다.',
      '- 첫 출시에서는 반복적으로 같은 문제를 겪고 구매 이유가 분명한 사용자부터 잡는 것이 적합합니다.',
      '',
      '**초기 판단**',
      '- 전체 시장보다 반복 사용 맥락이 선명한 세그먼트를 먼저 잡아야 합니다.',
      '- 시장 규모는 넓게 보되, 첫 제품은 작은 강한 니즈에서 출발하는 방향이 좋습니다.',
    ].join('\n')
  }

  if (kind === 'consumption_keywords') {
    return [
      '## 소비 트렌드 리서치',
      '',
      `**분석 대상**`,
      `- 프로젝트: ${projectLabel}`,
      `- 카테고리: ${category}`,
      '',
      '**1. 구매 동기**',
      '- 사용자는 기능 자체보다 자신의 루틴, 감정, 공간을 더 잘 관리할 수 있다는 기대에 반응합니다.',
      '- 즉각적인 효용과 장기적으로 쌓이는 습관 가치가 함께 설득 포인트가 됩니다.',
      '',
      '**2. 소비 키워드**',
      '- 루틴화: 반복 사용을 통해 생활 패턴에 들어오는 제품',
      '- 자기조절: 사용자가 스스로 상태를 조절하고 있다는 감각',
      '- 조용한 효용: 과시보다 일상에 자연스럽게 녹아드는 도움',
      '- 공간 정돈감: 제품이 놓이는 자리 자체가 정돈된 분위기를 만드는 가치',
      '',
      '**3. 디자인/개발 시사점**',
      '- 기능 설명보다 사용 후 변화가 먼저 느껴져야 합니다.',
      '- 사용 부담이 낮고, 매일 반복해도 피로하지 않은 인터랙션이 중요합니다.',
    ].join('\n')
  }

  return [
    '## 경쟁사 리서치',
    '',
    `**분석 대상**`,
    `- 프로젝트: ${projectLabel}`,
    `- 카테고리: ${category}`,
    '',
    '**1. 직접 경쟁군**',
    '- 같은 문제를 같은 제품 카테고리 안에서 해결하는 브랜드입니다.',
    '- 기능, 가격, 사용성, 형태 완성도가 직접 비교 기준이 됩니다.',
    '',
    '**2. 간접 경쟁군**',
    '- 앱, 생활 도구, 가구/오브제, 기존 습관처럼 같은 문제를 다른 방식으로 해결하는 대안입니다.',
    '- 사용자가 이미 익숙하게 쓰는 방식이라 전환 장벽을 함께 봐야 합니다.',
    '',
    '**3. 포지셔닝 축 제안**',
    '- X축: 기능 중심 ↔ 감성/공간 중심',
    '- Y축: 고관여/전문적 사용 ↔ 일상적/가벼운 사용',
    '- 현재 제품은 기능의 명확함과 공간 속 조용한 존재감을 함께 가진 영역을 노릴 수 있습니다.',
    '',
    '**초기 차별화 방향**',
    '- 기존 대안보다 사용 진입 장벽을 낮추고, 제품을 계속 쓰게 만드는 경험적 이유를 강화해야 합니다.',
    '- 경쟁사 대비 “매일 쓰기 쉬운 루틴 도구”라는 포지션이 유효합니다.',
  ].join('\n')
}

function buildDirectionCardBlock(kind: DirectionResearchKind, summary: string) {
  return [
    `<<AIDEE_DIRECTION_CARD:${kind}>>`,
    summary.trim(),
    '<</AIDEE_DIRECTION_CARD>>',
  ].join('\n')
}

function buildDirectionVisualizationResponse({
  kind,
  messages,
  summary,
}: {
  kind: DirectionResearchKind
  messages: NormalizedMessage[]
  summary: string
}) {
  const completedKinds = new Set<DirectionResearchKind>(
    DIRECTION_RESEARCH_KINDS.filter((candidate) =>
      candidate === kind ? true : hasDirectionCard(messages, candidate)
    )
  )
  const isComplete = completedKinds.size === DIRECTION_RESEARCH_KINDS.length
  const baseText = [
    buildDirectionCardBlock(kind, summary),
    `${DIRECTION_CARD_LABELS[kind]} 카드를 만들었습니다.`,
    '',
    isComplete
      ? '시장 규모, 소비 트렌드, 경쟁사 분석 카드가 모두 정리되었습니다.'
      : '다른 분석 위젯도 이어서 눌러볼 수 있습니다.',
  ].join('\n')
  const finalText = isComplete
    ? appendStageTransitionPrompt(baseText, 'step_4_style')
    : baseText

  return new Response(finalText, {
    headers: {
      'Content-Type': 'text/plain; charset=utf-8',
      'x-aidee-current-stage': 'step_3_direction',
      'x-aidee-next-stage': isComplete ? 'step_4_style' : 'step_3_direction',
      'x-aidee-transition': 'no',
      'x-aidee-reason': `${kind}_visualized`,
    },
  })
}

function buildDirectionFlowResponse({
  messages,
  forceImageGeneration,
  lastUserMessage,
  project,
}: {
  messages: NormalizedMessage[]
  forceImageGeneration: ChatRequestBody['forceImageGeneration']
  lastUserMessage: string
  project: ProjectRecord | null
}) {
  const forceKind = getDirectionResearchKindFromForce(forceImageGeneration)

  if (forceKind) {
    return buildDirectionVisualizationResponse({
      kind: forceKind,
      messages,
      summary: extractDirectionSummaryFromVisualizationCommand(lastUserMessage),
    })
  }

  const revisionRequestTarget =
    getVisualizationRevisionTargetFromText(lastUserMessage)
  const recentRevisionTarget = getRecentVisualizationRevisionTarget(messages)

  if (isDirectionVisualizationRevisionTarget(revisionRequestTarget)) {
    return buildStageTextResponse({
      text: buildVisualizationRevisionQuestion(revisionRequestTarget),
      stageKey: 'step_3_direction',
      reason: `${revisionRequestTarget}_revision_question`,
    })
  }

  if (
    isDirectionVisualizationRevisionTarget(recentRevisionTarget) &&
    lastUserMessage.trim()
  ) {
    return buildStageTextResponse({
      text: buildRevisedDirectionVisualizationSummary({
        target: recentRevisionTarget,
        messages,
        revisionText: lastUserMessage,
      }),
      stageKey: 'step_3_direction',
      reason: `${recentRevisionTarget}_revised_summary`,
    })
  }

  const requestedKind = getDirectionResearchKindFromText(lastUserMessage)

  if (requestedKind) {
    return new Response(
      buildDirectionResearchText({
        kind: requestedKind,
        project,
      }),
      {
        headers: {
          'Content-Type': 'text/plain; charset=utf-8',
          'x-aidee-current-stage': 'step_3_direction',
          'x-aidee-next-stage': 'step_3_direction',
          'x-aidee-transition': 'no',
          'x-aidee-reason': `${requestedKind}_research`,
        },
      }
    )
  }

  return new Response(buildDirectionWidgetsResponse(), {
    headers: {
      'Content-Type': 'text/plain; charset=utf-8',
      'x-aidee-current-stage': 'step_3_direction',
      'x-aidee-next-stage': 'step_3_direction',
      'x-aidee-transition': 'no',
      'x-aidee-reason': 'direction_widgets',
    },
  })
}

async function generateStyleReferenceImages({
  project,
  referenceImages,
  conversation,
  apiKey,
}: {
  project: ProjectRecord | null
  referenceImages: ReferenceImageRecord[]
  conversation: string
  apiKey: string
}) {
  const attempts = [
    {
      label: 'full',
      prompt: buildStyleReferencePrompt({
        project,
        referenceImages,
        conversation,
      }),
      count: 3,
    },
    {
      label: 'compact',
      prompt: buildCompactStyleReferencePrompt({
        project,
        referenceImages,
      }),
      count: 3,
    },
  ]
  let bestPartialPayload: GeneratedImageBlock | null = null

  for (const attempt of attempts) {
    console.log('[style-images] attempt start', {
      label: attempt.label,
      promptLength: attempt.prompt.length,
      count: attempt.count,
      projectTitle: project?.title || 'Untitled project',
      referenceCount: referenceImages.length,
    })

    try {
      const payload = await generateNanoBananaImages({
        prompt: attempt.prompt,
        count: attempt.count,
        apiKey,
      })
      payload.purpose = 'style_reference'

      console.log('[style-images] attempt success', {
        label: attempt.label,
        imageCount: payload.images.length,
        model: payload.model,
      })

      if (payload.images.length >= 3) {
        return payload
      }

      if (
        payload.images.length > 0 &&
        (!bestPartialPayload ||
          payload.images.length > bestPartialPayload.images.length)
      ) {
        bestPartialPayload = payload
      }

      console.warn('[style-images] partial result ignored', {
        label: attempt.label,
        imageCount: payload.images.length,
      })
    } catch (error) {
      console.error('[style-images] attempt failed', {
        label: attempt.label,
        error,
      })
    }
  }

  return bestPartialPayload
}

function hasStyleReferenceSelection(text: string) {
  return /([1-3])\s*번|이미지\s*([1-3])|레퍼런스\s*([1-3])|첫\s*번째|두\s*번째|세\s*번째|1\s*번째|2\s*번째|3\s*번째|선택|확정/.test(
    text
  )
}

function hasStyleKeywordSelection(text: string) {
  return /스타일\s*키워드\s*선택\s*완료|감정\s*키워드|색감\s*키워드|형태\s*키워드|촉감\s*키워드/.test(
    text
  )
}

function extractSelectedStyleKeywords(text: string) {
  return text
    .replace(/^스타일\s*키워드\s*선택\s*완료\s*/i, '')
    .trim()
}

function buildStyleKeywordPickerResponse() {
  return [
    '<<AIDEE_STYLE_KEYWORD_PICKER>>',
    '<</AIDEE_STYLE_KEYWORD_PICKER>>',
    '',
    '## 스타일 키워드 선택',
    '',
    '감정, 색감, 형태, 촉감 키워드를 선택해 스타일 컨셉의 기준을 잡아볼게요.',
    '각 항목에서 최대 5개까지 선택할 수 있습니다.',
  ].join('\n')
}

function buildStyleReferenceProposal(lastUserMessage: string) {
  const selectedMatch = lastUserMessage.match(
    /스타일\s*레퍼런스\s*([1-3])번\s*선택/i
  )
  const selectedIndex = selectedMatch?.[1] ?? '1'

  return [
    '## 선택한 스타일 레퍼런스',
    '',
    `**선택한 분위기**`,
    `- 스타일 위젯 ${selectedIndex}번`,
    '',
    '**스타일 해석**',
    '- 선택한 이미지는 제품의 감정, 색감, 형태, 촉감 키워드를 하나의 시각 언어로 묶는 방향입니다.',
    '- 전체 인상은 과하게 장식적이기보다 사용자가 매일 부담 없이 받아들일 수 있는 균형감에 가깝습니다.',
    '',
    '**색감 방향**',
    '- 주조색은 차분하고 안정적인 톤을 중심으로 두고, 포인트 컬러는 기능적 신호나 리듬감을 줄 때만 제한적으로 사용합니다.',
    '',
    '**형태 방향**',
    '- 형태는 한눈에 역할을 이해할 수 있도록 단순하게 잡고, 모서리와 비례에서 부드러운 사용감을 전달합니다.',
    '',
    '**촉감/소재 방향**',
    '- 표면은 손이 자주 닿아도 피로하지 않은 매트하거나 은은한 질감을 우선합니다.',
    '- 제품이 놓이는 공간 안에서 조용하지만 분명한 존재감을 갖도록 CMF를 정리합니다.',
  ].join('\n')
}

async function buildStyleReferenceOptionsResponse({
  project,
  referenceImages,
  selectedKeywords,
  apiKey,
}: {
  project: ProjectRecord | null
  referenceImages: ReferenceImageRecord[]
  selectedKeywords: string
  apiKey: string
}) {
  const generatedImagePayload = await generateNanoBananaImages({
    prompt: buildStyleKeywordReferencePrompt({
      project,
      referenceImages,
      selectedKeywords,
    }),
    count: 3,
    apiKey,
  })
  generatedImagePayload.purpose = 'style_reference'

  return new Response(
    appendGeneratedImagesBlock({
      text: [
        '선택한 키워드를 바탕으로 스타일 분위기 3가지를 제안합니다.',
        '이미지가 보이는 위젯 중 가장 가까운 방향 하나를 선택해주세요.',
      ].join('\n'),
      payload: generatedImagePayload,
    }),
    {
      headers: {
        'Content-Type': 'text/plain; charset=utf-8',
        'x-aidee-current-stage': 'step_4_style',
        'x-aidee-next-stage': 'step_4_style',
        'x-aidee-transition': 'no',
        'x-aidee-reason': 'style_reference_options_generated',
      },
    }
  )
}

async function buildStyleMoodboardResponse({
  project,
  referenceImages,
  styleProposal,
  apiKey,
}: {
  project: ProjectRecord | null
  referenceImages: ReferenceImageRecord[]
  styleProposal: string
  apiKey: string
}) {
  const generatedImagePayload = await generateNanoBananaImages({
    prompt: buildMoodboardPrompt({
      project,
      referenceImages,
      styleProposal,
    }),
    count: 1,
    apiKey,
  })
  generatedImagePayload.purpose = 'moodboard'

  return new Response(
    appendGeneratedImagesBlock({
      text: appendStageTransitionPrompt(
        '선택한 스타일 레퍼런스를 기준으로 무드보드를 생성했습니다.',
        'step_5_design'
      ),
      payload: generatedImagePayload,
    }),
    {
      headers: {
        'Content-Type': 'text/plain; charset=utf-8',
        'x-aidee-current-stage': 'step_4_style',
        'x-aidee-next-stage': 'step_5_design',
        'x-aidee-transition': 'no',
        'x-aidee-reason': 'style_moodboard_generated',
      },
    }
  )
}

async function buildStyleFlowResponse({
  project,
  referenceImages,
  messages,
  lastUserMessage,
  forceImageGeneration,
  apiKey,
}: {
  project: ProjectRecord | null
  referenceImages: ReferenceImageRecord[]
  messages: NormalizedMessage[]
  lastUserMessage: string
  forceImageGeneration: ChatRequestBody['forceImageGeneration']
  apiKey: string
}) {
  if (forceImageGeneration === 'style_moodboard_visualization') {
    return buildStyleMoodboardResponse({
      project,
      referenceImages,
      styleProposal: lastUserMessage,
      apiKey,
    })
  }

  const revisionRequestTarget =
    getVisualizationRevisionTargetFromText(lastUserMessage)
  const recentRevisionTarget = getRecentVisualizationRevisionTarget(messages)

  if (revisionRequestTarget === 'style_reference') {
    return buildStageTextResponse({
      text: buildVisualizationRevisionQuestion(revisionRequestTarget),
      stageKey: 'step_4_style',
      reason: 'style_reference_revision_question',
    })
  }

  if (recentRevisionTarget === 'style_reference' && lastUserMessage.trim()) {
    return buildStageTextResponse({
      text: buildRevisedStyleReferenceSummary({
        messages,
        revisionText: lastUserMessage,
      }),
      stageKey: 'step_4_style',
      reason: 'style_reference_revised_summary',
    })
  }

  if (
    forceImageGeneration === 'style_reference_options' ||
    hasStyleKeywordSelection(lastUserMessage)
  ) {
    return buildStyleReferenceOptionsResponse({
      project,
      referenceImages,
      selectedKeywords: extractSelectedStyleKeywords(lastUserMessage),
      apiKey,
    })
  }

  if (hasStyleReferenceSelection(lastUserMessage)) {
    return new Response(buildStyleReferenceProposal(lastUserMessage), {
      headers: {
        'Content-Type': 'text/plain; charset=utf-8',
        'x-aidee-current-stage': 'step_4_style',
        'x-aidee-next-stage': 'step_4_style',
        'x-aidee-transition': 'no',
        'x-aidee-reason': 'style_reference_selected',
      },
    })
  }

  return new Response(buildStyleKeywordPickerResponse(), {
    headers: {
      'Content-Type': 'text/plain; charset=utf-8',
      'x-aidee-current-stage': 'step_4_style',
      'x-aidee-next-stage': 'step_4_style',
      'x-aidee-transition': 'no',
      'x-aidee-reason': 'style_keyword_picker',
    },
  })
}

function isDesignRevisionRequest(text: string) {
  return /수정|바꿔|변경|조정|다듬|발전|고도화|대안|새로|다시|재생성|추가|더\s*보여|비교/i.test(
    text
  )
}

function hasDesignFinalSelection(text: string) {
  if (!text.trim() || isDesignRevisionRequest(text)) {
    return false
  }

  return [
    /(?:디자인|시안|렌더|안)\s*(?:을|으로|은|는)?\s*(?:확정|최종|진행|좋아|좋습니다|갈게|가겠습니다|할게|하겠습니다)/i,
    /(?:이|그)\s*(?:디자인|시안|렌더|안)\s*(?:으로|이|가)?\s*(?:확정|최종|진행|좋아|좋습니다)/i,
    /(?:[1-3]|첫|두|세)\s*(?:번|번째)?\s*(?:시안|안|이미지)?\s*(?:으로|을|가)?\s*(?:확정|최종|진행|갈게|가겠습니다|좋아|좋습니다|할게|하겠습니다)/i,
    /(?:A|a)[.)]?\s*(?:이\s*)?안\s*확정/i,
    /최종\s*확정/i,
  ].some((pattern) => pattern.test(text))
}

function buildInitialPrompt(project: ProjectRecord | null) {
  const title = project?.title || '새 프로젝트'

  return [
    `${title} 프로젝트가 방금 생성되었습니다.`,
    '저장된 프로젝트 정보(requirements)를 바탕으로 Project Direction 고정 템플릿을 출력하고, 아이디어 텍스트는 한국어 100단어 이내 문장으로 정리한 뒤 제품의 구체적인 모습이나 추가 설명이 있는지 질문하세요.',
    'Project Direction의 고정 표시 라벨은 Target Timeline, Project Scope, Key Features입니다. 제품 카테고리와 최소 예산은 별도 소제목 없이 보여주세요.',
    '답변은 한국어로 작성하세요.',
    '이 턴에서는 전체 프로세스 단계 설명을 하지 마세요.',
  ].join('\n')
}

function buildExpertCallPrompt(expert: ExpertKey) {
  const expertLabel = getExpertDefinition(expert).label

  return [
    `[전문가 호출: ${expertLabel}]`,
    '현재까지의 대화 내용과 프로젝트 정보를 바탕으로 답변하세요.',
    '새로운 질문을 임의로 만들지 말고, 지금 시점에서 이 전문가 관점으로 판단해야 할 내용을 정리하세요.',
    '전체 STEP 흐름은 유지하되, 이 답변만큼은 선택된 전문가의 관점이 분명히 드러나야 합니다.',
  ].join('\n')
}

function modelPromisedRfpWithoutDocument(text: string) {
  return (
    /RFP|제안요청서|문서\s*생성/i.test(text) &&
    (/잠시(?:만)?\s*(?:기다려|기다려\s*주세요|기다려주세요)/i.test(text) ||
      /생성\s*중/i.test(text) ||
      /생성해\s*드리겠습니다/i.test(text) ||
      /넘어가겠습니다/i.test(text)) &&
    !text.includes('# 제품 제안요청서') &&
    !text.includes('## 1. 프로젝트 개요')
  )
}

function resolveIntentStageKey({
  currentStageKey,
  lastUserMessage: _lastUserMessage,
}: {
  currentStageKey: StageKey
  lastUserMessage: string
}): StageKey {
  void _lastUserMessage
  return currentStageKey
}

function getStageSpecificInstruction(currentStageKey: StageKey) {
  switch (currentStageKey) {
    case 'step_0_start':
      return `
[현재 단계 운영]
- 지금은 STEP 0 프로젝트 시작 공통 확인 구간입니다.
- 프로젝트 생성 직후에는 먼저 Project Direction 고정 템플릿으로 저장된 정보를 기준점으로 정리하고, 제품의 구체적인 모습이나 추가 설명이 있는지 질문 1개로 끝내세요.
- Project Direction의 고정 표시 라벨은 Target Timeline, Project Scope, Key Features입니다. 제품 카테고리와 최소 예산은 별도 소제목 없이 보여주고, 아이디어 텍스트는 한국어 100단어 이내 문장으로 정리하세요.
- 사용자의 추가 입력을 받은 뒤에는 전체 내용(프로젝트 목표, 제품 카테고리, 예산/기간 범위, 최종 활용 목적)을 정리하고, 프로세스 확인하기 버튼을 안내하세요.
- 사용자가 프로세스 확인하기를 선택하기 전에는 전체 프로세스 1~7단계를 설명하지 마세요.
- 전체 톤은 친절하고 차분하게 유지하세요.
- STEP 0에서는 Persona Card를 절대 출력하지 마세요.
`.trim()
    case 'step_1_idea':
      return `
[현재 단계 운영]
- 지금은 STEP 1입니다.
- 제품 아이디어와 개발 조건을 친절하게 정리하세요.
- STEP 1 확정 조건이 충족되면 즉시 다음 단계로 이동하지 말고, STEP 2로 넘어간다는 안내와 이 단계에서 할 일 1줄을 말한 뒤 진행할지 물어보세요.
- STEP 1에서는 Persona Card를 절대 출력하지 마세요. Persona Card는 STEP 2 전용 산출물입니다.
`.trim()
    case 'step_2_persona':
      return `
[현재 단계 운영]
- 지금은 STEP 2 사용자 명확화 단계입니다.
- 이 단계는 Problem Statements → Keywords: Experience → Keywords: Relationship → Persona Summary → Persona Card 순서로 진행합니다.
- Problem Statements는 현재 상황, 불편함, 근본적 니즈를 직접 묻지 말고 여러 사용 장면 질문을 통해 도출합니다.
- Problem Statements 카드는 01. Context, 02. Problems, 03. Needs로 구성하고, 각 항목은 소제목 1개와 약 2줄 설명으로 작성합니다.
- Problem Statements의 소제목은 설명 내용을 짧게 요약한 완결된 문구여야 하며, 설명과 같은 문장을 반복하거나 중간에서 끊긴 문구로 쓰지 않습니다.
- Problem Statements 카드가 만들어진 뒤에는 사용자가 확정하기를 누르기 전까지 다음 질문으로 넘어가지 않습니다.
- Keywords: Experience는 감정, 행동, 공간 관점에서 각각 1개씩만 묻지 말고 총 9개 질문을 이어간 뒤, 감정 12개 + 행동 12개 + 공간 8개 = 총 32개 키워드로 정리합니다.
- Keywords: Experience와 Keywords: Relationship에 정리되는 각 키워드는 최대 10글자 이내의 짧은 명사형/구 형태로 작성합니다.
- Keywords: Experience 질문을 사용자에게 보여줄 때는 "감정 1", "행동 2", "공간 3" 같은 관점/번호 라벨을 질문 앞에 붙이지 않습니다.
- Keywords: Experience 요약에는 키워드 정리가 완료되었다는 문구만 출력하고, 버튼 안내 문장은 본문에 쓰지 않습니다.
- 한 번에 질문은 1개만 합니다.
- 각 묶음의 텍스트 정리 뒤에는 수정하기와 확정하기 버튼을 먼저 제공하고, 사용자가 확정하기를 누른 뒤 나타나는 시각화하기 버튼을 눌렀을 때만 다음 묶음으로 이어갑니다.
- Persona Card에는 Demographic Info, Persona Story, Problem & Needs, Current Behavior, Lifestyle Context, Relationship Keyword가 들어갑니다.
`.trim()
    case 'step_2_research':
      return `
[현재 단계 운영]
- 지금은 STEP 2의 리서치 단계입니다.
- 반드시 리서치 출력 템플릿에 맞춘 결과를 작성하세요.
- 리서치 결과가 끝나면 STEP 3으로 넘어갈 수 있도록 스타일/기능 방향을 여는 질문 1개로 마무리하세요.
`.trim()
    case 'step_3_direction':
      return `
[현재 단계 운영]
- 지금은 STEP 3 디자인/개발 방향성 도출 단계입니다.
- 이 단계는 1. 시장 규모, 2. 소비 트렌드, 3. 경쟁사 위젯으로 나누어 진행합니다.
- 사용자가 위젯을 누르면 해당 항목에 대해 텍스트 리서치를 제공합니다.
- 각 리서치 아래에는 수정하기와 확정하기 버튼이 먼저 제공되고, 확정 후 시각화하기 버튼이 제공됩니다.
- 시장 규모는 Tam Sam Som 카드, 소비 트렌드는 Keywords:Consumption 카드, 경쟁사는 Positioning Map: Brand 카드로 시각화합니다.
- 세 카드가 모두 만들어진 뒤에만 STEP 4 스타일 컨셉 도출 단계로 넘어갈지 묻습니다.
`.trim()
    case 'step_4_style':
      return `
[현재 단계 운영]
- 지금은 STEP 4 스타일 컨셉 도출 단계입니다.
- 질문형 대화가 아니라 감정, 색감, 형태, 촉감 키워드 선택 위젯으로 시작합니다.
- 각 항목에는 30개 키워드를 제시하고 사용자는 최대 5개까지 선택합니다.
- 선택 키워드를 종합하여 이미지가 보이는 스타일 분위기 위젯 3개를 제안합니다.
- 사용자가 스타일 위젯 1개를 선택하면 선택한 스타일 레퍼런스를 텍스트로 정리합니다.
- 텍스트 아래의 수정하기와 확정하기 버튼을 먼저 제공하고, 확정 후 시각화하기 버튼으로 무드보드를 생성한 뒤 STEP 5로 넘어갈지 묻습니다.
`.trim()
    case 'step_5_design':
    return `
[현재 단계 운영]
- 지금은 STEP 5 디자인 제안 단계입니다.
- STEP 4에서 선택한 스타일 레퍼런스를 기준으로 디자인 시안을 제안하세요.
- STEP 5에서 4개짜리 초기 디자인 시안 세트는 최대 1회만 생성합니다.
- 이미 디자인 시안 이미지가 대화에 있으면 새 4개 세트를 다시 만들지 말고, 사용자가 선택한 1안을 기준으로 형태 / CMF / 기능 디테일을 발전시키세요.
- 후속 이미지가 꼭 필요할 때도 선택된 1안의 개선 렌더 1장만 생성하세요. 비교용 2~4안 재생성은 사용자가 명시적으로 요청한 경우에만 허용합니다.
- 사용자가 시안을 확정하거나 "1번으로 진행", "이 안으로 확정"처럼 최종 선택을 말하면 추가 질문이나 이미지 생성 없이 STEP 6 RFP 문서 생성으로 바로 넘어가세요.
- 디자인 시안 1안과 수정 여부가 확정되기 전에는 RFP로 넘어가지 마세요.
`.trim()
    case 'step_6_rfp':
      return `
[현재 단계 운영]
- 지금은 STEP 6 평가 및 제품개발 기획안 생성 단계입니다.
- 정보가 충분하면 반드시 제품개발 기획안/RFP 출력 템플릿대로 문서를 작성하세요.
- 정보가 부족하면 RFP를 쓰지 말고 부족한 항목 1개만 질문하세요.
- 어떤 경우에도 Persona Card 템플릿을 출력하지 마세요. RFP 문서와 Persona Card는 서로 다른 산출물입니다.
- 문서 작성이 끝나면 다음으로 STEP 7 협력업체 연결 단계로 넘어갈지 물어보세요.
- 시스템은 이 RFP 본문을 그대로 PDF로 저장할 수 있습니다.
- 따라서 "PDF로는 제공할 수 없다", "파일 형태로 직접 생성할 수 없다", "복사해서 사용해달라" 같은 제한 문구를 절대 말하지 마세요.
`.trim()
    case 'step_6_company':
      return `
[현재 단계 운영]
- 지금은 STEP 7 협력업체 연결 단계입니다.
- RFP 생성 이후의 실행 연결 단계로, Persona Card나 RFP 본문을 새로 생성하지 마세요.
- 먼저 현재 프로젝트에 필요한 협력 유형을 디자인 고도화 / 브랜드·런칭·시장 검증 / 시제품 제작 중 1개로 판단하세요.
- 실제 업체명·전화번호·홈페이지는 검증된 검색 결과 없이는 지어내지 마세요.
- 업체 DB 또는 검색 결과가 없으면, 추천 협력 유형과 업체 선별 기준, 문의 시 전달할 핵심 RFP 요약만 출력하세요.
`.trim()
    case 'step_4_definition':
      return `
[현재 단계 운영]
- 지금은 기존 세션의 STEP 4입니다. 새 흐름의 STEP 4 스타일 컨셉 도출로 전환하세요.
- 스타일 레퍼런스 이미지 3장 생성과 선택을 진행하고, next_stage는 step_4_style로 설정하세요.
`.trim()
    case 'step_5_rfp':
      return `
[현재 단계 운영]
- 지금은 기존 세션의 RFP 단계입니다.
- 스타일 레퍼런스 선택과 디자인 시안 확정이 대화에 없으면 RFP를 생성하지 말고 STEP 4 스타일 컨셉 도출로 되돌리세요.
- 모두 충족되어 있다면 STEP 6 RFP 문서 생성 지침을 따르세요.
`.trim()
    default:
      return ''
  }
}

function buildSystemPrompt({
  project,
  referenceImages,
  currentStageKey,
  activeExpert,
  expertCall,
}: {
  project: ProjectRecord | null
  referenceImages: ReferenceImageRecord[]
  currentStageKey: StageKey
  activeExpert: ExpertKey
  expertCall: boolean
}) {
  const projectContext = buildProjectContext(project, referenceImages)
  const stageInstruction = getStageSpecificInstruction(currentStageKey)
  const expertInstruction = expertCall ? getExpertPrompt(activeExpert) : ''
  const expertLabel = expertCall
    ? getExpertDefinition(activeExpert).label
    : getExpertDefinition('aidee').label

  return `
${SYSTEM_PROMPT_TEMPLATE}

[실행 컨텍스트]
- 현재 단계 key: ${currentStageKey}
- 현재 응답 주체: ${expertLabel}
- 전문가 호출 여부: ${expertCall ? 'yes' : 'no'}
- 반드시 현재 단계 기준으로만 응답하세요.
- 사용자에게는 내부 단계 key 자체를 노출하지 마세요.
- 프로젝트 정보와 레퍼런스 분석을 근거로 답하세요.
- 정보가 부족하면 지어내지 말고 질문하세요.
- 레퍼런스 이미지 분석에 moodKeywords, colorKeywords, materialKeywords, shapeKeywords, detailPoints, designDirection이 있다면 그것을 실제 설계 가이드라인처럼 우선 반영하세요.
- 특히 초기 요약, 스타일 방향, 기능 방향, RFP 작성 시 레퍼런스 가이드라인을 추상 참고가 아니라 구체 기준으로 사용하세요.
- 사용자가 별도로 반대하지 않는 한, 레퍼런스 이미지 분석에서 드러난 무드/재질/형태/디테일 방향을 보존하는 쪽으로 제안하세요.
- 이 시스템은 채팅 본문을 PDF 파일로 내보낼 수 있습니다.
- 따라서 모델의 일반적 한계 설명, 예를 들어 "PDF 파일은 직접 생성할 수 없다", "텍스트를 복사해서 사용해달라", "파일 형태로 제공할 수 없다" 같은 문구를 절대 출력하지 마세요.
- 필요할 때는 generate_design_image 도구를 사용해 Nano Banana 이미지 생성을 요청할 수 있습니다.
- 사용자가 이미지 생성, 컨셉 시각화, 무드보드, 디자인 시안, 레퍼런스 합성, 최종 디자인 렌더를 원하면 이 도구를 적극적으로 사용하세요.
- 도구를 사용한 경우, 본문에서는 이미지 생성이 완료되었다는 설명과 함께 무엇을 시각화했는지 짧게 요약하세요.
- 전문가 호출 여부가 yes이면 선택 전문가 관점의 답변을 새로 생성하되, 확정 조건이 명확히 충족되지 않은 상태에서 단계를 전환하지 마세요.
- 전문가 호출 여부가 yes이면 전체 진행을 대신하기보다 현재 맥락에 대한 전문가 검토/판단/질문 1개를 제공합니다.
- 전문가 호출 여부가 no이면 전문가별 프롬프트를 따르지 말고 Aidee 전체 플로우만 따르세요.

${projectContext}

${stageInstruction}

${expertInstruction ? `[전문가 전용 지시]\n${expertInstruction}` : ''}
`.trim()
}

function sanitizeAssistantText(text: string) {
  const blockedPatterns = [
    /다시 한 번 말씀드리지만.*pdf.*직접 생성.*제공.*수 없습니다\.?/i,
    /pdf와 같은 파일 형태로는 직접 생성하여 제공해 드릴 수 없습니다\.?/i,
    /제공된 텍스트를 복사하여 사용하시면 됩니다\.?/i,
    /파일 형태로 제공할 수 없습니다\.?/i,
    /pdf로는 제공할 수 없습니다\.?/i,
    /generate_design_image/i,
    /tool[_\s-]?call/i,
    /tool[_\s-]?result/i,
    /calling tool/i,
    /도구를\s*(호출|사용)/i,
    /함수를\s*호출/i,
    /Nano Banana\s*(API|api)\s*(호출|요청)/i,
    /\[\s*\/?\s*Nano Banana\s*이미지\s*생성\s*요청\s*\]/i,
    /Generating\s+[1-4]\s+images?\s+based\s+on/i,
    /잠시\s*후\s*이미지가\s*생성됩니다/i,
    /잠시(?:만)?\s*(?:기다려|기다려\s*주세요|기다려주세요)/i,
    /곧\s*(?:이미지|시안|렌더).*(?:생성|준비)/i,
    /^프롬프트\s*:/i,
    /^prompt\s*:/i,
    /^\(?이미지\s*생성\s*중\.\.\.\)?$/i,
    /^\(?image\s*generation\s*in\s*progress\.\.\.\)?$/i,
    /\(?\s*이미지\s*[1-4]\s*placeholder\s*\)?/i,
    /\(?\s*image\s*[1-4]\s*placeholder\s*\)?/i,
    /이미지\s*placeholder/i,
    /image\s*placeholder/i,
    /페르소나\s*이미지가\s*생성되었습니다/i,
    /\[시스템\s*참고:[\s\S]*?\]/i,
    /<<AIDEE_STAGE>>[\s\S]*?<<\/AIDEE_STAGE>>/i,
  ]

  const textWithoutNanoBananaBlock = text.replace(
    /\[\s*Nano Banana\s*이미지\s*생성\s*요청\s*\][\s\S]*?\[\s*\/\s*Nano Banana\s*이미지\s*생성\s*요청\s*\]/gi,
    ''
  )

  const lines = textWithoutNanoBananaBlock
    .split('\n')
    .filter((line) => !blockedPatterns.some((pattern) => pattern.test(line)))

  return lines
    .join('\n')
    .replace(/<<AIDEE_STAGE>>[\s\S]*?<<\/AIDEE_STAGE>>/gi, '')
    .trim()
}

function normalizeChoiceFormatting(text: string) {
  const protectedBlockPattern =
    /^<<AIDEE_(?:RFP_JSON|IMAGES)>>[\s\S]*<<\/AIDEE_(?:RFP_JSON|IMAGES)>>$/

  return text
    .split(/(<<AIDEE_(?:RFP_JSON|IMAGES)>>[\s\S]*?<<\/AIDEE_(?:RFP_JSON|IMAGES)>>)/g)
    .map((segment) => {
      if (protectedBlockPattern.test(segment.trim())) {
        return segment
      }

      const lines = segment.split('\n')
      let inChoiceBlock = false
      let choiceIndex = 0

      const looksLikeChoicePrompt = (line: string) => {
        const normalized = line.trim().toLowerCase()
        return (
          /\?$/.test(line.trim()) ||
          /선택지|고르|골라|아래 중|다음 중|원하시|원하는|choose|pick/.test(
            normalized
          )
        )
      }

      const looksLikeStepLine = (line: string) =>
        /^\s*(?:STEP\s*)?[1-7][.)]\s+/.test(line) ||
        /^\s*[1-7][.)]\s*(?:제품 아이디어|사용자 명확화|디자인\/개발|스타일 컨셉|디자인 제안|평가 및 RFP|협력업체)/.test(
          line
        )

      const convertAlphabeticStepLine = (line: string) =>
        line
          .replace(/^\s*(?:A|a)[.)]\s*(제품 아이디어)/, '1. $1')
          .replace(/^\s*(?:B|b)[.)]\s*(사용자 명확화)/, '2. $1')
          .replace(/^\s*(?:C|c)[.)]\s*(디자인\/개발)/, '3. $1')

      const looksLikeAlphabeticStepLine = (line: string) =>
        /^\s*(?:A|a)[.)]\s*제품 아이디어/.test(line) ||
        /^\s*(?:B|b)[.)]\s*사용자 명확화/.test(line) ||
        /^\s*(?:C|c)[.)]\s*디자인\/개발/.test(line)

      const convertIndexedChoice = (line: string) =>
        line
          .replace(/^\s*(?:1|①)[.)]\s+/, 'A. ')
          .replace(/^\s*(?:2|②)[.)]\s+/, 'B. ')
          .replace(/^\s*(?:3|③)[.)]\s+/, 'C. ')
          .replace(/^\s*(?:A|a)[.)]\s+/, 'A. ')
          .replace(/^\s*(?:B|b)[.)]\s+/, 'B. ')
          .replace(/^\s*(?:C|c)[.)]\s+/, 'C. ')

      const normalizedLines = lines.map((line, index) => {
        const trimmed = line.trim()

        if (!trimmed) {
          inChoiceBlock = false
          choiceIndex = 0
          return line
        }

        if (trimmed.startsWith('<<AIDEE_')) {
          inChoiceBlock = false
          choiceIndex = 0
          return line
        }

        if (trimmed.startsWith('#')) {
          inChoiceBlock = false
          choiceIndex = 0
          return line
        }

        const previousNonEmptyLine = [...lines.slice(0, index)]
          .reverse()
          .find((candidate) => candidate.trim().length > 0)

        if (
          !inChoiceBlock &&
          !(
            previousNonEmptyLine &&
            looksLikeChoicePrompt(previousNonEmptyLine)
          ) &&
          looksLikeStepLine(trimmed)
        ) {
          inChoiceBlock = false
          choiceIndex = 0
          return line
        }

        if (
          !inChoiceBlock &&
          !(
            previousNonEmptyLine &&
            looksLikeChoicePrompt(previousNonEmptyLine)
          ) &&
          looksLikeAlphabeticStepLine(trimmed)
        ) {
          inChoiceBlock = false
          choiceIndex = 0
          return convertAlphabeticStepLine(line)
        }

        if (looksLikeChoicePrompt(line)) {
          inChoiceBlock = true
          choiceIndex = 0
          return line
        }

        if (
          inChoiceBlock &&
          /^[\-•*]\s+/.test(trimmed) &&
          choiceIndex < 3
        ) {
          const mappedPrefix = ['A. ', 'B. ', 'C. '][choiceIndex]
          choiceIndex += 1
          return line.replace(/^\s*[\-•*]\s+/, mappedPrefix)
        }

        if (
          inChoiceBlock &&
          /^\s*(?:1|①|2|②|3|③)[.)]\s+/.test(trimmed)
        ) {
          inChoiceBlock = true
          choiceIndex = Math.min(choiceIndex + 1, 3)
          return convertIndexedChoice(line)
        }

        if (
          previousNonEmptyLine &&
          looksLikeChoicePrompt(previousNonEmptyLine) &&
          /^[\-•*]\s+/.test(trimmed) &&
          choiceIndex < 3
        ) {
          inChoiceBlock = true
          const mappedPrefix = ['A. ', 'B. ', 'C. '][choiceIndex]
          choiceIndex += 1
          return line.replace(/^\s*[\-•*]\s+/, mappedPrefix)
        }

        inChoiceBlock = false
        choiceIndex = 0
        return line
      })

      return normalizedLines.join('\n')
    })
    .join('')
    .trim()
}

function formatRfpMarkdown(rfp: RfpDocument) {
  return [
    '# 제품 제안요청서',
    '',
    '## 1. 프로젝트 개요',
    `- 프로젝트명: ${rfp.projectName}`,
    `- 제품 한 줄 정의: ${rfp.oneLineDefinition}`,
    `- 프로젝트 목표: ${rfp.projectGoal}`,
    `- 최종 활용 목적: ${rfp.finalPurpose}`,
    '',
    '## 2. 페르소나',
    `- 메인 타겟: ${rfp.mainTarget}`,
    `- 사용 상황(TPO): ${rfp.usageContext}`,
    `- 핵심 니즈 / 문제: ${rfp.coreNeeds}`,
    '',
    '## 3. 제품 방향',
    `- 핵심 가치: ${rfp.coreValue}`,
    `- 스타일 키워드: ${rfp.styleKeywords.join(', ')}`,
    `- 피해야 하는 방향: ${rfp.avoidDirections.join(', ')}`,
    '',
    '## 4. 기능 요구사항',
    '- 반드시 포함할 핵심 기능',
    ...rfp.mustHaveFeatures.map((item) => `- ${item}`),
    '- 있으면 좋은 기능',
    ...rfp.niceToHaveFeatures.map((item) => `- ${item}`),
    '- 이번 범위에서 제외하는 기능',
    ...rfp.excludedFeatures.map((item) => `- ${item}`),
    '',
    '## 5. 구현 및 제작 조건',
    `- 예상 예산 범위: ${rfp.budgetRange}`,
    `- 목표 기간: ${rfp.timeline}`,
    `- 예상 크기 / 형태 조건: ${rfp.sizeOrForm}`,
    '- 구현 시 주의할 점',
    ...rfp.implementationNotes.map((item) => `- ${item}`),
    '',
    '## 6. 레퍼런스 및 시장 인사이트',
    `- 참고 이미지/레퍼런스 요약: ${rfp.referenceSummary}`,
    '- 리서치 핵심 인사이트',
    ...rfp.researchInsights.map((item) => `- ${item}`),
    '',
    '## 7. 성공 기준',
    ...rfp.successCriteria.map((item) => `- ${item}`),
    '',
    '## 8. 다음 액션',
    ...rfp.nextActions.map((item) => `- ${item}`),
  ].join('\n')
}

function buildFallbackRfpDocument({
  project,
  referenceImages,
}: {
  project: ProjectRecord | null
  referenceImages: ReferenceImageRecord[]
}): RfpDocument {
  const requirements = project?.requirements ?? {}
  const requirementText = JSON.stringify(requirements, null, 2)
  const referenceSummary =
    buildReferenceContext(referenceImages) || '레퍼런스 이미지 분석 결과 없음'
  const projectName = project?.title || '제목 없음'

  return {
    projectName,
    oneLineDefinition: `${projectName}의 제품화 방향을 정리한 디자인 RFP`,
    projectGoal:
      requirementText.length > 5
        ? '대화에서 확정된 요구사항을 기준으로 제품 디자인과 제작 범위를 명확히 정의'
        : '제품 아이디어를 실행 가능한 디자인 및 제작 요청사항으로 구체화',
    finalPurpose: '디자인 고도화, 시제품 제작, 협력업체 커뮤니케이션 기준 문서',
    mainTarget: '대화에서 정의한 핵심 사용자 및 사용 맥락',
    usageContext: '사용자가 제품 필요성을 느끼는 주요 일상 상황',
    coreNeeds: '기존 대안으로 충분히 해결되지 않은 사용 불편과 욕구',
    coreValue: '사용자 경험 개선과 제품 차별성 확보',
    styleKeywords: ['정돈된 사용성', '일관된 디자인 언어', '제작 가능성'],
    avoidDirections: ['확정되지 않은 기능 과잉', '제작 난이도를 높이는 불필요한 장식'],
    mustHaveFeatures: ['핵심 사용 상황을 해결하는 기본 기능', '선택된 디자인 방향을 반영한 형태와 재질'],
    niceToHaveFeatures: ['브랜드 확장에 활용 가능한 디테일', '사용 편의성을 높이는 부가 기능'],
    excludedFeatures: ['현재 RFP 범위를 벗어난 고위험 기능'],
    budgetRange: '협력업체 견적 산정 필요',
    timeline: '협력업체 협의 후 확정',
    sizeOrForm: '선택된 디자인 시안과 사용 환경을 기준으로 상세화',
    implementationNotes: ['초기 시제품 단계에서 구조 안정성과 제작 공정을 우선 검토'],
    referenceSummary: truncateText(referenceSummary, 500),
    researchInsights: ['타겟 사용자의 실제 사용 맥락과 디자인 선호를 기준으로 제품 방향 설정'],
    successCriteria: ['핵심 사용 문제 해결', '선택된 스타일 방향 반영', '시제품 제작 가능성 확보'],
    nextActions: ['RFP 기반 협력업체 문의', '견적 및 제작 범위 확인', '시제품 제작 일정 협의'],
  }
}

async function generateRfpResponse({
  google,
  project,
  referenceImages,
  messages,
}: {
  google: ReturnType<typeof createGoogleGenerativeAI>
  project: ProjectRecord | null
  referenceImages: ReferenceImageRecord[]
  messages: ModelMessage[]
}) {
  const requirementsText = buildRequirementsText(project)
  const referenceContext = buildReferenceContext(referenceImages)
  const conversation = buildConversationText(messages)

  let rfpDocument: RfpDocument

  try {
    const rfpObjectResult = await generateObject({
      model: google('gemini-2.5-flash'),
      schema: RFP_DOCUMENT_SCHEMA,
      prompt: buildRfpObjectPrompt({
        projectTitle: project?.title || '제목 없음',
        requirements: requirementsText,
        referenceContext,
        conversation,
      }),
    })

    rfpDocument = rfpObjectResult.object
  } catch (error) {
    console.error('RFP object generation failed, using fallback document:', error)
    rfpDocument = buildFallbackRfpDocument({ project, referenceImages })
  }

  const markdown = formatRfpMarkdown(rfpDocument)
  const finalText = `${markdown}

<<AIDEE_RFP_JSON>>
${JSON.stringify(rfpDocument, null, 2)}
<</AIDEE_RFP_JSON>>`

  return new Response(appendStageTransitionPrompt(finalText, 'step_6_company'), {
    headers: {
      'Content-Type': 'text/plain; charset=utf-8',
      'x-aidee-current-stage': 'step_6_rfp',
      'x-aidee-next-stage': 'step_6_company',
      'x-aidee-transition': 'no',
      'x-aidee-reason': 'rfp_completed',
    },
  })
}

function parseStageMeta(text: string, fallbackStageKey: StageKey) {
  const match = text.match(
    /<<AIDEE_STAGE>>[\s\n]*current_stage=(.+?)[\s\n]*next_stage=(.+?)[\s\n]*transition=(yes|no)[\s\n]*reason=(.+?)[\s\n]*<<\/AIDEE_STAGE>>/
  )

  const cleanedText = text.replace(
    /\n?<<AIDEE_STAGE>>[\s\S]*?<<\/AIDEE_STAGE>>\s*$/,
    ''
  )

  if (!match) {
    return {
      cleanedText: text.trim(),
      stageMeta: {
        currentStageKey: fallbackStageKey,
        nextStageKey: fallbackStageKey,
        transition: false,
        reason: 'missing_meta',
      } satisfies StageMeta,
    }
  }

  const [, rawCurrentStage, rawNextStage, rawTransition, rawReason] = match
  const parsedCurrentStage = rawCurrentStage.trim()
  const parsedNextStage = rawNextStage.trim()
  const currentStageKey: StageKey = isKnownStageKey(parsedCurrentStage)
    ? parsedCurrentStage
    : fallbackStageKey
  const nextStageKey: StageKey = isKnownStageKey(parsedNextStage)
    ? parsedNextStage
    : currentStageKey

  return {
    cleanedText: cleanedText.trim(),
    stageMeta: {
      currentStageKey,
      nextStageKey,
      transition:
        rawTransition.trim() === 'yes' && currentStageKey !== nextStageKey,
      reason: rawReason.trim() || 'unspecified',
    } satisfies StageMeta,
  }
}

function guardSequentialStageMeta(
  stageMeta: StageMeta,
  fallbackStageKey: StageKey
): StageMeta {
  const currentStageKey = isSameOrNextStage(
    fallbackStageKey,
    stageMeta.currentStageKey
  )
    ? stageMeta.currentStageKey
    : fallbackStageKey

  const nextStageKey = isSameOrNextStage(currentStageKey, stageMeta.nextStageKey)
    ? stageMeta.nextStageKey
    : currentStageKey

  const transition =
    stageMeta.transition &&
    currentStageKey !== nextStageKey &&
    getNextStageKey(currentStageKey) === nextStageKey

  if (
    currentStageKey === stageMeta.currentStageKey &&
    nextStageKey === stageMeta.nextStageKey &&
    transition === stageMeta.transition
  ) {
    return stageMeta
  }

  return {
    currentStageKey,
    nextStageKey,
    transition,
    reason: `blocked_non_sequential_${stageMeta.reason}`,
  }
}

function inferStageTransitionFromText({
  currentStageKey,
  text,
}: {
  currentStageKey: StageKey
  text: string
}): StageMeta | null {
  const transitionHints = [
    /다음 단계로 진행/i,
    /다음 단계로 이어/i,
    /다음 단계로 넘어/i,
    /STEP\s*[2-6](?:으로|에)\s*(?:넘어가|진입|이동)/i,
    /이 단계는 여기까지/i,
    /단계가 완료/i,
  ]

  const hasTransitionHint = transitionHints.some((pattern) => pattern.test(text))

  if (!hasTransitionHint) {
    return null
  }

  const nextStageKey = getNextStageKey(currentStageKey)
  if (
    !nextStageKey ||
    ![
      'step_0_start',
      'step_1_idea',
      'step_2_persona',
      'step_2_research',
      'step_3_direction',
      'step_4_style',
      'step_5_design',
      'step_6_rfp',
    ].includes(currentStageKey)
  ) {
    return null
  }

  return {
    currentStageKey,
    nextStageKey,
    transition: true,
    reason: 'stage_complete',
  }
}

function inferCurrentStageFromText(text: string): StageKey | null {
  const currentStagePatterns: Array<[RegExp, StageKey]> = [
    [
      /(?:지금|현재|이제|오늘)\s*(?:은|부터)?\s*STEP\s*0(?:\s|\.|:|입니다|단계)/i,
      'step_0_start',
    ],
    [
      /(?:지금|현재|이제|오늘)\s*(?:은|부터)?\s*STEP\s*1(?:\s|\.|:|입니다|단계)/i,
      'step_1_idea',
    ],
    [
      /(?:지금|현재|이제|오늘)\s*(?:은|부터)?\s*STEP\s*2(?:\s|\.|:|입니다|단계)/i,
      'step_2_persona',
    ],
    [
      /(?:지금|현재|이제|오늘)\s*(?:은|부터)?\s*STEP\s*3(?:\s|\.|:|입니다|단계)/i,
      'step_3_direction',
    ],
    [
      /(?:지금|현재|이제|오늘)\s*(?:은|부터)?\s*STEP\s*4(?:\s|\.|:|입니다|단계)/i,
      'step_4_style',
    ],
    [
      /(?:지금|현재|이제|오늘)\s*(?:은|부터)?\s*STEP\s*5(?:\s|\.|:|입니다|단계)/i,
      'step_5_design',
    ],
    [
      /(?:지금|현재|이제|오늘)\s*(?:은|부터)?\s*STEP\s*6(?:\s|\.|:|입니다|단계)/i,
      'step_6_rfp',
    ],
    [/STEP\s*3(?:으로|에)\s*(?:넘어왔|진입|들어왔|이동)/i, 'step_3_direction'],
    [/STEP\s*2(?:으로|에)\s*(?:넘어왔|넘어가|진입|들어왔|이동)/i, 'step_2_persona'],
    [/STEP\s*4(?:으로|에)\s*(?:넘어왔|진입|들어왔|이동)/i, 'step_4_style'],
    [/STEP\s*5(?:으로|에)\s*(?:넘어왔|진입|들어왔|이동)/i, 'step_5_design'],
    [/STEP\s*6(?:으로|에)\s*(?:넘어왔|진입|들어왔|이동)/i, 'step_6_rfp'],
  ]

  return (
    currentStagePatterns.find(([pattern]) => pattern.test(text))?.[1] ?? null
  )
}

export async function POST(req: Request) {
  try {
    const body = (await req.json()) as ChatRequestBody
    const forceImageGeneration = body.forceImageGeneration

    if (!Array.isArray(body?.messages)) {
      return new Response(
        JSON.stringify({ error: 'messages must be an array' }),
        {
          status: 400,
          headers: { 'Content-Type': 'application/json' },
        }
      )
    }

    const apiKey =
      process.env.GOOGLE_GENERATIVE_AI_API_KEY ?? process.env.GEMINI_API_KEY

    if (!apiKey) {
      return new Response(JSON.stringify({ error: 'API key not found' }), {
        status: 500,
        headers: { 'Content-Type': 'application/json' },
      })
    }

    const google = createGoogleGenerativeAI({ apiKey })

    if (
      !process.env.NEXT_PUBLIC_SUPABASE_URL ||
      !process.env.SUPABASE_SERVICE_ROLE_KEY
    ) {
      return new Response(JSON.stringify({ error: 'Supabase env missing' }), {
        status: 500,
        headers: { 'Content-Type': 'application/json' },
      })
    }

    const normalizedMessages = normalizeMessages(body.messages)
    const isInitialEntry = normalizedMessages.length === 0
    const requestedStageKey = body.currentStageKey ?? DEFAULT_STAGE_KEY
    const lastUserMessage =
      [...normalizedMessages]
        .reverse()
        .find((message) => message.role === 'user')?.content ?? ''
    const currentStageKey = resolveIntentStageKey({
      currentStageKey: requestedStageKey,
      lastUserMessage,
    })
    const activeExpert = isExpertKey(body.activeExpert)
      ? body.activeExpert
      : 'aidee'
    const expertCall = body.expertCall === true && activeExpert !== 'aidee'

    const supabase = createClient(
      process.env.NEXT_PUBLIC_SUPABASE_URL,
      process.env.SUPABASE_SERVICE_ROLE_KEY
    )

    let project: ProjectRecord | null = null
    let referenceImages: ReferenceImageRecord[] = []

    if (body.projectId) {
      const { data: projectData } = await supabase
        .from('projects')
        .select('title, requirements')
        .eq('id', body.projectId)
        .single()

      project = (projectData as ProjectRecord | null) ?? null

      const { data: referenceData } = await supabase
        .from('project_reference_images')
        .select('file_name, analysis_text, analysis_json')
        .eq('project_id', body.projectId)
        .eq('analysis_status', 'completed')

      referenceImages = (referenceData as ReferenceImageRecord[] | null) ?? []
    }

    if (
      !expertCall &&
      (currentStageKey === 'step_0_start' || currentStageKey === 'step_1_idea')
    ) {
      if (isInitialEntry) {
        return new Response(
          buildProjectCardResponse({
            project,
            referenceImages,
          }),
          {
            headers: {
              'Content-Type': 'text/plain; charset=utf-8',
              'x-aidee-current-stage': currentStageKey,
              'x-aidee-next-stage': currentStageKey,
              'x-aidee-transition': 'no',
              'x-aidee-reason': 'project_card_created',
            },
          }
        )
      }

      if (
        isProcessConfirmationRequest(lastUserMessage) &&
        hasProjectStartSummaryMessage(normalizedMessages) &&
        !hasProcessGuideMessage(normalizedMessages)
      ) {
        return new Response(buildProcessGuideResponse(), {
          headers: {
            'Content-Type': 'text/plain; charset=utf-8',
            'x-aidee-current-stage': currentStageKey,
            'x-aidee-next-stage': 'step_1_idea',
            'x-aidee-transition': 'no',
            'x-aidee-reason': 'process_confirmed',
          },
        })
      }

      if (
        hasProjectCardMessage(normalizedMessages) &&
        !hasProjectStartSummaryMessage(normalizedMessages) &&
        lastUserMessage.trim()
      ) {
        return new Response(
          buildProjectStartSummaryResponse({
            project,
            lastUserMessage,
          }),
          {
            headers: {
              'Content-Type': 'text/plain; charset=utf-8',
              'x-aidee-current-stage': currentStageKey,
              'x-aidee-next-stage': currentStageKey,
              'x-aidee-transition': 'no',
              'x-aidee-reason': 'project_start_summary_created',
            },
          }
        )
      }
    }

    const system = buildSystemPrompt({
      project,
      referenceImages,
      currentStageKey,
      activeExpert,
      expertCall,
    })

    const messages: ModelMessage[] = isInitialEntry
      ? [
          {
            role: 'user',
            content: buildInitialPrompt(project),
          },
        ]
      : expertCall
        ? [
            ...normalizedMessages,
            {
              role: 'user',
              content: buildExpertCallPrompt(activeExpert),
            },
          ]
        : normalizedMessages

    if (!expertCall && currentStageKey === 'step_2_persona') {
      const personaFlowResponse = buildPersonaFlowResponse({
        messages: normalizedMessages,
        forceImageGeneration,
        lastUserMessage,
        project,
      })

      if (personaFlowResponse) {
        return personaFlowResponse
      }
    }

    if (!expertCall && currentStageKey === 'step_3_direction') {
      return buildDirectionFlowResponse({
        messages: normalizedMessages,
        forceImageGeneration,
        lastUserMessage,
        project,
      })
    }

    if (!expertCall && currentStageKey === 'step_4_style') {
      return await buildStyleFlowResponse({
        project,
        referenceImages,
        messages: normalizedMessages,
        lastUserMessage,
        forceImageGeneration,
        apiKey,
      })
    }

    if (currentStageKey === 'step_6_rfp') {
      try {
        return await generateRfpResponse({
          google,
          project,
          referenceImages,
          messages,
        })
      } catch (error) {
        console.error('RFP structured generation failed:', error)
      }
    }

    let generatedImagePayload: GeneratedImageBlock | null = null
    const shouldGenerateStyleReferenceImages =
      currentStageKey === 'step_4_style' &&
      !hasStyleReferenceSelection(lastUserMessage)
    const isStyleReferenceSelectionTurn =
      currentStageKey === 'step_4_style' &&
      hasStyleReferenceSelection(lastUserMessage)
    const canGenerateImages = canGenerateImagesInStage(currentStageKey)

    if (
      forceImageGeneration === 'persona_visualization' &&
      currentStageKey === 'step_2_persona'
    ) {
      try {
        generatedImagePayload = await generateNanoBananaImages({
          prompt: buildPersonaImagePrompt({
            project,
            personaText: buildConversationText(messages),
          }),
          count: 1,
          apiKey,
        })
        generatedImagePayload.purpose = 'persona'

        return new Response(
          appendGeneratedImagesBlock({
            text:
              '좋아요. 확정한 사용자 정리를 바탕으로 Persona Card를 만들었습니다. 내용을 확인한 뒤 확정해주세요.',
            payload: generatedImagePayload,
          }),
          {
            headers: {
              'Content-Type': 'text/plain; charset=utf-8',
              'x-aidee-current-stage': 'step_2_persona',
              'x-aidee-next-stage': 'step_2_persona',
              'x-aidee-transition': 'no',
              'x-aidee-reason': 'persona_visualized',
            },
          }
        )
      } catch (error) {
        console.error('Persona visualization image generation failed:', error)
        return new Response(
          '페르소나 시각화 중 오류가 발생했습니다. 잠시 후 다시 시도해주세요.',
          {
            status: 500,
            headers: {
              'Content-Type': 'text/plain; charset=utf-8',
              'x-aidee-current-stage': 'step_2_persona',
              'x-aidee-next-stage': 'step_2_persona',
              'x-aidee-transition': 'no',
              'x-aidee-reason': 'persona_visualization_failed',
            },
          }
        )
      }
    }

    if (shouldGenerateStyleReferenceImages) {
      console.log('[style-images] direct style generation branch entered', {
        currentStageKey,
        lastUserMessage,
        conversationLength: buildConversationText(messages).length,
      })
      generatedImagePayload = await generateStyleReferenceImages({
        project,
        referenceImages,
        conversation: buildConversationText(messages),
        apiKey,
      })

      if (!generatedImagePayload) {
        return new Response(
          '스타일 레퍼런스 이미지 생성 중 오류가 발생했습니다. 잠시 후 다시 시도해주세요.',
          {
            status: 500,
            headers: {
              'Content-Type': 'text/plain; charset=utf-8',
              'x-aidee-current-stage': 'step_4_style',
              'x-aidee-next-stage': 'step_4_style',
              'x-aidee-transition': 'no',
              'x-aidee-reason': 'style_reference_image_generation_failed',
            },
          }
        )
      }
    }
    const shouldBypassModelTextForStyleImages = shouldGenerateStyleReferenceImages

    const shouldGenerateInitialDesignImages =
      (forceImageGeneration === 'initial_design' ||
        currentStageKey === 'step_5_design') &&
      !expertCall &&
      (forceImageGeneration === 'initial_design' ||
        !hasGeneratedDesignImagesInMessages(messages)) &&
      !hasDesignFinalSelection(lastUserMessage) &&
      (forceImageGeneration === 'initial_design' ||
        !isDesignRevisionRequest(lastUserMessage)) &&
      (forceImageGeneration === 'initial_design' ||
        hasStyleReferenceSelection(lastUserMessage) ||
        isImageGenerationRequest(lastUserMessage) ||
        /다음\s*단계|진행|디자인\s*제안|STEP\s*5/i.test(lastUserMessage) ||
        requestedStageKey === 'step_4_style')

    if (shouldGenerateInitialDesignImages) {
      try {
        generatedImagePayload = await generateNanoBananaImages({
          prompt: buildInitialDesignImagePrompt({
            project,
            referenceImages,
            conversation: buildConversationText(messages),
            userSelection: lastUserMessage,
          }),
          count: 4,
          apiKey,
        })
        generatedImagePayload.purpose = 'design'

        return new Response(
          appendGeneratedImagesBlock({
            text: [
              '선택한 스타일 레퍼런스를 기준으로 STEP 5 디자인 시안 4안을 생성했습니다.',
              '아래 시안 중 가장 발전시키고 싶은 1안을 선택해주세요.',
              '이후에는 선택한 1안을 기준으로 부분 수정과 최종 확정을 진행합니다.',
            ].join('\n'),
            payload: generatedImagePayload,
          }),
          {
            headers: {
              'Content-Type': 'text/plain; charset=utf-8',
              'x-aidee-current-stage': 'step_5_design',
              'x-aidee-next-stage': 'step_5_design',
              'x-aidee-transition': 'no',
              'x-aidee-reason': 'initial_design_images_generated',
            },
          }
        )
      } catch (error) {
        console.error('Initial design image generation failed:', error)
        return new Response(
          '이미지 생성 중 오류가 발생했습니다. 잠시 후 다시 시도해주세요.',
          {
            status: 500,
            headers: {
              'Content-Type': 'text/plain; charset=utf-8',
              'x-aidee-current-stage': 'step_5_design',
              'x-aidee-next-stage': 'step_5_design',
              'x-aidee-transition': 'no',
              'x-aidee-reason': 'initial_design_image_generation_failed',
            },
          }
        )
      }
    }

    const shouldGenerateDesignRevisionImage =
      (forceImageGeneration === 'design_revision' ||
        currentStageKey === 'step_5_design') &&
      !expertCall &&
      hasGeneratedDesignImagesInMessages(messages) &&
      (forceImageGeneration === 'design_revision' ||
        isDesignRevisionRequest(lastUserMessage)) &&
      !hasDesignFinalSelection(lastUserMessage)

    if (shouldGenerateDesignRevisionImage) {
      try {
        generatedImagePayload = await generateNanoBananaImages({
          prompt: buildFallbackImagePrompt({
            project,
            referenceImages,
            userRequest: [
              lastUserMessage,
              'Use the already selected STEP 5 design direction as the base. Generate an improved single product render unless the user explicitly asked for multiple alternatives.',
            ].join('\n'),
          }),
          count: extractRequestedImageCount(lastUserMessage),
          apiKey,
        })
        generatedImagePayload.purpose = 'design'

        return new Response(
          appendGeneratedImagesBlock({
            text: [
              '선택한 디자인 방향을 기준으로 수정 렌더를 생성했습니다.',
              '아래 이미지를 확인하고 추가 수정 또는 최종 확정 여부를 알려주세요.',
            ].join('\n'),
            payload: generatedImagePayload,
          }),
          {
            headers: {
              'Content-Type': 'text/plain; charset=utf-8',
              'x-aidee-current-stage': 'step_5_design',
              'x-aidee-next-stage': 'step_5_design',
              'x-aidee-transition': 'no',
              'x-aidee-reason': 'design_revision_image_generated',
            },
          }
        )
      } catch (error) {
        console.error('Design revision image generation failed:', error)
        return new Response(
          '이미지 생성 중 오류가 발생했습니다. 잠시 후 다시 시도해주세요.',
          {
            status: 500,
            headers: {
              'Content-Type': 'text/plain; charset=utf-8',
              'x-aidee-current-stage': 'step_5_design',
              'x-aidee-next-stage': 'step_5_design',
              'x-aidee-transition': 'no',
              'x-aidee-reason': 'design_revision_image_generation_failed',
            },
          }
        )
      }
    }

    if (generatedImagePayload && shouldGenerateStyleReferenceImages) {
      const styleText = getStyleReferenceIntro(generatedImagePayload.images.length)
      return new Response(
        appendGeneratedImagesBlock({
          text: styleText,
          payload: generatedImagePayload,
        }),
        {
          headers: {
            'Content-Type': 'text/plain; charset=utf-8',
            'x-aidee-current-stage': currentStageKey,
            'x-aidee-next-stage': currentStageKey,
            'x-aidee-transition': 'no',
            'x-aidee-reason': 'style_references_generated',
          },
        }
      )
    }

    const result = await generateText({
      model: google('gemini-2.5-flash'),
      system,
      messages: isStyleReferenceSelectionTurn
        ? [
            ...messages,
            {
              role: 'user',
              content:
                'Internal instruction: The user has selected one of the already generated style reference images. Do not call generate_design_image or any image generation tool in this turn. Summarize the selected direction into shape, color, and material guidelines, then move to STEP 5 if conditions are met.',
            },
          ]
        : messages,
      stopWhen: ({ steps }) =>
        steps.length >= 1 &&
        steps[steps.length - 1].toolCalls.length === 0,
      tools:
        isStyleReferenceSelectionTurn || !canGenerateImages
          ? undefined
          : {
              generate_design_image: tool({
                description:
                  'Generate product concept or final design images with Gemini Nano Banana and return data URL images.',
                inputSchema: z.object({
                  prompt: z
                    .string()
                    .min(1)
                    .describe(
                      'Detailed image generation prompt for the desired visual'
                    ),
                  count: z
                    .number()
                    .int()
                    .min(1)
                    .max(4)
                    .default(1)
                    .describe('Number of image variations to create'),
                  model: z
                    .enum([
                      'gemini-2.5-flash-image',
                      'gemini-3.1-flash-image-preview',
                    ])
                    .default('gemini-2.5-flash-image')
                    .describe('Nano Banana model to use'),
                }),
                execute: async ({ prompt, count, model }) => {
                  generatedImagePayload = await generateNanoBananaImages({
                    prompt,
                    count,
                    model,
                    apiKey,
                  })
                  generatedImagePayload.purpose = 'design'

                  return {
                    status: 'success',
                    message: `${generatedImagePayload.images.length} image variation(s) generated successfully.`,
                    count: generatedImagePayload.images.length,
                    model: generatedImagePayload.model,
                  }
                },
              }),
            },
    })

    const { cleanedText, stageMeta: parsedStageMeta } = parseStageMeta(
      result.text,
      currentStageKey
    )
    let stageMeta = expertCall
      ? {
          ...parsedStageMeta,
          currentStageKey,
          nextStageKey: currentStageKey,
          transition: false,
          reason: 'expert_call',
        }
      : parsedStageMeta

    if (!expertCall) {
      if (currentStageKey === 'step_4_definition') {
        stageMeta = {
          currentStageKey,
          nextStageKey: 'step_4_style',
          transition: true,
          reason: 'legacy_step_map',
        }
      } else if (currentStageKey === 'step_5_rfp') {
        stageMeta = {
          currentStageKey,
          nextStageKey: 'step_6_company',
          transition: true,
          reason: 'legacy_step_map',
        }
      } else if (
        currentStageKey === 'step_4_style' &&
        hasStyleReferenceSelection(lastUserMessage)
      ) {
        stageMeta = {
          currentStageKey,
          nextStageKey: 'step_5_design',
          transition: true,
          reason: 'style_reference_selected',
        }
      }
    }

    stageMeta = guardSequentialStageMeta(stageMeta, currentStageKey)

    let finalText = normalizeChoiceFormatting(sanitizeAssistantText(cleanedText))
    const modelPrintedNanoBananaPlaceholder =
      hasNanoBananaPlaceholder(cleanedText) ||
      hasNanoBananaPlaceholder(result.text)
    const modelPromisedImageWithoutTool =
      !generatedImagePayload &&
      canGenerateImages &&
      /(?:이미지|시안|렌더).*(?:생성|제작|만들|아래|확인)|(?:생성|제작|만들).*(?:이미지|시안|렌더)|(?:시안|디자인)\s*[1-3]\s*(?:안|번)|[1-3]\s*안\s*[:：]|디자인\s*의도|형태\s*[:：]|색감\s*[:：]|재질\s*[:：]/i.test(
        finalText
      )
    const modelPromisedStyleReferencesWithoutImages =
      !generatedImagePayload &&
      /(?:스타일|컨셉|레퍼런스|무드).*(?:시안|이미지).*(?:3\s*(?:가지|개|장|안)|세\s*(?:가지|개|장|안))|(?:3\s*(?:가지|개|장|안)|세\s*(?:가지|개|장|안)).*(?:스타일|컨셉|레퍼런스|무드).*(?:시안|이미지)|아래\s*3\s*(?:가지|개|장|안).*(?:시안|스타일|이미지)|A\.\s*첫\s*번째[\s\S]*B\.\s*두\s*번째[\s\S]*C\.\s*세\s*번째/i.test(
        finalText
      )

    if (shouldBypassModelTextForStyleImages) {
      finalText = getStyleReferenceIntro(generatedImagePayload?.images.length ?? 3)
    }

    const personaContextForCard = [buildConversationText(messages), finalText].join(
      '\n\n'
    )
    const shouldHandlePersonaCardCandidate =
      !expertCall &&
      currentStageKey === 'step_2_persona' &&
      (isPersonaCardText(finalText) || isPersonaSummaryDraftText(finalText))

    if (
      shouldHandlePersonaCardCandidate &&
      !getPersonaClarificationStatus(personaContextForCard).isComplete
    ) {
      finalText = buildPersonaClarificationQuestion(personaContextForCard, project)
    } else if (
      shouldHandlePersonaCardCandidate
    ) {
      finalText = buildPersonaCardTextFromDraft(finalText)
    }

    if (
      !generatedImagePayload &&
      !expertCall &&
      forceImageGeneration === 'persona_visualization' &&
      currentStageKey === 'step_2_persona' &&
      getPersonaClarificationStatus(
        [buildConversationText(messages), finalText].join('\n\n')
      ).isComplete &&
      (forceImageGeneration === 'persona_visualization' ||
        isPersonaCardText(finalText) ||
        isPersonaImagePlaceholderText(cleanedText) ||
        isPersonaImagePlaceholderText(finalText))
    ) {
      try {
        generatedImagePayload = await generateNanoBananaImages({
          prompt: buildPersonaImagePrompt({
            project,
            personaText: [buildConversationText(messages), finalText].join('\n\n'),
          }),
          count: 1,
          apiKey,
        })
        generatedImagePayload.purpose = 'persona'
        finalText = '좋아요. 확정한 사용자 정리를 바탕으로 페르소나 시각화를 만들었습니다.'
      } catch (error) {
        console.error('Persona image generation failed:', error)
      }
    }

    if (
      !generatedImagePayload &&
      canGenerateImages &&
      (isImageGenerationRequest(lastUserMessage) || modelPromisedImageWithoutTool)
    ) {
      try {
        generatedImagePayload = await generateNanoBananaImages({
          prompt: buildFallbackImagePrompt({
            project,
            referenceImages,
            userRequest: lastUserMessage,
          }),
          count:
            currentStageKey === 'step_5_design' &&
            !hasGeneratedDesignImagesInMessages(messages)
              ? 3
              : extractRequestedImageCount(lastUserMessage),
          apiKey,
        })
        generatedImagePayload.purpose = 'design'

        if (!finalText.trim()) {
          finalText =
            '이미지를 생성했습니다. 아래 시안을 확인하고, 원하는 수정 방향이 있으면 바로 말씀해주세요.'
        } else if (!/이미지|시안|렌더/.test(finalText)) {
          finalText = `${finalText}\n\n이미지를 생성했습니다. 아래 시안을 확인하고 원하는 수정 방향을 알려주세요.`
        }
      } catch (error) {
        console.error('Fallback Nano Banana image generation failed:', error)
      }
    }

    const shouldGenerateStyleReferencesAfterModel =
      !generatedImagePayload &&
      !expertCall &&
      !hasStyleReferenceSelection(lastUserMessage) &&
      (currentStageKey === 'step_4_style' ||
        stageMeta.currentStageKey === 'step_4_style' ||
        stageMeta.nextStageKey === 'step_4_style' ||
        modelPrintedNanoBananaPlaceholder ||
        modelPromisedStyleReferencesWithoutImages)

    if (shouldGenerateStyleReferencesAfterModel) {
      generatedImagePayload = await generateStyleReferenceImages({
        project,
        referenceImages,
        conversation: buildConversationText(messages),
        apiKey,
      })

      if (generatedImagePayload) {
        finalText = getStyleReferenceIntro(generatedImagePayload.images.length)
      } else if (
        modelPrintedNanoBananaPlaceholder ||
        modelPromisedStyleReferencesWithoutImages
      ) {
        finalText =
          '스타일 레퍼런스 이미지를 생성하지 못했습니다. 잠시 후 다시 시도해주세요.'
        stageMeta = {
          currentStageKey: 'step_4_style',
          nextStageKey: 'step_4_style',
          transition: false,
          reason: 'style_reference_image_generation_failed',
        }
      } else if (!finalText.trim()) {
        finalText =
          '스타일 레퍼런스 이미지를 생성하지 못했습니다. 잠시 후 다시 시도해주세요.'
      } else if (!/선택|이미지|레퍼런스/.test(finalText)) {
        finalText = `${finalText}\n\n스타일 레퍼런스 이미지를 생성하지 못했습니다. 잠시 후 다시 시도해주세요.`
      }

      if (generatedImagePayload) {
        stageMeta = {
          currentStageKey: 'step_4_style',
          nextStageKey: 'step_4_style',
          transition: false,
          reason: 'style_references_generated',
        }
      }
    }

    if (generatedImagePayload) {
      finalText = appendGeneratedImagesBlock({
        text: finalText,
        payload: generatedImagePayload,
      })
    }

    if (
      canRequestRfpStage(currentStageKey) &&
      (currentStageKey === 'step_6_rfp' || currentStageKey === 'step_5_rfp') &&
      (stageMeta.currentStageKey === 'step_6_rfp' ||
        stageMeta.nextStageKey === 'step_6_rfp' ||
        modelPromisedRfpWithoutDocument(finalText)) &&
      !generatedImagePayload
    ) {
      try {
        return await generateRfpResponse({
          google,
          project,
          referenceImages,
          messages,
        })
      } catch (error) {
        console.error('RFP structured generation failed:', error)
        finalText =
          'RFP 문서 생성 중 오류가 발생했습니다. 잠시 후 다시 시도해주세요.'
        stageMeta = {
          currentStageKey: 'step_6_rfp',
          nextStageKey: 'step_6_rfp',
          transition: false,
          reason: 'rfp_generation_failed',
        }
      }
    }

    const shouldGenerateRfpJson =
      canRequestRfpStage(currentStageKey) &&
      (currentStageKey === 'step_6_rfp' || currentStageKey === 'step_5_rfp') &&
      (stageMeta.currentStageKey === 'step_6_rfp' ||
        stageMeta.nextStageKey === 'step_6_rfp' ||
        stageMeta.nextStageKey === 'step_5_rfp') &&
      (cleanedText.includes('# 제품 제안요청서') ||
        cleanedText.includes('## 1. 프로젝트 개요'))

    if (shouldGenerateRfpJson) {
      try {
        const requirementsText = buildRequirementsText(project)
        const referenceContext = buildReferenceContext(referenceImages)
        const conversation = buildConversationText(messages)

        const rfpObjectResult = await generateObject({
          model: google('gemini-2.5-flash'),
          schema: RFP_DOCUMENT_SCHEMA,
          prompt: buildRfpObjectPrompt({
            projectTitle: project?.title || '제목 없음',
            requirements: requirementsText,
            referenceContext,
            conversation,
          }),
        })

        finalText = `${finalText}

<<AIDEE_RFP_JSON>>
${JSON.stringify(rfpObjectResult.object, null, 2)}
<</AIDEE_RFP_JSON>>`
      } catch (error) {
        console.error('RFP JSON generation failed:', error)
      }
    }

    const inferredStageMeta = inferStageTransitionFromText({
      currentStageKey: stageMeta.currentStageKey,
      text: finalText,
    })
    const inferredCurrentStageKey = inferCurrentStageFromText(finalText)

    if (
      inferredCurrentStageKey &&
      inferredCurrentStageKey !== stageMeta.currentStageKey &&
      isSameOrNextStage(stageMeta.currentStageKey, inferredCurrentStageKey)
    ) {
      stageMeta = {
        currentStageKey: inferredCurrentStageKey,
        nextStageKey: inferredCurrentStageKey,
        transition: false,
        reason: 'explicit_current_stage_text',
      }
    } else if (inferredStageMeta) {
      stageMeta = inferredStageMeta
    }

    stageMeta = guardSequentialStageMeta(stageMeta, currentStageKey)

    if (stageMeta.transition) {
      finalText = appendStageTransitionPrompt(finalText, stageMeta.nextStageKey)
    }

    return new Response(finalText, {
      headers: {
        'Content-Type': 'text/plain; charset=utf-8',
        'x-aidee-current-stage': stageMeta.currentStageKey,
        'x-aidee-next-stage': stageMeta.nextStageKey,
        'x-aidee-transition': 'no',
        'x-aidee-reason': stageMeta.reason,
      },
    })
  } catch (error: unknown) {
    const message =
      error instanceof Error ? error.message : 'Unknown chat API error'

    console.error('API Error:', error)

    return new Response(JSON.stringify({ error: message }), {
      status: 500,
      headers: { 'Content-Type': 'application/json' },
    })
  }
}
