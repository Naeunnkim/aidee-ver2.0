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
type ProjectHintDomain =
  | 'lighting'
  | 'interior'
  | 'furniture'
  | 'wearable'
  | 'digital'
  | 'study'
  | 'kitchen'
  | 'rest'
  | 'activity'
  | 'hygiene'
  | 'generic'

type ContextualHintKind =
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

  if (/조명|무드등|램프|빛|색\s*변화/i.test(text)) {
    return '밝기와 분위기가 바뀌는 생활 공간'
  }

  if (/인테리어|소품|오브제|전시|장식/i.test(text)) {
    return '집이나 매장의 눈에 띄는 자리'
  }

  if (/가구|의자|테이블|수납/i.test(text)) {
    return '머무는 시간이 긴 생활 공간'
  }

  if (/패션|악세서리|액세서리|착용|웨어러블/i.test(text)) {
    return '몸에 가까이 닿는 착용 맥락'
  }

  if (/디지털|기기|스마트|IoT|센서|앱/i.test(text)) {
    return '손이 자주 가는 사용 자리'
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

  if (/조명|무드등|램프|빛|색\s*변화/i.test(text)) {
    return '밝기나 분위기를 바꾸고 싶은 순간'
  }

  if (/인테리어|소품|오브제|전시|장식|가구/i.test(text)) {
    return '공간을 정리하거나 분위기를 바꾸는 순간'
  }

  if (/패션|악세서리|액세서리|착용|웨어러블/i.test(text)) {
    return '외출 전 착용감과 인상을 고르는 순간'
  }

  if (/디지털|기기|스마트|IoT|센서|앱/i.test(text)) {
    return '상태를 확인하거나 기능을 조작해야 하는 순간'
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

  if (/조명|무드등|램프|빛|색\s*변화/i.test(text)) {
    return '밝기나 분위기가 원하는 대로 맞지 않는 불편'
  }

  if (/인테리어|소품|오브제|전시|장식|가구/i.test(text)) {
    return '공간과 제품이 자연스럽게 어울리지 않는 불편'
  }

  if (/패션|악세서리|액세서리|착용|웨어러블/i.test(text)) {
    return '착용감이나 인상이 어색해지는 불편'
  }

  if (/디지털|기기|스마트|IoT|센서|앱/i.test(text)) {
    return '조작과 상태 확인이 번거로운 불편'
  }

  return '현재 방식에서 반복되는 불편'
}

function inferProjectHintDomain(text: string): ProjectHintDomain {
  if (/조명|무드등|램프|빛|색\s*변화/i.test(text)) {
    return 'lighting'
  }

  if (/인테리어|소품|오브제|전시|장식/i.test(text)) {
    return 'interior'
  }

  if (/가구|의자|테이블|수납/i.test(text)) {
    return 'furniture'
  }

  if (/패션|악세서리|액세서리|착용|웨어러블/i.test(text)) {
    return 'wearable'
  }

  if (/디지털|기기|스마트|IoT|센서|앱/i.test(text)) {
    return 'digital'
  }

  if (/공부|학습|시험|과제|자격|집중|몰입/i.test(text)) {
    return 'study'
  }

  if (/주방|요리|식사|음식|식품|카페/i.test(text)) {
    return 'kitchen'
  }

  if (/침실|수면|휴식|잠|리빙|거실/i.test(text)) {
    return 'rest'
  }

  if (/운동|피트니스|헬스|러닝|야외|이동|휴대/i.test(text)) {
    return 'activity'
  }

  if (/욕실|세면|샤워|위생|뷰티/i.test(text)) {
    return 'hygiene'
  }

  return 'generic'
}

function inferProjectPurposeLabel(usage: string, productLabel: string) {
  if (/개인\s*소장|전시/i.test(usage)) {
    return '개인 공간에서 오래 보고 쓰는 목적'
  }

  if (/대량\s*판매|판매/i.test(usage)) {
    return '여러 구매자가 쉽게 이해하고 살 수 있는 목적'
  }

  if (/크라우드\s*펀딩|펀딩/i.test(usage)) {
    return '초기 후원자가 빠르게 매력을 이해하는 목적'
  }

  if (/브랜드\s*런칭|런칭/i.test(usage)) {
    return '브랜드 첫인상을 분명히 만드는 목적'
  }

  return `${productLabel} 사용 목적`
}

function inferProjectGoalLabel(goal: string) {
  if (/아이디어\s*구체화/i.test(goal)) {
    return '사용 장면과 핵심 가치를 먼저 정리해야 하는 단계'
  }

  if (/2D|3D|시각화/i.test(goal)) {
    return '형태와 사용 모습을 시각화해야 하는 단계'
  }

  if (/시제품|사업화|제작/i.test(goal)) {
    return '제작성과 판매 가능성을 함께 봐야 하는 단계'
  }

  return '프로젝트 목표에 맞게 검증해야 하는 단계'
}

function inferProjectFormFactorLabel(size: string, productLabel: string) {
  if (/손바닥|10cm/i.test(size)) {
    return '손에 들거나 작은 자리에 둘 수 있는 형태'
  }

  if (/소형|10~50cm/i.test(size)) {
    return '선반이나 책상 위에 둘 수 있는 형태'
  }

  if (/중형|50~100cm/i.test(size)) {
    return '한 공간의 포인트가 되는 형태'
  }

  if (/대형|100cm/i.test(size)) {
    return '공간 배치에 직접 영향을 주는 형태'
  }

  return `${productLabel}의 실제 형태`
}

function inferProjectAudienceLabel(text: string, productLabel: string) {
  if (/대량\s*판매|판매|크라우드\s*펀딩|펀딩|브랜드\s*런칭|런칭/i.test(text)) {
    return '구매를 검토하는 초기 고객'
  }

  if (/반려|펫|강아지|고양이/i.test(text)) {
    return '반려동물과 함께 사는 보호자'
  }

  if (/아이|아기|육아|어린이|부모/i.test(text)) {
    return '아이를 돌보는 보호자'
  }

  if (/공부|학습|시험|과제|학생|수험/i.test(text)) {
    return '학습과 과제를 병행하는 학생'
  }

  if (/업무|사무|오피스|직장|프리랜서/i.test(text)) {
    return '업무 공간을 직접 관리하는 직장인'
  }

  if (/패션|악세서리|액세서리|착용|웨어러블/i.test(text)) {
    return '착용감과 인상을 함께 보는 사용자'
  }

  if (/조명|인테리어|가구|소품|오브제/i.test(text)) {
    return '생활 공간을 직접 꾸미는 사용자'
  }

  if (/디지털|기기|스마트|IoT|센서|앱/i.test(text)) {
    return '기능을 빠르게 익히고 쓰는 사용자'
  }

  return `${productLabel}을 실제로 써볼 사용자`
}

function inferProjectActionLabel(
  text: string,
  featureLabel: string,
  productLabel: string
) {
  if (/빛|색\s*변화|조명|무드등|램프/i.test(text)) {
    return '밝기나 색을 조절하려던 중'
  }

  if (/센서|감지/i.test(text)) {
    return '상태나 움직임을 확인하려던 중'
  }

  if (/IoT|스마트|앱/i.test(text)) {
    return '앱이나 기기로 제어하려던 중'
  }

  if (/조립|분해|설치/i.test(text)) {
    return '설치하거나 형태를 바꾸려던 중'
  }

  if (/패션|악세서리|액세서리|착용|웨어러블/i.test(text)) {
    return '착용하고 외출하려던 중'
  }

  if (/가구|인테리어|소품|오브제/i.test(text)) {
    return '공간에 배치하고 쓰려던 중'
  }

  if (/단순\s*구조물/i.test(text)) {
    return '꺼내거나 배치해 바로 쓰려던 중'
  }

  if (featureLabel && featureLabel !== '핵심 기능') {
    return `${featureLabel}을 실제로 사용하려던 중`
  }

  return `${productLabel}을 실제로 써보려던 중`
}

function inferProjectAlternativeLabel(
  text: string,
  featureLabel: string,
  productLabel: string
) {
  if (/빛|색\s*변화|조명|무드등|램프/i.test(text)) {
    return '기존 조명이나 스위치로 임시 조절함'
  }

  if (/센서|감지|IoT|스마트|앱|디지털|기기/i.test(text)) {
    return '스마트폰이나 기존 기기로 따로 확인함'
  }

  if (/가구|인테리어|소품|오브제/i.test(text)) {
    return '기존 가구나 소품 배치를 바꿔서 버팀'
  }

  if (/패션|악세서리|액세서리|착용|웨어러블/i.test(text)) {
    return '기존 액세서리나 소지품으로 대체함'
  }

  if (featureLabel && featureLabel !== '핵심 기능') {
    return `기존 방법으로 ${featureLabel}을 대신 처리함`
  }

  return `${productLabel} 없이 기존 방식으로 임시 대응함`
}

function buildProjectHintContext(project: ProjectRecord | null) {
  const requirements = getRequirements(project)
  const snapshot = buildProjectStartSnapshot(project)
  const categoryValue = getFirstListValue(snapshot.category)
  const featureValue = getFirstListValue(snapshot.features)
  const rawIdea = getHintRequirementText(requirements, 'idea')
  const rawUsage = getHintRequirementText(requirements, 'usage')
  const productLabel = compactHintText(
    categoryValue || snapshot.title,
    '이 제품',
    18
  )
  const categoryLabel = compactHintText(
    categoryValue || snapshot.category,
    productLabel,
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
  const purposeLabel = inferProjectPurposeLabel(rawUsage || snapshot.usage, productLabel)
  const goalLabel = inferProjectGoalLabel(snapshot.goal)
  const formFactorLabel = inferProjectFormFactorLabel(snapshot.size, productLabel)
  const projectText = [
    snapshot.title,
    snapshot.category,
    snapshot.features,
    snapshot.goal,
    snapshot.size,
    snapshot.usage,
    rawIdea,
  ].join(' ')

  return {
    domain: inferProjectHintDomain(projectText),
    productLabel,
    categoryLabel,
    ideaLabel,
    featureLabel,
    purposeLabel,
    goalLabel,
    formFactorLabel,
    spaceLabel: inferProjectSpaceLabel(projectText, productLabel),
    situationLabel: inferProjectSituationLabel(projectText, productLabel),
    painLabel: inferProjectPainLabel(projectText),
    audienceLabel: inferProjectAudienceLabel(projectText, productLabel),
    actionLabel: inferProjectActionLabel(projectText, featureLabel, productLabel),
    alternativeLabel: inferProjectAlternativeLabel(
      projectText,
      featureLabel,
      productLabel
    ),
  }
}

function buildPersonaStageExampleChoices(
  context: ProjectHintContext
): Partial<Record<ContextualHintKind, [string, string, string]>> {
  const {
    domain,
    productLabel,
    featureLabel,
    spaceLabel,
    painLabel,
  } = context

  const generic: Partial<Record<ContextualHintKind, [string, string, string]>> = {
    persona_user: [
      `20대 후반 ${productLabel}을 직접 써보려는 직장인`,
      `30대 초반 ${spaceLabel}을 자주 관리하는 프리랜서`,
      '40대 초반 판매 가능성을 검토하는 소상공인',
    ],
    persona_usage: [
      `${spaceLabel}에서 기존 방식이 번거롭게 느껴질 때`,
      `${featureLabel}을 바로 써야 하는 상황이 생길 때`,
      `${productLabel}을 두거나 꺼내 쓰기 전에 망설일 때`,
    ],
    persona_pain: [
      `사용 전 준비 과정이 매번 길어짐`,
      `${featureLabel}까지 이어지는 단계가 번거로움`,
      `${spaceLabel}에 자연스럽게 맞지 않아 손이 덜 감`,
    ],
    persona_decision: [
      `처음 봐도 쓰임이 바로 이해되는지`,
      `기존 방식보다 준비와 조작이 줄어드는지`,
      `공간에 두고 오래 써도 부담스럽지 않은지`,
    ],
    problem_scene: [
      `기존 방식으로 처리하다 시간이 길어질 때`,
      `${spaceLabel}에서 사용 위치를 다시 맞춰야 할 때`,
      `${productLabel} 없이 대체품으로 버티다 한계를 느낄 때`,
    ],
    problem_action: [
      `${featureLabel}을 써서 원하는 상태를 맞추려 함`,
      `${productLabel}을 꺼내거나 배치해 바로 써보려 함`,
      `기존 도구를 치우고 새 방식으로 처리하려 함`,
    ],
    problem_interruption: [
      `사용 위치나 순서가 매번 달라짐`,
      `${featureLabel}까지 이어지는 과정이 번거로움`,
      `${spaceLabel}의 주변 요소가 사용을 방해함`,
    ],
    problem_workaround: [
      `기존 제품을 억지로 조합해서 해결함`,
      `손으로 위치나 상태를 계속 다시 맞춤`,
      `불편을 참고 다음 사용까지 미룸`,
    ],
    problem_residue: [
      `사용 후에도 같은 번거로움이 다시 생김`,
      `제품을 써야 할 이유가 선명하게 남지 않음`,
      `${painLabel}이 다른 상황에서도 반복됨`,
    ],
    problem_change: [
      `준비와 조작이 줄어 바로 사용할 수 있음`,
      `기존 방식보다 덜 고민하고 행동할 수 있음`,
      `${spaceLabel}에서 제품의 역할이 자연스럽게 자리 잡음`,
    ],
  }

  switch (domain) {
    case 'lighting':
      return {
        ...generic,
        persona_user: [
          '20대 후반 원룸에서 생활하는 직장인',
          '30대 초반 집에서 일하는 프리랜서',
          '40대 초반 작은 매장의 분위기를 관리하는 운영자',
        ],
        persona_usage: [
          '퇴근 후 방의 밝기를 낮춰 쉬려고 할 때',
          '잠들기 전 침대 옆 조도를 맞출 때',
          '손님을 맞기 전 거실 분위기를 정리할 때',
        ],
        persona_pain: [
          '스위치나 앱 조작이 매번 번거로움',
          '눈부심과 어두움 사이에서 밝기를 맞추기 어려움',
          '공간 분위기와 조명 색이 따로 느껴짐',
        ],
        persona_decision: [
          '빛이 눈에 부담 없이 부드러운지',
          '조작 없이 원하는 분위기로 바뀌는지',
          '꺼져 있을 때도 공간에 잘 어울리는지',
        ],
        problem_scene: [
          '퇴근 후 방을 쉬는 분위기로 바꾸려 할 때',
          '잠들기 전 강한 조명을 끄고 싶을 때',
          '손님이 오기 전 거실 분위기를 맞출 때',
        ],
        problem_action: [
          '스위치를 찾거나 앱을 열어 조도를 바꾸려 함',
          '눈부심을 줄이려고 위치나 각도를 조정함',
          '공간 분위기에 맞는 색을 고르려 함',
        ],
        problem_interruption: [
          '조작 과정이 길어 쉬는 흐름이 깨짐',
          '밝기가 너무 세거나 어두워 다시 조정해야 함',
          '전원선이나 위치 때문에 원하는 자리에 두기 어려움',
        ],
        problem_workaround: [
          '기존 스탠드와 천장 조명을 번갈아 켬',
          '휴대폰 손전등이나 간접 조명으로 임시 조절함',
          '커튼이나 가구 위치로 빛을 가려 봄',
        ],
        problem_residue: [
          '매일 비슷한 조작을 다시 반복하게 됨',
          '쉬고 싶은 분위기가 바로 만들어지지 않음',
          '조명이 공간의 일부라기보다 따로 놓인 물건처럼 보임',
        ],
        problem_change: [
          '방에 들어오면 바로 원하는 밝기로 전환됨',
          '쉬는 시간과 작업 시간이 빛으로 구분됨',
          '조명을 켜지 않은 상태에서도 공간이 정돈돼 보임',
        ],
      }
    case 'interior':
      return {
        ...generic,
        persona_user: [
          '20대 후반 원룸을 꾸미는 직장인',
          '30대 초반 신혼집을 정리하는 직장인',
          '소규모 카페나 쇼룸을 운영하는 자영업자',
        ],
        persona_usage: [
          '새 물건을 들여놓기 전 전체 분위기를 맞춰볼 때',
          '손님이 오기 전 눈에 띄는 자리를 정리할 때',
          '사진을 찍거나 전시할 공간을 준비할 때',
        ],
        persona_pain: [
          '예쁘지만 실제 공간에서는 따로 떠 보임',
          '놓을 자리를 바꿔도 정돈된 느낌이 잘 안 남',
          '관리하거나 옮기는 과정이 예상보다 번거로움',
        ],
        persona_decision: [
          '기존 가구와 색감이 자연스럽게 이어지는지',
          '작은 공간에서도 답답해 보이지 않는지',
          '관리와 이동이 부담스럽지 않은지',
        ],
        problem_scene: [
          '빈 공간이 허전해 보이지만 무엇을 둘지 애매할 때',
          '손님이 보기 전에 공간의 첫인상을 정리하고 싶을 때',
          '사진이나 전시 구도를 맞추다 균형이 깨져 보일 때',
        ],
        problem_action: [
          '소품 위치를 옮겨 보며 어울리는 자리를 찾음',
          '가구와 색감을 맞춰 보려고 주변 물건을 치움',
          '공간의 시선이 모이는 지점을 만들려 함',
        ],
        problem_interruption: [
          '기존 물건과 크기나 색감이 충돌함',
          '놓을 자리가 바뀔 때마다 분위기가 흐트러짐',
          '관리하기 어려워 금방 방치된 느낌이 생김',
        ],
        problem_workaround: [
          '기존 소품을 여러 개 겹쳐 배치함',
          '사진이나 패브릭으로 빈 공간을 임시로 가림',
          '마음에 들지 않는 배치를 그대로 둠',
        ],
        problem_residue: [
          '공간의 첫인상이 여전히 정돈되지 않음',
          '사용할 때보다 치우거나 관리할 때 부담이 큼',
          '소품이 공간의 의미보다 장식으로만 남음',
        ],
        problem_change: [
          '작은 배치만으로도 공간의 중심이 생김',
          '기존 물건과 조화되어 오래 두기 쉬워짐',
          '손님이나 사용자가 공간의 의도를 바로 느낄 수 있음',
        ],
      }
    case 'furniture':
      return {
        ...generic,
        persona_user: [
          '30대 초반 집에서 오래 머무는 직장인',
          '20대 후반 작업 공간을 직접 꾸미는 프리랜서',
          '작은 매장이나 스튜디오를 운영하는 자영업자',
        ],
        persona_usage: [
          '앉거나 기대는 시간이 길어질 때',
          '작은 공간에 물건과 동선을 함께 맞춰야 할 때',
          '기존 가구가 생활 방식과 맞지 않는다고 느낄 때',
        ],
        persona_pain: [
          '크기와 동선이 맞지 않아 공간이 답답해짐',
          '사용 자세가 불편해 오래 쓰기 어려움',
          '수납이나 이동이 필요한데 구조가 따라주지 않음',
        ],
        persona_decision: [
          '몸에 닿는 부분이 오래 써도 편한지',
          '공간 크기와 동선에 맞게 놓을 수 있는지',
          '조립과 이동, 관리가 현실적인지',
        ],
        problem_scene: [
          '작은 방에서 가구 위치를 다시 잡아야 할 때',
          '오래 앉아 있거나 기대다 불편함이 쌓일 때',
          '물건을 치우고 꺼내는 동선이 계속 꼬일 때',
        ],
        problem_action: [
          '가구를 옮겨 보며 편한 위치를 찾음',
          '앉는 자세나 높이를 다시 맞추려 함',
          '수납과 사용 동선을 한 번에 해결하려 함',
        ],
        problem_interruption: [
          '크기 때문에 문이나 이동 동선이 막힘',
          '높이나 각도가 몸에 맞지 않음',
          '필요한 물건을 꺼내려면 주변을 다시 치워야 함',
        ],
        problem_workaround: [
          '방석이나 받침을 추가해 임시로 맞춤',
          '가구 위치를 자주 바꾸며 버팀',
          '수납함이나 보조 테이블을 따로 더 둠',
        ],
        problem_residue: [
          '공간은 차지하지만 사용 만족이 낮게 남음',
          '몸의 피로와 정리 부담이 계속 쌓임',
          '생활 방식이 가구에 맞춰져 버림',
        ],
        problem_change: [
          '자세와 동선이 자연스럽게 맞아짐',
          '작은 공간에서도 답답함이 줄어듦',
          '수납과 사용 행동이 한 자리에서 이어짐',
        ],
      }
    case 'wearable':
      return {
        ...generic,
        persona_user: [
          '20대 후반 출근 룩을 신경 쓰는 직장인',
          '30대 초반 외출이 잦은 프리랜서',
          '20대 초반 취향이 드러나는 소품을 고르는 대학생',
        ],
        persona_usage: [
          '외출 전 옷차림의 마지막 포인트를 고를 때',
          '오래 착용해도 거슬리지 않아야 할 때',
          '일상과 격식 있는 자리 모두에 맞춰야 할 때',
        ],
        persona_pain: [
          '예뻐 보여도 착용하면 불편하거나 무거움',
          '옷과 어울리지 않아 자주 빼게 됨',
          '보관과 휴대 중 형태가 쉽게 흐트러짐',
        ],
        persona_decision: [
          '오래 착용해도 걸리거나 무겁지 않은지',
          '여러 스타일에 자연스럽게 어울리는지',
          '보관과 휴대 중 손상 걱정이 적은지',
        ],
        problem_scene: [
          '외출 직전 착용할 소품을 고르다 망설일 때',
          '하루 종일 착용해야 해서 편안함이 중요할 때',
          '옷차림은 맞췄지만 포인트가 부족해 보일 때',
        ],
        problem_action: [
          '옷에 맞춰 착용 위치나 조합을 바꿔 봄',
          '거울 앞에서 무게감과 인상을 확인함',
          '가방에 넣어도 형태가 유지될지 확인함',
        ],
        problem_interruption: [
          '피부나 옷에 걸려 착용감이 거슬림',
          '스타일이 강해 다른 옷과 맞추기 어려움',
          '보관 중 엉키거나 흠집이 생김',
        ],
        problem_workaround: [
          '비슷한 기존 액세서리로 대체함',
          '불편한 부분을 참고 짧게만 착용함',
          '어울리지 않으면 아예 착용을 포기함',
        ],
        problem_residue: [
          '구매 후에도 자주 손이 가지 않음',
          '착용감 때문에 좋은 첫인상이 오래가지 않음',
          '특정 옷차림에만 맞아 활용도가 낮아짐',
        ],
        problem_change: [
          '외출 전 선택 시간이 줄어듦',
          '착용한 상태를 오래 의식하지 않아도 됨',
          '일상복과 특별한 자리 모두에 자연스럽게 맞음',
        ],
      }
    case 'digital':
      return {
        ...generic,
        persona_user: [
          '20대 후반 새 기기를 빠르게 써보는 직장인',
          '30대 초반 업무 도구를 직접 관리하는 프리랜서',
          '20대 초반 기능보다 사용 편의성을 먼저 보는 대학생',
        ],
        persona_usage: [
          '상태를 바로 확인해야 하는데 화면이나 알림이 흩어질 때',
          '앱이나 기기 조작을 반복해야 할 때',
          '새 기능을 익히기 전에 사용 흐름이 끊길 때',
        ],
        persona_pain: [
          '조작 단계가 많아 바로 쓰기 어려움',
          '상태 확인이 늦어 다음 행동을 정하기 어려움',
          '기기와 앱 사이의 연결이 자연스럽지 않음',
        ],
        persona_decision: [
          '처음 써도 핵심 기능을 바로 이해할 수 있는지',
          '알림과 상태 표시가 필요한 만큼만 보이는지',
          '기존 기기나 앱과 연결이 번거롭지 않은지',
        ],
        problem_scene: [
          '급하게 상태를 확인해야 하는데 정보가 흩어져 있을 때',
          '반복 조작 때문에 하던 일을 멈춰야 할 때',
          '새 기능을 쓰려다 설정 과정에서 막힐 때',
        ],
        problem_action: [
          '앱을 열어 상태를 확인하고 다음 행동을 정함',
          '기기 버튼이나 화면을 반복해서 조작함',
          '설정값을 바꿔 원하는 작동 상태를 맞춤',
        ],
        problem_interruption: [
          '설정 메뉴가 복잡해 원하는 기능을 찾기 어려움',
          '알림이 많거나 부족해 상태 판단이 늦어짐',
          '연결이 끊겨 같은 조작을 다시 해야 함',
        ],
        problem_workaround: [
          '스마트폰 앱을 여러 번 열어 따로 확인함',
          '기본 설정만 유지한 채 세부 기능은 포기함',
          '기존 기기와 수동 기록으로 보완함',
        ],
        problem_residue: [
          '기능은 많지만 실제로 쓰는 기능은 줄어듦',
          '상태를 놓쳐 다시 확인하는 일이 반복됨',
          '새 기기를 써도 생활 흐름이 크게 나아지지 않음',
        ],
        problem_change: [
          '필요한 상태가 한눈에 보여 다음 행동이 빨라짐',
          '반복 조작 없이 원하는 기능으로 바로 이어짐',
          '기기 사용이 일상 흐름 안에 자연스럽게 들어옴',
        ],
      }
    case 'study':
      return {
        ...generic,
        persona_user: [
          '20대 초반 시험을 준비하는 대학생',
          '20대 후반 자격증 공부를 병행하는 직장인',
          '30대 초반 집에서 집중 시간을 관리하는 프리랜서',
        ],
        persona_usage: [
          '책상에 앉았지만 바로 몰입이 안 될 때',
          '알림이나 주변 소리 때문에 다시 집중해야 할 때',
          '공부와 휴식 시간을 스스로 나눠야 할 때',
        ],
        persona_pain: [
          '시작 전 준비와 정리가 길어짐',
          '방해를 한 번 받으면 다시 몰입하기 어려움',
          '공부 시간이 길어질수록 자세와 리듬이 흐트러짐',
        ],
        persona_decision: [
          '집중 시작까지 걸리는 시간이 줄어드는지',
          '방해를 줄이되 부담스럽지 않게 쓸 수 있는지',
          '책상 위에 오래 두어도 거슬리지 않는지',
        ],
        problem_scene: [
          '공부를 시작하려고 앉았지만 주변이 산만할 때',
          '한 번 흐트러진 뒤 다시 책상으로 돌아와야 할 때',
          '긴 학습 시간 중 쉬는 타이밍을 놓칠 때',
        ],
        problem_action: [
          '책상 위 물건을 치우고 공부를 시작하려 함',
          '휴대폰 알림을 줄이고 다시 집중하려 함',
          '쉬는 시간 뒤 같은 흐름으로 돌아오려 함',
        ],
        problem_interruption: [
          '알림이나 주변 소리가 계속 주의를 끔',
          '정리되지 않은 책상 때문에 시작이 늦어짐',
          '시간 감각이 흐려져 쉬는 타이밍을 놓침',
        ],
        problem_workaround: [
          '휴대폰을 멀리 두거나 무음으로 바꿈',
          '타이머 앱과 메모를 따로 켜 둠',
          '책상 위 물건을 매번 다시 정리함',
        ],
        problem_residue: [
          '공부를 시작하는 데 에너지가 계속 쓰임',
          '짧은 방해에도 집중 시간이 쉽게 끊김',
          '학습 리듬이 매일 일정하게 남지 않음',
        ],
        problem_change: [
          '앉으면 바로 공부 모드로 전환됨',
          '방해 뒤에도 짧은 절차로 다시 돌아옴',
          '공부와 휴식의 리듬이 눈에 보이게 정리됨',
        ],
      }
    case 'kitchen':
      return {
        ...generic,
        persona_user: [
          '30대 초반 집밥을 자주 준비하는 직장인',
          '20대 후반 좁은 주방을 쓰는 1인 가구',
          '40대 초반 카페나 작은 매장을 운영하는 자영업자',
        ],
        persona_usage: [
          '요리 전 재료와 도구를 빠르게 꺼내야 할 때',
          '식사 후 정리와 보관이 한꺼번에 밀릴 때',
          '좁은 조리대에서 여러 행동을 이어가야 할 때',
        ],
        persona_pain: [
          '재료와 도구를 찾는 시간이 길어짐',
          '사용 후 세척과 보관이 번거로움',
          '조리대가 금방 복잡해져 다음 행동이 막힘',
        ],
        persona_decision: [
          '세척과 보관이 간단한지',
          '좁은 공간에서도 바로 꺼내 쓸 수 있는지',
          '음식 준비 흐름을 방해하지 않는지',
        ],
        problem_scene: [
          '요리 중 필요한 도구를 바로 찾아야 할 때',
          '식사 후 정리할 물건이 한꺼번에 쌓일 때',
          '좁은 조리대에서 다음 재료를 올려야 할 때',
        ],
        problem_action: [
          '재료를 꺼내고 손질 순서를 맞추려 함',
          '사용한 도구를 씻고 다시 보관하려 함',
          '조리대 위 공간을 확보하려 함',
        ],
        problem_interruption: [
          '도구 위치가 달라져 찾는 시간이 늘어남',
          '물기나 음식물이 남아 바로 보관하기 어려움',
          '공간이 부족해 준비와 정리가 계속 겹침',
        ],
        problem_workaround: [
          '임시로 접시나 쟁반 위에 물건을 모아 둠',
          '자주 쓰는 도구만 밖에 꺼내 둠',
          '정리는 식사 후로 미뤄 둠',
        ],
        problem_residue: [
          '요리 후 주방이 오래 어수선하게 남음',
          '다음 사용 전 다시 정리해야 함',
          '보관 방식이 일정하지 않아 같은 문제가 반복됨',
        ],
        problem_change: [
          '준비와 정리가 같은 흐름 안에서 이어짐',
          '자주 쓰는 도구를 바로 찾을 수 있음',
          '좁은 조리대에서도 사용 순서가 덜 꼬임',
        ],
      }
    case 'rest':
      return {
        ...generic,
        persona_user: [
          '30대 초반 퇴근 후 회복 시간이 필요한 직장인',
          '20대 후반 작은 방에서 생활하는 1인 가구',
          '40대 초반 가족의 생활 공간을 관리하는 보호자',
        ],
        persona_usage: [
          '퇴근 후 집에 들어와 긴장을 풀고 싶을 때',
          '잠들기 전 주변을 차분하게 정리할 때',
          '거실에서 휴식과 생활 행동이 섞일 때',
        ],
        persona_pain: [
          '쉬려고 해도 주변 자극이 계속 남음',
          '잠들기 전 정리와 준비가 길어짐',
          '생활 공간이 휴식 분위기로 잘 전환되지 않음',
        ],
        persona_decision: [
          '긴장을 낮추는 데 실제로 도움이 되는지',
          '잠들기 전에도 부담 없이 쓸 수 있는지',
          '생활 공간에 조용히 어울리는지',
        ],
        problem_scene: [
          '집에 돌아와 쉬고 싶지만 공간이 어수선할 때',
          '잠들기 전 작은 자극에도 신경이 쓰일 때',
          '거실에서 휴식과 정리 행동이 겹칠 때',
        ],
        problem_action: [
          '주변 물건을 치우고 몸을 쉬게 하려 함',
          '잠들기 전 필요한 것만 남기고 정리함',
          '휴식 분위기로 공간을 바꾸려 함',
        ],
        problem_interruption: [
          '눈에 띄는 물건과 소리가 계속 거슬림',
          '정리할 일이 남아 휴식으로 바로 넘어가지 못함',
          '공간 분위기가 쉬는 상태로 잘 바뀌지 않음',
        ],
        problem_workaround: [
          '불을 끄거나 음악을 틀어 분위기만 바꿈',
          '눈에 거슬리는 물건을 한쪽으로 밀어 둠',
          '정리는 다음 날로 미루고 그냥 쉼',
        ],
        problem_residue: [
          '쉰 뒤에도 공간이 정돈되지 않은 느낌이 남음',
          '잠들기 전 같은 정리 과정을 반복함',
          '휴식 시간이 생활 관리 시간처럼 느껴짐',
        ],
        problem_change: [
          '집에 들어오면 휴식 상태로 더 빨리 전환됨',
          '잠들기 전 정리 부담이 줄어듦',
          '생활 공간이 더 차분하게 유지됨',
        ],
      }
    case 'activity':
      return {
        ...generic,
        persona_user: [
          '20대 후반 운동을 생활화하려는 직장인',
          '30대 초반 이동이 잦은 프리랜서',
          '20대 초반 야외 활동을 자주 하는 대학생',
        ],
        persona_usage: [
          '외출 전 필요한 물건을 빠르게 챙길 때',
          '운동 중 손이나 몸의 움직임을 방해받지 않아야 할 때',
          '이동 중에도 바로 꺼내 쓰기 쉬워야 할 때',
        ],
        persona_pain: [
          '휴대할 때 무게나 부피가 부담됨',
          '움직이는 중 꺼내고 넣는 과정이 번거로움',
          '땀이나 충격 때문에 관리가 신경 쓰임',
        ],
        persona_decision: [
          '움직임을 방해하지 않는지',
          '가볍고 빠르게 꺼내 쓸 수 있는지',
          '야외 사용 후 관리가 쉬운지',
        ],
        problem_scene: [
          '운동을 시작하기 전 물건을 챙기다 빠뜨릴 때',
          '이동 중 필요한 기능을 바로 써야 할 때',
          '야외에서 땀이나 충격을 신경 써야 할 때',
        ],
        problem_action: [
          '가방이나 주머니에서 빠르게 꺼내려 함',
          '움직임을 멈추지 않고 상태를 확인함',
          '사용 후 바로 정리하거나 보관하려 함',
        ],
        problem_interruption: [
          '무게나 흔들림 때문에 움직임이 불편함',
          '꺼내는 과정에서 운동 흐름이 끊김',
          '사용 후 오염이나 파손이 걱정됨',
        ],
        problem_workaround: [
          '필요한 물건을 최소한으로 줄여 챙김',
          '기존 파우치나 가방에 따로 넣어 둠',
          '불편한 기능은 운동 중 사용하지 않음',
        ],
        problem_residue: [
          '챙기는 과정이 번거로워 사용 빈도가 낮아짐',
          '움직임 중에는 제품을 제대로 쓰지 못함',
          '사용 후 관리 부담이 다음 사용을 막음',
        ],
        problem_change: [
          '외출 전 준비 시간이 줄어듦',
          '움직임을 멈추지 않고 필요한 기능을 쓸 수 있음',
          '사용 후 정리까지 간단하게 이어짐',
        ],
      }
    case 'hygiene':
      return {
        ...generic,
        persona_user: [
          '30대 초반 위생 관리를 꼼꼼히 하는 직장인',
          '20대 후반 욕실 수납을 정리하려는 1인 가구',
          '40대 초반 가족의 욕실 용품을 관리하는 보호자',
        ],
        persona_usage: [
          '세면이나 샤워 후 바로 정리해야 할 때',
          '물기가 남은 제품을 위생적으로 보관해야 할 때',
          '매일 쓰는 관리 용품을 빠르게 꺼내야 할 때',
        ],
        persona_pain: [
          '물기와 오염 때문에 보관이 찝찝함',
          '세척과 건조 과정이 번거로움',
          '욕실 공간이 좁아 물건이 쉽게 섞임',
        ],
        persona_decision: [
          '세척과 건조가 쉽게 이어지는지',
          '물기 있는 공간에서도 안정적으로 둘 수 있는지',
          '매일 써도 관리 부담이 적은지',
        ],
        problem_scene: [
          '사용 후 물기 있는 상태로 둘 곳이 애매할 때',
          '아침 준비 중 관리 용품을 빠르게 찾아야 할 때',
          '욕실 정리 후에도 위생 상태가 신경 쓰일 때',
        ],
        problem_action: [
          '사용한 물건을 씻고 말리려 함',
          '욕실 선반에서 필요한 용품을 바로 찾음',
          '물기가 튀지 않게 위치를 다시 잡음',
        ],
        problem_interruption: [
          '물기가 고여 냄새나 오염이 걱정됨',
          '좁은 선반에서 물건이 쉽게 섞임',
          '세척 후 말릴 위치가 부족함',
        ],
        problem_workaround: [
          '수건이나 휴지 위에 임시로 올려둠',
          '기존 컵이나 통에 여러 용품을 함께 꽂아 둠',
          '완전히 마르기 전 서랍에 넣어 둠',
        ],
        problem_residue: [
          '매일 쓰는 물건인데도 위생 걱정이 남음',
          '정리해도 욕실이 금방 어수선해짐',
          '세척과 건조를 따로 챙겨야 함',
        ],
        problem_change: [
          '사용 후 바로 씻고 말리는 흐름이 생김',
          '욕실 선반에서 필요한 물건을 바로 찾음',
          '위생 관리가 별도 일이 아니라 일상 동작에 붙음',
        ],
      }
    default:
      return generic
  }
}

function buildContextualHintChoices(
  context: ProjectHintContext,
  kind: ContextualHintKind
): [string, string, string] {
  const personaStageChoices = buildPersonaStageExampleChoices(context)[kind]

  if (personaStageChoices) {
    return personaStageChoices
  }

  const {
    productLabel,
    categoryLabel,
    featureLabel,
    purposeLabel,
    goalLabel,
    formFactorLabel,
    spaceLabel,
    situationLabel,
    painLabel,
    audienceLabel,
    actionLabel,
    alternativeLabel,
  } = context
  const productTerm = `‘${productLabel}’`
  const featureTerm = `‘${featureLabel}’`

  const choices: [string, string, string] = (() => {
    switch (kind) {
      case 'persona_user':
        return [
          `20-30대 ${audienceLabel}으로 ${situationLabel}을 자주 겪는 사람`,
          `30-40대 ${spaceLabel}에서 ${productTerm}을 직접 써볼 사람`,
          `${purposeLabel}에 맞는 ${categoryLabel} 제품을 비교해볼 사람`,
        ]
      case 'persona_usage':
        return [
          `${spaceLabel}에서 ${situationLabel}이 생기는 때`,
          `${actionLabel}`,
          `${formFactorLabel}가 실제로 놓이거나 쓰이는 때`,
        ]
      case 'persona_pain':
        return [
          `${painLabel}`,
          `${featureTerm}이 기대한 순간에 바로 이어지지 않는 불편`,
          `${formFactorLabel}가 ${spaceLabel}에 잘 맞지 않는 불편`,
        ]
      case 'persona_decision':
        return [
          `${featureTerm}이 기대한 상황에서 이해되고 작동하는지`,
          `${formFactorLabel}가 ${spaceLabel}에 자연스럽게 맞는지`,
          `${goalLabel}에서 우선 검증할 부분이 분명한지`,
        ]
      case 'problem_scene':
        return [
          `${spaceLabel}에서 ${situationLabel}이 생긴 순간`,
          `${actionLabel.replace(/려던 중$/, '려는 순간').replace(/하던 중$/, '하는 순간')}`,
          `${painLabel} 때문에 기존 방식으로는 부족하다고 느낀 순간`,
        ]
      case 'problem_action':
        return [
          `${actionLabel}`,
          `${featureTerm}을 통해 원하는 상태를 만들려던 중`,
          `${spaceLabel}에서 ${productTerm}을 배치하거나 확인하던 중`,
        ]
      case 'problem_interruption':
        return [
          `${painLabel}`,
          `${featureTerm}이 필요한 순간에 바로 이어지지 않음`,
          `${spaceLabel}의 제약 때문에 ${productTerm} 사용이 자연스럽지 않음`,
        ]
      case 'problem_workaround':
        return [
          `${alternativeLabel}`,
          `${featureTerm} 대신 수동으로 확인하거나 조절함`,
          `${painLabel}이 생기는 상황을 미루거나 다른 제품으로 우회함`,
        ]
      case 'problem_residue':
        return [
          `${painLabel}이 반복됨`,
          `${productTerm}을 쓰는 이유가 사용자에게 선명하게 남지 않음`,
          `${spaceLabel}에서 불편한 사용 패턴이 계속 남음`,
        ]
      case 'problem_change':
        return [
          `${situationLabel}이 더 자연스럽고 짧아짐`,
          `${featureTerm} 덕분에 사용자가 덜 고민하고 행동함`,
          `${spaceLabel}에서 ${productTerm}의 역할이 분명해짐`,
        ]
      case 'experience_emotion_entry':
        return [
          `처음 봐도 ${productTerm}의 쓰임이 이해되는 신뢰감`,
          `${productTerm} 아이디어가 기대감으로 이어짐`,
          `${spaceLabel}에 두거나 가까이 둘 때 어색하지 않은 편안함`,
        ]
      case 'experience_emotion_relief':
        return [
          `${painLabel}이 줄어드는 안도감`,
          `${featureTerm}이 번거로운 행동을 덜어주는 가벼움`,
          `${situationLabel}이 매끄러워지는 안정감`,
        ]
      case 'experience_emotion_after':
        return [
          `${productTerm}을 다시 쓰고 싶다는 만족감`,
          `${painLabel}이 줄었다는 체감`,
          `${spaceLabel}이 의도한 상태로 남는 여유감`,
        ]
      case 'experience_behavior_start':
        return [
          `${productTerm}을 처음 사용하기까지 망설임이 줄어듦`,
          `${featureTerm}을 바로 이해하고 써봄`,
          `${spaceLabel}에서 손이 자연스럽게 감`,
        ]
      case 'experience_behavior_recovery':
        return [
          `${painLabel}이 생겨도 쉽게 조정하고 다시 사용함`,
          `${featureTerm}을 통해 원하는 상태로 빠르게 돌아감`,
          `${spaceLabel}에서 위치나 사용 방식을 다시 맞춤`,
        ]
      case 'experience_behavior_routine':
        return [
          `${productTerm} 사용이 ${situationLabel} 전후의 자연스러운 습관이 됨`,
          `${featureTerm}을 기준으로 사용자가 스스로 조절함`,
          `${spaceLabel}에 두는 위치와 사용 시점이 일정해짐`,
        ]
      case 'experience_space_impression':
        return [
          `${formFactorLabel}가 ${spaceLabel}에 자연스럽게 놓이는 인상`,
          `${productTerm}의 역할이 바로 보이는 인상`,
          `${productTerm}이 ${purposeLabel}와 잘 맞아 보이는 인상`,
        ]
      case 'experience_space_meaning':
        return [
          `${spaceLabel}을 더 안정적이거나 쓰기 편하게 느낌`,
          `${productTerm}이 ${situationLabel}의 기준점이 됨`,
          `${featureTerm}이 공간 안에서 필요한 신호가 됨`,
        ]
      case 'experience_space_relation':
        return [
          `${spaceLabel}의 기존 물건과 충돌하지 않음`,
          `${formFactorLabel}가 주변 동선이나 사용 흐름을 방해하지 않음`,
          `${productTerm}이 주변 행동을 조용히 도와줌`,
        ]
      case 'relationship_interruption':
        return [
          `${painLabel}을 만드는 기존 대안과 거리를 둠`,
          `${featureTerm}이 번거로운 행동을 줄이는 기준이 됨`,
          `${productTerm}이 다른 제품으로 이탈하지 않게 붙잡음`,
        ]
      case 'relationship_space':
        return [
          `${spaceLabel}이 ${productTerm}의 주 사용 맥락이 됨`,
          `${formFactorLabel}가 공간의 분위기나 동선에 맞음`,
          `${featureTerm}이 공간 안 행동을 바꿈`,
        ]
      case 'relationship_time':
        return [
          `${situationLabel}의 시작과 끝이 분명해짐`,
          `${featureTerm}이 필요한 타이밍을 놓치지 않게 함`,
          `${painLabel}을 줄이기 위해 사용 리듬을 다시 잡음`,
        ]
      case 'problem_revision':
        return [
          `${spaceLabel}의 실제 사용 장면을 더 구체화`,
          `${painLabel}을 프로젝트 맥락에 맞게 더 선명하게 정리`,
          `${featureTerm}이 해결해야 할 필요를 ${goalLabel}에 맞춰 재정리`,
        ]
    }
  })()

  return choices.map((choice) =>
    choice.replace(/^[^:：]{1,12}\s*단서\s*[:：]\s*/i, '').trim()
  ) as [string, string, string]
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
    /(?:감정\s*2:\s*)?(?:방해나\s*압박|불편함)이\s*줄어들\s*때\s*어떤\s*감정\s*변화가\s*생기면\s*좋을까요\?/i,
  emotionAfter:
    /(?:감정\s*3:\s*)?사용을\s*마친\s*뒤\s*사용자가\s*어떤\s*감정을\s*남기길\s*원하나요\?/i,
  behaviorStart:
    /(?:행동\s*1:\s*)?제품이\s*사용자의\s*시작\s*행동을\s*어떻게\s*도와야\s*할까요\?/i,
  behaviorRecovery:
    /(?:행동\s*2:\s*)?(?:흐름이\s*끊겼을|사용이\s*매끄럽지\s*않을)\s*때\s*어떤\s*회복\s*행동이\s*생기면\s*좋을까요\?/i,
  behaviorRoutine:
    /(?:행동\s*3:\s*)?사용자가\s*(?:시간과\s*루틴|사용\s*시점과\s*반복\s*행동)을\s*어떻게\s*다루게\s*되길\s*원하나요\?/i,
  spaceImpression:
    /(?:공간\s*1:\s*)?제품이\s*놓인\s*공간에서\s*어떤\s*첫인상을\s*주면\s*좋을까요\?/i,
  spaceMeaning:
    /(?:공간\s*2:\s*)?사용자가\s*그\s*공간을\s*어떻게\s*느끼게\s*만들면\s*좋을까요\?/i,
  spaceRelation:
    /(?:공간\s*3:\s*)?주변\s*물건이나\s*(?:생활|사용)\s*흐름과\s*어떤\s*관계를\s*맺으면\s*좋을까요\?/i,
}

const PERSONA_RELATIONSHIP_QUESTIONS = {
  interruption: /사용자와\s*기존\s*(?:방해|불편)\s*요소는\s*어떤\s*관계에\s*가깝나요\?/i,
  space: /사용자와\s*제품이\s*놓이는\s*공간은\s*어떤\s*관계에\s*가깝나요\?/i,
  time:
    /사용자는\s*(?:집중과\s*휴식\s*시간|.+?전후의\s*시간)을\s*어떻게\s*인식하길\s*원하나요\?/i,
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
      '불편함이 줄어들 때 어떤 감정 변화가 생기면 좋을까요?',
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
      '사용이 매끄럽지 않을 때 어떤 회복 행동이 생기면 좋을까요?',
    choices: buildContextualHintChoices(
      hintContext,
      'experience_behavior_recovery'
    ),
  })
}

function buildExperienceBehaviorRoutineQuestion(hintContext: ProjectHintContext) {
  return buildPersonaChoiceQuestion({
    question:
      '사용자가 사용 시점과 반복 행동을 어떻게 다루게 되길 원하나요?',
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
      '주변 물건이나 사용 흐름과 어떤 관계를 맺으면 좋을까요?',
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
      '기존 불편 요소와의 관계: 사용자와 기존 불편 요소는 어떤 관계에 가깝나요?',
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
    intro: '좋아요. 마지막으로 사용 흐름 속 시간을 어떻게 인식하는지 정리해볼게요.',
    question:
      `시간과의 관계: 사용자는 ${hintContext.situationLabel} 전후의 시간을 어떻게 인식하길 원하나요?`,
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
      ? /##\s*(?:시장\s*규모\s*리서치|Tam\s*Sam\s*Som)/i
      : target === 'consumption_keywords'
        ? /##\s*(?:소비\s*트렌드\s*리서치|Keywords\s*:\s*Consumption)/i
        : /##\s*(?:경쟁사\s*리서치|Positioning\s*Map\s*:\s*Brand)/i
  const fallbackHeading =
    target === 'market_size'
      ? '## Tam Sam Som'
      : target === 'consumption_keywords'
        ? '## Keywords:Consumption'
        : '## Positioning Map: Brand'

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

type MarketSizeReference = {
  label: string
  currentValue: number
  tenYearValue: number
}

function formatMarketAmount(value: number) {
  return `${value.toFixed(1)}억 달러`
}

function getMarketSizeReferences(projectText: string, category: string) {
  const references: MarketSizeReference[] = []

  if (/앱|서비스|디지털|소프트웨어|플랫폼|생산성|루틴|집중|학습/i.test(projectText)) {
    references.push({
      label: '생산성 앱 시장',
      currentValue: 119.6,
      tenYearValue: 180.9,
    })
  }

  if (/조명|무드등|램프|빛|스마트\s*조명/i.test(projectText)) {
    references.push({
      label: '스마트 조명 시장',
      currentValue: 98.6,
      tenYearValue: 173.8,
    })
  }

  if (/가구|인테리어|소품|오브제|리빙/i.test(projectText)) {
    references.push({
      label: '홈 인테리어 및 리빙 제품 시장',
      currentValue: 214.2,
      tenYearValue: 326.5,
    })
  }

  if (/웨어러블|착용|패션|액세서리|악세서리/i.test(projectText)) {
    references.push({
      label: '웨어러블 및 라이프스타일 액세서리 시장',
      currentValue: 76.4,
      tenYearValue: 143.2,
    })
  }

  if (references.length > 0) {
    return references.slice(0, 2)
  }

  return [
    {
      label: `${category && category !== '미정' ? category : '제품'} 시장`,
      currentValue: 82.4,
      tenYearValue: 128.7,
    },
    {
      label: '연관 라이프스타일 솔루션 시장',
      currentValue: 64.8,
      tenYearValue: 104.5,
    },
  ]
}

function getDecadeMarketValue(reference: MarketSizeReference, decadeIndex: number) {
  if (decadeIndex === 0) {
    return reference.currentValue
  }

  if (decadeIndex === 1) {
    return reference.tenYearValue
  }

  const firstDecadeRatio = reference.tenYearValue / reference.currentValue
  const dampingRatios = [1, firstDecadeRatio, 1.36, 1.28, 1.2, 1.14]
  let value = reference.tenYearValue

  for (let index = 2; index <= decadeIndex; index += 1) {
    value *= dampingRatios[index] ?? 1.1
  }

  return value
}

function getCombinedMarketValue(
  references: MarketSizeReference[],
  decadeIndex: number
) {
  return references.reduce(
    (sum, reference) => sum + getDecadeMarketValue(reference, decadeIndex),
    0
  )
}

function buildMarketSummarySentence(
  references: MarketSizeReference[],
  currentYear: number
) {
  return references
    .map((reference, index) => {
      const connector = index === 0 ? '은' : '도'

      return `${reference.label}${connector} ${currentYear}년 약 ${formatMarketAmount(
        reference.currentValue
      )}에서 향후 10년 후 약 ${formatMarketAmount(
        reference.tenYearValue
      )}로 성장할 것으로 전망됩니다.`
    })
    .join(' ')
}

function buildProjectGoalLine(snapshot: ReturnType<typeof buildProjectStartSnapshot>) {
  const goalSource =
    snapshot.goal ||
    snapshot.usage ||
    snapshot.ideaSummary ||
    `${snapshot.title}의 제품화 방향 정리`
  const cleaned = cleanSingleLineText(goalSource)
    .replace(/\.$/, '')
    .replace(/가 목표입니다$/, '')

  return `${cleaned}을 기준으로, 사용자가 선택할 분명한 이유와 확장 가능한 개발 방향을 만드는 것`
}

function getBrandPositioningCompetitors(projectText: string) {
  if (/조명|무드등|램프|빛|스마트\s*조명/i.test(projectText)) {
    return {
      rationalFunctional: ['Xiaomi', 'Govee'],
      premiumFunctional: ['Philips Hue', 'Lutron'],
      rationalLifestyle: ['IKEA', 'MUJI'],
      premiumLifestyle: ['Dyson Lightcycle', 'BenQ ScreenBar'],
    }
  }

  if (/앱|서비스|디지털|생산성|루틴|집중|학습|타이머/i.test(projectText)) {
    return {
      rationalFunctional: ['Forest', 'Focus To-Do'],
      premiumFunctional: ['Toggl Track', 'RescueTime'],
      rationalLifestyle: ['Notion', 'TickTick'],
      premiumLifestyle: ['reMarkable', 'Moleskine'],
    }
  }

  if (/가구|인테리어|소품|오브제|리빙/i.test(projectText)) {
    return {
      rationalFunctional: ['IKEA', 'Xiaomi'],
      premiumFunctional: ['Herman Miller', 'Steelcase'],
      rationalLifestyle: ['MUJI', 'HAY'],
      premiumLifestyle: ['Vitra', 'Fritz Hansen'],
    }
  }

  if (/웨어러블|착용|패션|액세서리|악세서리/i.test(projectText)) {
    return {
      rationalFunctional: ['Xiaomi', 'Amazfit'],
      premiumFunctional: ['Garmin', 'Fitbit'],
      rationalLifestyle: ['Casio', 'Swatch'],
      premiumLifestyle: ['Apple Watch', 'Oura Ring'],
    }
  }

  return {
    rationalFunctional: ['Xiaomi', 'IKEA'],
    premiumFunctional: ['Philips Hue', 'Garmin'],
    rationalLifestyle: ['MUJI', 'Notion'],
    premiumLifestyle: ['Apple', 'Dyson'],
  }
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
  const usage = snapshot.usage || '활용 목적 미정'
  const projectText = [
    snapshot.title,
    snapshot.category,
    snapshot.goal,
    snapshot.features,
    snapshot.usage,
    snapshot.ideaSummary,
  ].join(' ')

  if (kind === 'market_size') {
    const currentYear = new Date().getFullYear()
    const references = getMarketSizeReferences(projectText, category)
    const tam = getCombinedMarketValue(references, 1)
    const sam = tam * 0.35
    const som = sam * 0.08

    return [
      '## Tam Sam Som',
      '',
      buildMarketSummarySentence(references, currentYear),
      '',
      '**전체 시장 규모 전망**',
      `- 현재 (${currentYear}): ${formatMarketAmount(
        getCombinedMarketValue(references, 0)
      )}`,
      `- 향후 10년 후 (${currentYear + 10}): ${formatMarketAmount(
        getCombinedMarketValue(references, 1)
      )}`,
      `- 향후 20년 후 (${currentYear + 20}): ${formatMarketAmount(
        getCombinedMarketValue(references, 2)
      )}`,
      `- 향후 30년 후 (${currentYear + 30}): ${formatMarketAmount(
        getCombinedMarketValue(references, 3)
      )}`,
      `- 향후 40년 후 (${currentYear + 40}): ${formatMarketAmount(
        getCombinedMarketValue(references, 4)
      )}`,
      `- 향후 50년 후 (${currentYear + 50}): ${formatMarketAmount(
        getCombinedMarketValue(references, 5)
      )}`,
      '',
      `**01. TAM 전체시장 ${formatMarketAmount(tam)}**`,
      '- 향후 10년 기준 서비스가 진입 및 확장 가능한 글로벌 전체 시장 규모를 의미하며, 산업 성장성과 수요를 포함한 최대 시장 범위를 나타냅니다.',
      '',
      `**02. SAM 유효시장 ${formatMarketAmount(sam)}**`,
      `- ${usage} 목적과 직접 연결되는 사용 상황, 구매 가능 고객군, 실제 제품 형태와 유통 가능성을 반영한 접근 가능 시장입니다.`,
      '',
      `**03. SOM 자사목표시장 ${formatMarketAmount(som)}**`,
      '- 초기 출시 단계에서 현실적으로 확보할 수 있는 목표 시장을 의미하며, 반복 사용 니즈가 선명하고 구매 전환 가능성이 높은 고객군을 중심으로 산정합니다.',
      '',
      '**PROJECT GOAL**',
      `- ${buildProjectGoalLine(snapshot)}`,
      '',
      `${references
        .map(
          (reference) =>
            `${reference.label}은 ${currentYear}년 약 ${formatMarketAmount(
              reference.currentValue
            )}에서 향후 10년 후 약 ${formatMarketAmount(
              reference.tenYearValue
            )}로 성장할 것으로 제시됩니다.`
        )
        .join(' ')}`,
    ].join('\n')
  }

  if (kind === 'consumption_keywords') {
    return [
      '## Keywords:Consumption',
      '',
      '**한 줄 요약**',
      `- 사용자는 ${projectLabel}을 단순 기능성 제품이 아니라 자신의 루틴, 공간, 감정 만족을 함께 정돈하는 생활형 소비로 판단합니다.`,
      '',
      '**설명**',
      `사용자는 ${category} 관련 제품을 단순한 기능성 도구로만 소비하지 않고, 자신의 루틴과 사용 환경을 정돈하기 위한 생활형 소비로 인식하는 경향을 보입니다. 구매 판단에서는 실제로 도움이 되는 기능적 필요와 함께, 가까운 공간에 두고 싶다는 감성적 만족, 주변 환경과의 조화, 부담 없는 가격대가 함께 고려됩니다. 또한 사용자는 스마트폰 앱, 저가 대체재, 기존 생활 도구와 비교하며 구매 필요성을 판단하기 때문에, 소비로 이어지기 위해서는 "왜 이 제품을 따로 사야 하는가"에 대한 명확한 이유가 필요합니다.`,
      '',
      '**Keywords**',
      '- 유혹',
      '- 이탈',
      '- 알림',
      '- 과잉 연결',
      '- 거리두기',
      '- 개인 공간',
      '- 몰입 공간',
      '- 개인화',
      '- 안정감',
      '- 시간 흐름',
      '- 리듬 전환',
      '- 루틴화',
      '- 책상 정돈',
      '- 감성 만족',
      '- 기능 필요',
      '- 가격 부담',
      '- 대체재 비교',
      '- 구매 이유',
      '- 사용 지속',
      '- 조용한 효용',
      '- 공간 조화',
      '- 집중 신호',
      '- 습관 형성',
      '- 자기 조절',
      '- 작은 보상',
      '- 즉각 효용',
      '- 생활 밀착',
      '- 사용 장벽',
      '- 반복 구매',
      '- 소유 욕구',
      '- 브랜드 신뢰',
      '- 시각 정돈',
      '- 감정 전환',
      '',
      '**정리**',
      '- 소비로 연결되기 위해서는 기능의 필요성, 공간에 두고 싶은 감성, 기존 대체재 대비 차별 이유가 동시에 보여야 합니다.',
    ].join('\n')
  }

  const competitors = getBrandPositioningCompetitors(projectText)

  return [
    '## Positioning Map: Brand',
    '',
    '**포지셔닝 축**',
    '- X축: 프리미엄 가격 ↔ 합리적 가격',
    '- Y축: 기능 중심 ↔ 라이프스타일 중심',
    '',
    '**합리적 기능형**',
    '- 낮은 가격과 기본 기능을 중심으로 구매되는 제품군입니다.',
    `- 경쟁사 브랜드: ${competitors.rationalFunctional.join(', ')}`,
    '',
    '**프리미엄 기능형**',
    '- 성능, 정확성, 기술 신뢰도를 중심으로 높은 가격을 형성하는 제품군입니다.',
    `- 경쟁사 브랜드: ${competitors.premiumFunctional.join(', ')}`,
    '',
    '**합리적 라이프스타일형**',
    '- 공간 분위기와 취향을 반영하지만 접근 가능한 가격대를 유지하는 제품군입니다.',
    `- 경쟁사 브랜드: ${competitors.rationalLifestyle.join(', ')}`,
    '',
    '**프리미엄 라이프스타일형**',
    '- 디자인 완성도, 브랜드 감성, 소장 가치가 강한 제품군입니다.',
    `- 경쟁사 브랜드: ${competitors.premiumLifestyle.join(', ')}`,
    '',
    '**OUR BRAND**',
    `- ${projectLabel}은 저가 대체재처럼 기능만 강조하거나, 고가 프리미엄 제품군처럼 높은 가격 시장으로 바로 진입하기보다, 합리적인 가격대 안에서 사용자의 생활 감성과 ${usage} 맥락을 완성하는 "합리적 라이프스타일형"으로 포지셔닝하는 것이 적합합니다.`,
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
      : '다른 트렌드도 궁금하다면 위로 가서 시장 규모, 소비 트렌드, 경쟁사 위젯 중 하나를 눌러주세요.',
    '리서치가 충분하다면 다음 STEP 4. 스타일 컨셉 도출 단계로 넘어가겠습니다. 진행할까요?',
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
- 지금은 STEP 3 개발 방향성 도출 단계입니다.
- 이 단계는 1. 시장 규모, 2. 소비 트렌드, 3. 경쟁사 위젯으로 나누어 진행합니다.
- 사용자가 위젯을 누르면 해당 항목에 대해 텍스트 리서치를 제공합니다.
- 각 리서치 아래에는 시각화하기 버튼만 제공됩니다.
- 시장 규모는 Tam Sam Som 카드, 소비 트렌드는 Keywords:Consumption 카드, 경쟁사는 Positioning Map: Brand 카드로 시각화합니다.
- 한 카드가 만들어진 뒤에는 다른 트렌드도 궁금하다면 위로 가서 위젯을 누르도록 안내하세요.
- 리서치가 충분하다면 다음 STEP 4 스타일 컨셉 도출 단계로 넘어갈지 묻습니다.
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
- 지금은 STEP 5 디자인 시안 확정 단계입니다.
- STEP 4에서 선택한 스타일 레퍼런스를 기준으로 디자인 시안을 제안하세요.
- STEP 5에서 4개짜리 초기 디자인 시안 세트는 최대 1회만 생성합니다.
- 이미 디자인 시안 이미지가 대화에 있으면 새 4개 세트를 다시 만들지 말고, 사용자가 선택한 1안을 기준으로 형태 / CMF / 기능 디테일을 발전시키세요.
- 후속 이미지가 꼭 필요할 때도 선택된 1안의 개선 렌더 1장만 생성하세요. 비교용 2~4안 재생성은 사용자가 명시적으로 요청한 경우에만 허용합니다.
- 사용자가 시안을 확정하거나 "1번으로 진행", "이 안으로 확정"처럼 최종 선택을 말하면 추가 질문이나 이미지 생성 없이 STEP 6 프로젝트 기획안 생성으로 바로 넘어가세요.
- 디자인 시안 1안과 수정 여부가 확정되기 전에는 프로젝트 기획안 생성 단계로 넘어가지 마세요.
`.trim()
    case 'step_6_rfp':
      return `
[현재 단계 운영]
- 지금은 STEP 6 프로젝트 기획안 생성 단계입니다.
- 정보가 충분하면 반드시 프로젝트 기획안 출력 템플릿대로 문서를 작성하세요.
- 정보가 부족하면 프로젝트 기획안을 쓰지 말고 부족한 항목 1개만 질문하세요.
- 어떤 경우에도 Persona Card 템플릿을 출력하지 마세요. 프로젝트 기획안 문서와 Persona Card는 서로 다른 산출물입니다.
- 문서 작성이 끝나면 다음으로 STEP 7 협력 파트너 매칭 단계로 넘어갈지 물어보세요.
- 시스템은 이 프로젝트 기획안 본문을 그대로 PDF로 저장할 수 있습니다.
- 따라서 "PDF로는 제공할 수 없다", "파일 형태로 직접 생성할 수 없다", "복사해서 사용해달라" 같은 제한 문구를 절대 말하지 마세요.
`.trim()
    case 'step_6_company':
      return `
[현재 단계 운영]
- 지금은 STEP 7 협력 파트너 매칭 단계입니다.
- 프로젝트 기획안 생성 이후의 실행 연결 단계로, Persona Card나 프로젝트 기획안 본문을 새로 생성하지 마세요.
- 먼저 현재 프로젝트에 필요한 협력 유형을 디자인 고도화 / 브랜드·런칭·시장 검증 / 시제품 제작 중 1개로 판단하세요.
- 실제 업체명·전화번호·홈페이지는 검증된 검색 결과 없이는 지어내지 마세요.
- 업체 DB 또는 검색 결과가 없으면, 추천 협력 유형과 업체 선별 기준, 문의 시 전달할 핵심 기획안 요약만 출력하세요.
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
- 지금은 기존 세션의 프로젝트 기획안 생성 단계입니다.
- 스타일 레퍼런스 선택과 디자인 시안 확정이 대화에 없으면 프로젝트 기획안을 생성하지 말고 STEP 4 스타일 컨셉 도출로 되돌리세요.
- 모두 충족되어 있다면 STEP 6 프로젝트 기획안 생성 지침을 따르세요.
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
- 특히 초기 요약, 스타일 방향, 기능 방향, 프로젝트 기획안 작성 시 레퍼런스 가이드라인을 추상 참고가 아니라 구체 기준으로 사용하세요.
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
        /^\s*[1-7][.)]\s*(?:개발 조건|사용자 명확화|개발 방향성|스타일 컨셉|디자인 시안 확정|프로젝트 기획안 생성|협력 파트너)/.test(
          line
        )

      const convertAlphabeticStepLine = (line: string) =>
        line
          .replace(/^\s*(?:A|a)[.)]\s*(개발 조건)/, '1. $1')
          .replace(/^\s*(?:B|b)[.)]\s*(사용자 명확화)/, '2. $1')
          .replace(/^\s*(?:C|c)[.)]\s*(개발 방향성)/, '3. $1')

      const looksLikeAlphabeticStepLine = (line: string) =>
        /^\s*(?:A|a)[.)]\s*개발 조건/.test(line) ||
        /^\s*(?:B|b)[.)]\s*사용자 명확화/.test(line) ||
        /^\s*(?:C|c)[.)]\s*개발 방향성/.test(line)

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
    oneLineDefinition: `${projectName}의 제품화 방향을 정리한 프로젝트 기획안`,
    projectGoal:
      requirementText.length > 5
        ? '대화에서 확정된 요구사항을 기준으로 제품 디자인과 제작 범위를 명확히 정의'
        : '제품 아이디어를 실행 가능한 디자인 및 제작 요청사항으로 구체화',
    finalPurpose: '디자인 고도화, 시제품 제작, 협력 파트너 커뮤니케이션 기준 문서',
    mainTarget: '대화에서 정의한 핵심 사용자 및 사용 맥락',
    usageContext: '사용자가 제품 필요성을 느끼는 주요 일상 상황',
    coreNeeds: '기존 대안으로 충분히 해결되지 않은 사용 불편과 욕구',
    coreValue: '사용자 경험 개선과 제품 차별성 확보',
    styleKeywords: ['정돈된 사용성', '일관된 디자인 언어', '제작 가능성'],
    avoidDirections: ['확정되지 않은 기능 과잉', '제작 난이도를 높이는 불필요한 장식'],
    mustHaveFeatures: ['핵심 사용 상황을 해결하는 기본 기능', '선택된 디자인 방향을 반영한 형태와 재질'],
    niceToHaveFeatures: ['브랜드 확장에 활용 가능한 디테일', '사용 편의성을 높이는 부가 기능'],
    excludedFeatures: ['현재 프로젝트 기획안 범위를 벗어난 고위험 기능'],
    budgetRange: '협력 파트너 견적 산정 필요',
    timeline: '협력 파트너 협의 후 확정',
    sizeOrForm: '선택된 디자인 시안과 사용 환경을 기준으로 상세화',
    implementationNotes: ['초기 시제품 단계에서 구조 안정성과 제작 공정을 우선 검토'],
    referenceSummary: truncateText(referenceSummary, 500),
    researchInsights: ['타겟 사용자의 실제 사용 맥락과 디자인 선호를 기준으로 제품 방향 설정'],
    successCriteria: ['핵심 사용 문제 해결', '선택된 스타일 방향 반영', '시제품 제작 가능성 확보'],
    nextActions: ['프로젝트 기획안 기반 협력 파트너 문의', '견적 및 제작 범위 확인', '시제품 제작 일정 협의'],
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
          '프로젝트 기획안 생성 중 오류가 발생했습니다. 잠시 후 다시 시도해주세요.'
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
