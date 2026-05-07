import { google } from '@ai-sdk/google'
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
import { type StageKey, getNextStageKey, isKnownStageKey } from '@/lib/study'

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

const DEFAULT_STAGE_KEY: StageKey = 'step_1_idea'

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

async function generateStyleReferenceImages({
  project,
  referenceImages,
  conversation,
}: {
  project: ProjectRecord | null
  referenceImages: ReferenceImageRecord[]
  conversation: string
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
    {
      label: 'compact-single',
      prompt: buildCompactStyleReferencePrompt({
        project,
        referenceImages,
      }),
      count: 1,
    },
  ]

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
      })

      console.log('[style-images] attempt success', {
        label: attempt.label,
        imageCount: payload.images.length,
        model: payload.model,
      })

      if (payload.images.length > 0) {
        return payload
      }
    } catch (error) {
      console.error('[style-images] attempt failed', {
        label: attempt.label,
        error,
      })
    }
  }

  return null
}

function hasStyleReferenceSelection(text: string) {
  return /([1-3])\s*번|이미지\s*([1-3])|레퍼런스\s*([1-3])|선택|확정/.test(text)
}

function buildInitialPrompt(project: ProjectRecord | null) {
  const title = project?.title || '새 프로젝트'

  return [
    `${title} 프로젝트가 방금 생성되었습니다.`,
    '저장된 프로젝트 정보(requirements)와 레퍼런스 이미지 분석 결과를 바탕으로 현재 프로젝트를 짧게 요약하고, 사용자가 다음에 무엇을 말하면 좋을지 자연스럽게 안내해주세요.',
    '답변은 한국어로 작성하세요.',
    '구조는 다음 순서를 지키세요:',
    '1. 프로젝트 요약 2~4문장',
    '2. 레퍼런스 이미지에서 우선 참고해야 할 방향 2~3개 bullet',
    '3. 마지막에 사용자가 바로 답할 수 있는 질문 1개',
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

function isRfpDocumentRequest(text: string) {
  return /rfp|제안요청서|제안\s*요청서|문서\s*생성|pdf/i.test(text)
}

function isCompanyConnectionRequest(text: string) {
  return /협력\s*업체|업체\s*연결|업체\s*추천|파트너|vendor|company/i.test(text)
}

function resolveIntentStageKey({
  currentStageKey,
  lastUserMessage,
}: {
  currentStageKey: StageKey
  lastUserMessage: string
}): StageKey {
  if (isCompanyConnectionRequest(lastUserMessage)) {
    return 'step_6_company'
  }

  if (isRfpDocumentRequest(lastUserMessage)) {
    return 'step_6_rfp'
  }

  return currentStageKey
}

function getStageSpecificInstruction(currentStageKey: StageKey) {
  switch (currentStageKey) {
    case 'step_1_idea':
      return `
[현재 단계 운영]
- 지금은 STEP 1입니다.
- 프로젝트 생성 직후에는 먼저 저장된 정보를 기준점으로 정리하고, 부족한 정보를 묻는 질문 1개로 끝내세요.
- STEP 1 확정 조건이 충족되면 STEP 2로 넘어가기 위해 사용자 명확화 질문을 이어가세요.
- STEP 1에서는 Persona Card를 절대 출력하지 마세요. Persona Card는 STEP 2 전용 산출물입니다.
`.trim()
    case 'step_2_persona':
      return `
[현재 단계 운영]
- 지금은 STEP 2의 페르소나 정리 단계입니다.
- 사용자 답변을 바탕으로 페르소나를 구체화하세요.
- 조건이 충족되면 반드시 Persona Card 템플릿을 출력하고, 마지막에 "이 페르소나로 리서치를 진행할까요, 아니면 페르소나를 수정할까요?" 질문을 넣으세요.
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
- 지금은 STEP 3입니다.
- 감성 / 기능 / 심미 중 1순위, 핵심 가치 키워드, 덜 중요하게 가져갈 요소를 정리하세요.
- 확정 조건이 충족되면 STEP 4 스타일 컨셉 도출로 넘어가세요.
`.trim()
    case 'step_4_style':
      return `
[현재 단계 운영]
- 지금은 STEP 4 스타일 컨셉 도출 단계입니다.
- 사용자가 선택할 수 있도록 Nano Banana 스타일 레퍼런스 이미지 3장을 생성해야 합니다.
- 이미지 3장을 제시한 뒤에는 반드시 마음에 드는 이미지 1개 선택을 요청하고, 선택 전에는 STEP 5로 넘어가지 마세요.
- 사용자가 이미지를 선택하면 선택된 레퍼런스를 기준으로 형태 / 색감 / 재질 중 최소 2개 방향성을 확정하고 STEP 5 디자인 제안으로 넘어가세요.
`.trim()
    case 'step_5_design':
      return `
[현재 단계 운영]
- 지금은 STEP 5 디자인 제안 단계입니다.
- STEP 4에서 선택한 스타일 레퍼런스를 기준으로 디자인 시안을 제안하세요.
- 필요하면 generate_design_image 도구로 제품 렌더 또는 3D 시안 1~2장을 생성하세요.
- 디자인 시안 1안과 수정 여부가 확정되기 전에는 RFP로 넘어가지 마세요.
- 시안이 확정되면 STEP 6 RFP 문서 생성으로 넘어가세요.
`.trim()
    case 'step_6_rfp':
      return `
[현재 단계 운영]
- 지금은 STEP 6 평가 및 RFP 문서 생성 단계입니다.
- 정보가 충분하면 반드시 RFP 출력 템플릿대로 문서를 작성하세요.
- 정보가 부족하면 RFP를 쓰지 말고 부족한 항목 1개만 질문하세요.
- 어떤 경우에도 Persona Card 템플릿을 출력하지 마세요. RFP 문서와 Persona Card는 서로 다른 산출물입니다.
- RFP 본문 작성이 끝나면 마지막에 "RFP 문서 다운로드 후 협력업체 연결로 이어갈 수 있습니다."를 1줄로 안내하세요.
- 시스템은 이 RFP 본문을 그대로 PDF로 저장할 수 있습니다.
- 따라서 "PDF로는 제공할 수 없다", "파일 형태로 직접 생성할 수 없다", "복사해서 사용해달라" 같은 제한 문구를 절대 말하지 마세요.
`.trim()
    case 'step_6_company':
      return `
[현재 단계 운영]
- 지금은 STEP 6-2 협력업체 연결 단계입니다.
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
    /<<AIDEE_STAGE>>[\s\S]*?<<\/AIDEE_STAGE>>/i,
  ]

  const lines = text
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
        return convertIndexedChoice(line)
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

function inferStageTransitionFromText({
  currentStageKey,
  text,
}: {
  currentStageKey: StageKey
  text: string
}): StageMeta | null {
  const transitionHints = [
    /다음 단계 진행은 UI 버튼/i,
    /다음 단계로 넘어가기/i,
    /다음 단계로 진행/i,
    /다음 단계로 이어/i,
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
    !['step_2_research', 'step_3_direction', 'step_5_design'].includes(
      currentStageKey
    )
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

export async function POST(req: Request) {
  try {
    const body = (await req.json()) as ChatRequestBody

    if (!Array.isArray(body?.messages)) {
      return new Response(
        JSON.stringify({ error: 'messages must be an array' }),
        {
          status: 400,
          headers: { 'Content-Type': 'application/json' },
        }
      )
    }

    if (!process.env.GOOGLE_GENERATIVE_AI_API_KEY) {
      return new Response(JSON.stringify({ error: 'Gemini API key missing' }), {
        status: 500,
        headers: { 'Content-Type': 'application/json' },
      })
    }

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

    if (currentStageKey === 'step_6_rfp') {
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

        const markdown = formatRfpMarkdown(rfpObjectResult.object)
        const finalText = `${markdown}

<<AIDEE_RFP_JSON>>
${JSON.stringify(rfpObjectResult.object, null, 2)}
<</AIDEE_RFP_JSON>>`

        return new Response(finalText, {
          headers: {
            'Content-Type': 'text/plain; charset=utf-8',
            'x-aidee-current-stage': 'step_6_rfp',
            'x-aidee-next-stage': 'step_6_company',
            'x-aidee-transition': 'yes',
            'x-aidee-reason': 'rfp_completed',
          },
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
      })
    }
    const shouldBypassModelTextForStyleImages = shouldGenerateStyleReferenceImages

    if (generatedImagePayload && shouldGenerateStyleReferenceImages) {
      const styleText =
        '스타일 레퍼런스 3장을 생성했습니다. 마음에 드는 방향을 하나 선택해주세요.'
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
                  })

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

    let finalText = normalizeChoiceFormatting(sanitizeAssistantText(cleanedText))

    if (shouldBypassModelTextForStyleImages) {
      finalText =
        '스타일 레퍼런스 3장을 생성했습니다. 마음에 드는 방향을 하나 선택해주세요.'
    }

    if (
      !generatedImagePayload &&
      canGenerateImages &&
      isImageGenerationRequest(lastUserMessage)
    ) {
      try {
        generatedImagePayload = await generateNanoBananaImages({
          prompt: buildFallbackImagePrompt({
            project,
            referenceImages,
            userRequest: lastUserMessage,
          }),
          count: extractRequestedImageCount(lastUserMessage),
        })

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

    if (
      !generatedImagePayload &&
      ((canGenerateImages &&
        (currentStageKey === 'step_4_style' ||
          stageMeta.nextStageKey === 'step_4_style')) ||
        (currentStageKey === 'step_5_design' &&
          stageMeta.nextStageKey === 'step_5_design')) &&
      !hasStyleReferenceSelection(lastUserMessage)
    ) {
      generatedImagePayload = await generateStyleReferenceImages({
        project,
        referenceImages,
        conversation: buildConversationText(messages),
      })

      if (!finalText.trim()) {
        finalText =
          generatedImagePayload?.images.length === 1
            ? '스타일 레퍼런스 1장을 생성했습니다. 마음에 드는 방향을 선택해주세요.'
            : '스타일 레퍼런스 3장을 생성했습니다. 마음에 드는 방향을 하나 선택해주세요.'
      } else if (!/선택|이미지|레퍼런스/.test(finalText)) {
        finalText = `${finalText}\n\n${
          generatedImagePayload?.images.length === 1
            ? '스타일 레퍼런스 1장을 생성했습니다. 마음에 드는 방향을 선택해주세요.'
            : '스타일 레퍼런스 3장을 생성했습니다. 마음에 드는 방향을 하나 선택해주세요.'
        }`
      }
    }

    if (generatedImagePayload) {
      finalText = appendGeneratedImagesBlock({
        text: finalText,
        payload: generatedImagePayload,
      })
    }

    const shouldGenerateRfpJson =
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

    if (inferredStageMeta) {
      stageMeta = inferredStageMeta
    }

    return new Response(finalText, {
      headers: {
        'Content-Type': 'text/plain; charset=utf-8',
        'x-aidee-current-stage': stageMeta.currentStageKey,
        'x-aidee-next-stage': stageMeta.nextStageKey,
        'x-aidee-transition': stageMeta.transition ? 'yes' : 'no',
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
