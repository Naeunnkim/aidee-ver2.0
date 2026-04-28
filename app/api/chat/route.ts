import { google } from '@ai-sdk/google'
import { generateObject, generateText, tool, type ModelMessage } from 'ai'
import { createClient } from '@supabase/supabase-js'
import { z } from 'zod'

import {
  appendGeneratedImagesBlock,
  type GeneratedImageBlock,
  generateNanoBananaImages,
} from '@/lib/image-generation'
import { SYSTEM_PROMPT_TEMPLATE } from '@/lib/prompts'
import { RFP_DOCUMENT_SCHEMA, buildRfpObjectPrompt } from '@/lib/rfp'
import { type StageKey, isKnownStageKey } from '@/lib/study'

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

function isGeneratedImageBlock(value: unknown): value is GeneratedImageBlock {
  return (
    typeof value === 'object' &&
    value !== null &&
    'images' in value &&
    Array.isArray(value.images) &&
    value.images.every((image) => typeof image === 'string') &&
    'prompt' in value &&
    typeof value.prompt === 'string' &&
    'model' in value &&
    typeof value.model === 'string'
  )
}

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
  const guidelineBlock = buildReferenceGuidelineBlock(referenceImages)

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

function getStageSpecificInstruction(currentStageKey: StageKey) {
  switch (currentStageKey) {
    case 'step_1_idea':
      return `
[현재 단계 운영]
- 지금은 STEP 1입니다.
- 프로젝트 생성 직후에는 먼저 저장된 정보를 기준점으로 정리하고, 부족한 정보를 묻는 질문 1개로 끝내세요.
- STEP 1 확정 조건이 충족되면 STEP 2로 넘어가기 위해 사용자 명확화 질문을 이어가세요.
- STEP 2로 넘어갈 준비가 되었다면 Persona Card를 출력하고, 마지막에 반드시 "리서치 진행 / 페르소나 수정" 선택을 물으세요.
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
- 스타일 키워드, 핵심 기능, 제외할 기능, 현실 범위를 정리하는 데 집중하세요.
- 확정 조건이 충족되면 짧게 요약하고 STEP 4로 넘어갈 수 있는 질문 1개만 하세요.
`.trim()
    case 'step_4_definition':
      return `
[현재 단계 운영]
- 지금은 STEP 4입니다.
- 제품 한 줄 정의, 사용자 가치 제안, 포함/제외 범위를 명확히 고정하세요.
- 확정 조건이 충족되면 RFP 생성을 위한 마지막 확인 후 STEP 5로 넘어가세요.
`.trim()
    case 'step_5_rfp':
      return `
[현재 단계 운영]
- 지금은 STEP 5입니다.
- 정보가 충분하면 반드시 RFP 출력 템플릿대로 문서를 작성하세요.
- 정보가 부족하면 RFP를 쓰지 말고 부족한 항목 1개만 질문하세요.
- 시스템은 이 RFP 본문을 그대로 PDF로 저장할 수 있습니다.
- 따라서 "PDF로는 제공할 수 없다", "파일 형태로 직접 생성할 수 없다", "복사해서 사용해달라" 같은 제한 문구를 절대 말하지 마세요.
- 사용자가 문서 형태를 요청하면, 실제로 다운로드 가능한 문서의 본문이라고 가정하고 완성된 RFP만 제시하세요.
`.trim()
    default:
      return ''
  }
}

function buildSystemPrompt({
  project,
  referenceImages,
  currentStageKey,
}: {
  project: ProjectRecord | null
  referenceImages: ReferenceImageRecord[]
  currentStageKey: StageKey
}) {
  const projectContext = buildProjectContext(project, referenceImages)
  const stageInstruction = getStageSpecificInstruction(currentStageKey)

  return `
${SYSTEM_PROMPT_TEMPLATE}

[실행 컨텍스트]
- 현재 단계 key: ${currentStageKey}
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

${projectContext}

${stageInstruction}

[단계 메타 출력 규칙]
- 사용자에게 보여줄 실제 답변을 모두 작성한 뒤, 마지막 줄 아래에 반드시 아래 형식의 메타 블록을 추가하세요.
- 메타 블록은 사용자에게 보여주기 위한 내용이 아니며, 형식을 절대 바꾸지 마세요.
- current_stage와 next_stage는 다음 중 하나만 사용하세요:
  step_1_idea, step_2_persona, step_2_research, step_3_direction, step_4_definition, step_5_rfp
- transition은 yes 또는 no만 사용하세요.
- reason은 짧은 영어 snake_case로 작성하세요.

<<AIDEE_STAGE>>
current_stage=${currentStageKey}
next_stage=${currentStageKey}
transition=no
reason=stay
<</AIDEE_STAGE>>
`.trim()
}

function sanitizeAssistantText(text: string) {
  const blockedPatterns = [
    /다시 한 번 말씀드리지만.*pdf.*직접 생성.*제공.*수 없습니다\.?/i,
    /pdf와 같은 파일 형태로는 직접 생성하여 제공해 드릴 수 없습니다\.?/i,
    /제공된 텍스트를 복사하여 사용하시면 됩니다\.?/i,
    /파일 형태로 제공할 수 없습니다\.?/i,
    /pdf로는 제공할 수 없습니다\.?/i,
  ]

  const lines = text
    .split('\n')
    .filter((line) => !blockedPatterns.some((pattern) => pattern.test(line)))

  return lines.join('\n').trim()
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
    const currentStageKey = body.currentStageKey ?? DEFAULT_STAGE_KEY

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
    })

    const messages: ModelMessage[] = isInitialEntry
      ? [
          {
            role: 'user',
            content: buildInitialPrompt(project),
          },
        ]
      : normalizedMessages

    let generatedImagePayload: GeneratedImageBlock | null = null
    const lastUserMessage =
      [...normalizedMessages]
        .reverse()
        .find((message) => message.role === 'user')?.content ?? ''

    const result = await generateText({
      model: google('gemini-2.5-flash'),
      system,
      messages,
      stopWhen: ({ steps }) =>
        steps.length >= 1 &&
        steps[steps.length - 1].toolCalls.length === 0,
      tools: {
        generate_design_image: tool({
          description:
            'Generate product concept or final design images with Gemini Nano Banana and return data URL images.',
          inputSchema: z.object({
            prompt: z
              .string()
              .min(1)
              .describe('Detailed image generation prompt for the desired visual'),
            count: z
              .number()
              .int()
              .min(1)
              .max(4)
              .default(1)
              .describe('Number of image variations to create'),
            model: z
              .enum(['gemini-2.5-flash-image', 'gemini-3.1-flash-image-preview'])
              .default('gemini-3.1-flash-image-preview')
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

    const { cleanedText, stageMeta } = parseStageMeta(
      result.text,
      currentStageKey
    )

    let finalText = sanitizeAssistantText(cleanedText)

    if (!generatedImagePayload && isImageGenerationRequest(lastUserMessage)) {
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

    if (generatedImagePayload) {
      finalText = appendGeneratedImagesBlock({
        text: finalText,
        payload: generatedImagePayload,
      })
    }

    const shouldGenerateRfpJson =
      stageMeta.nextStageKey === 'step_5_rfp' &&
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
