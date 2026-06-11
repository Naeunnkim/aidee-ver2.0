import { createElement, type ReactElement } from 'react'

import { generateObject, type ModelMessage } from 'ai'
import { google } from '@ai-sdk/google'
import { renderToBuffer, type DocumentProps } from '@react-pdf/renderer'

import {
  OnePageRfpPdfDocument,
  RawRfpPdfDocument,
  RfpPdfDocument,
} from '@/lib/rfp-pdf'
import {
  RFP_DOCUMENT_SCHEMA,
  type RfpDocument,
  buildRfpObjectPrompt,
} from '@/lib/rfp'
import { extractGeneratedImagesBlock } from '@/lib/image-generation'
import { createClient } from '@/lib/supabase/server'

export const runtime = 'nodejs'
export const maxDuration = 60

type ProjectRecord = {
  title: string | null
  requirements: Record<string, unknown> | null
}

type ReferenceImageRecord = {
  file_name: string | null
  analysis_text: string | null
  analysis_json: unknown
}

type MessageRecord = {
  role: string
  content: string
}

type RequestBody = {
  projectId?: string
  rfpJson?: RfpDocument | null
  rfpContent?: string
  projectTitle?: string
  layout?: 'one-page' | 'full'
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
            .map((part) =>
              'text' in part && typeof part.text === 'string' ? part.text : ''
            )
            .filter(Boolean)
            .join('\n')
        : ''

      return `[${message.role}] ${text}`
    })
    .join('\n\n')
}

function sanitizeFilename(value: string) {
  return value.replace(/[\\/:*?"<>|]+/g, '-').trim() || 'aidee-rfp'
}

function getAsciiDownloadFilename(baseName: string) {
  const normalized = baseName.replace(/[^a-zA-Z0-9._-]+/g, '-').replace(/-+/g, '-')
  return (normalized.trim().replace(/^-+|-+$/g, '') || 'aidee-rfp') + '.pdf'
}

function getSelectedReferenceImage(messages: MessageRecord[]) {
  const latestSelectionIndex = [...messages]
    .reverse()
    .map((message) => {
      if (message.role !== 'user') {
        return null
      }

      const match = message.content.match(/스타일\s*레퍼런스\s*([1-4])번/)
      return match ? Number(match[1]) - 1 : null
    })
    .find((index): index is number => typeof index === 'number')

  const latestImageBlock = [...messages]
    .reverse()
    .map((message) => extractGeneratedImagesBlock(message.content).imageBlock)
    .find((imageBlock) => imageBlock?.images.length)

  if (!latestImageBlock?.images.length) {
    return null
  }

  if (
    typeof latestSelectionIndex === 'number' &&
    latestImageBlock.images[latestSelectionIndex]
  ) {
    return latestImageBlock.images[latestSelectionIndex]
  }

  return latestImageBlock.images[0]
}

async function generateRfpJson({
  project,
  referenceImages,
  messages,
}: {
  project: ProjectRecord | null
  referenceImages: ReferenceImageRecord[]
  messages: MessageRecord[]
}) {
  const requirementsText = buildRequirementsText(project)
  const referenceContext = buildReferenceContext(referenceImages)
  const conversation = buildConversationText(
    messages.map(
      (message) =>
        ({
          role:
            message.role === 'assistant' ||
            message.role === 'system' ||
            message.role === 'user'
              ? message.role
              : 'user',
          content: message.content,
        }) satisfies ModelMessage
    )
  )

  const result = await generateObject({
    model: google('gemini-2.5-flash'),
    schema: RFP_DOCUMENT_SCHEMA,
    prompt: buildRfpObjectPrompt({
      projectTitle: project?.title || '제목 없음',
      requirements: requirementsText,
      referenceContext,
      conversation,
    }),
  })

  return result.object
}

export async function POST(request: Request) {
  try {
    const body = (await request.json().catch(() => null)) as RequestBody | null
    const projectId = body?.projectId
    console.log('[rfp/download] request start', {
      hasProjectId: Boolean(projectId),
      hasInlineRfpJson: Boolean(body?.rfpJson),
      hasInlineRfpContent: Boolean(body?.rfpContent),
    })

    if (!projectId) {
      return new Response(JSON.stringify({ error: 'projectId is required' }), {
        status: 400,
        headers: { 'Content-Type': 'application/json' },
      })
    }

    if (!process.env.GOOGLE_GENERATIVE_AI_API_KEY) {
      return new Response(JSON.stringify({ error: 'Gemini API key missing' }), {
        status: 500,
        headers: { 'Content-Type': 'application/json' },
      })
    }

    const supabase = await createClient()
    const {
      data: { user },
    } = await supabase.auth.getUser()

    if (!user) {
      return new Response(JSON.stringify({ error: 'Unauthorized' }), {
        status: 401,
        headers: { 'Content-Type': 'application/json' },
      })
    }
    console.log('[rfp/download] user ok', { userId: user.id })

    const { data: projectData } = await supabase
      .from('projects')
      .select('title, requirements')
      .eq('id', projectId)
      .single()

    const { data: referenceData } = await supabase
      .from('project_reference_images')
      .select('file_name, analysis_text, analysis_json')
      .eq('project_id', projectId)
      .eq('analysis_status', 'completed')

    const { data: messageData } = await supabase
      .from('messages')
      .select('role, content')
      .eq('project_id', projectId)
      .order('seq_order', { ascending: true })

    const project = (projectData as ProjectRecord | null) ?? null
    const referenceImages = (referenceData as ReferenceImageRecord[] | null) ?? []
    const messages = (messageData as MessageRecord[] | null) ?? []
    console.log('[rfp/download] data loaded', {
      hasProject: Boolean(project),
      referenceCount: referenceImages.length,
      messageCount: messages.length,
    })

    const shouldUseOnePageLayout = body?.layout !== 'full'
    const hasInlineRfpContent =
      typeof body?.rfpContent === 'string' && body.rfpContent.trim().length > 0

    let pdfDocument: ReactElement<DocumentProps>
    let filenameBase: string

    if (hasInlineRfpContent && !shouldUseOnePageLayout) {
      const safeTitle =
        (typeof body?.projectTitle === 'string' && body.projectTitle.trim()) ||
        project?.title ||
        'aidee-rfp'

      console.log('[rfp/download] using inline rfp content', {
        title: safeTitle,
      })

      pdfDocument = createElement(RawRfpPdfDocument, {
        title: safeTitle,
        content: body!.rfpContent!,
      }) as ReactElement<DocumentProps>

      filenameBase = sanitizeFilename(safeTitle)
    } else {
      const rfpJson = body?.rfpJson
        ? RFP_DOCUMENT_SCHEMA.parse(body.rfpJson)
        : await generateRfpJson({
            project,
            referenceImages,
            messages,
          })
      console.log('[rfp/download] rfp json ready', {
        projectName: rfpJson.projectName,
      })

      if (shouldUseOnePageLayout) {
        pdfDocument = createElement(OnePageRfpPdfDocument, {
          rfp: rfpJson,
          selectedReferenceImage: getSelectedReferenceImage(messages),
        }) as ReactElement<DocumentProps>
      } else {
        pdfDocument = createElement(RfpPdfDocument, {
          rfp: rfpJson,
        }) as ReactElement<DocumentProps>
      }

      filenameBase = sanitizeFilename(rfpJson.projectName)
    }

    const pdfBuffer = await renderToBuffer(pdfDocument)
    console.log('[rfp/download] pdf buffer ready', {
      bytes: pdfBuffer.byteLength,
    })
    const encodedFilename = encodeURIComponent(`${filenameBase}.pdf`)
    const asciiFilename = getAsciiDownloadFilename(filenameBase)

    return new Response(new Uint8Array(pdfBuffer), {
      status: 200,
      headers: {
        'Content-Type': 'application/pdf',
        'Content-Disposition': `attachment; filename="${asciiFilename}"; filename*=UTF-8''${encodedFilename}`,
      },
    })
  } catch (error) {
    const message =
      error instanceof Error ? error.message : 'Failed to generate project plan PDF'

    console.error('Project plan PDF error:', error)

    return new Response(JSON.stringify({ error: message }), {
      status: 500,
      headers: { 'Content-Type': 'application/json' },
    })
  }
}
