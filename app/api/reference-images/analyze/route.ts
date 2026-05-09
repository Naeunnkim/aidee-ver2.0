import { createGoogleGenerativeAI } from '@ai-sdk/google'
import { generateText } from 'ai'
import { createClient } from '@supabase/supabase-js'
import { NextResponse } from 'next/server'

const ANALYSIS_PROMPT = `
이 이미지는 사용자가 업로드한 프로젝트 레퍼런스 이미지입니다.
이 이미지를 제품/공간/스타일 레퍼런스로 보고 한국어로 분석해주세요.

반드시 아래 JSON 형식으로만 응답하세요. 코드블록 마크다운은 사용하지 마세요.
{
  "summary": "이미지 한 줄 요약",
  "category": "이미지 유형 또는 오브제 유형",
  "moodKeywords": ["키워드1", "키워드2", "키워드3"],
  "colorKeywords": ["색상1", "색상2"],
  "materialKeywords": ["재질1", "재질2"],
  "shapeKeywords": ["형태1", "형태2"],
  "detailPoints": ["디테일 포인트 1", "디테일 포인트 2"],
  "designDirection": ["디자인 방향 1", "디자인 방향 2"]
}

분석 기준:
1. 이미지의 핵심 피사체/유형
2. 전체 분위기와 무드
3. 주요 색상
4. 추정 재질
5. 형태적 특징
6. 참고할 만한 디테일
7. 향후 컨셉 생성이나 이미지 생성 시 참고할 수 있는 방향
`.trim()

type StoredImageMetadata = {
  file_path?: string
  mime_type?: string
  file_size?: number
  analysis?: Record<string, unknown>
}

type ReferenceImageRow = {
  id: string
  project_id: string
  file_name: string | null
  image_url: string
  analysis_status: 'pending' | 'completed' | 'failed' | string
  analysis_text: string | null
  analysis_json: StoredImageMetadata | null
}

function getSupabaseAdmin() {
  const supabaseUrl = process.env.NEXT_PUBLIC_SUPABASE_URL
  const supabaseServiceRoleKey = process.env.SUPABASE_SERVICE_ROLE_KEY
  const supabaseAnonKey = process.env.NEXT_PUBLIC_SUPABASE_ANON_KEY

  if (!supabaseUrl) {
    throw new Error('NEXT_PUBLIC_SUPABASE_URL is missing')
  }

  const key = supabaseServiceRoleKey || supabaseAnonKey
  if (!key) {
    throw new Error(
      'SUPABASE_SERVICE_ROLE_KEY or NEXT_PUBLIC_SUPABASE_ANON_KEY is missing'
    )
  }

  return createClient(supabaseUrl, key, {
    auth: {
      persistSession: false,
      autoRefreshToken: false,
    },
  })
}

function getGoogleModel() {
  const apiKey =
    process.env.GOOGLE_GENERATIVE_AI_API_KEY ?? process.env.GEMINI_API_KEY

  if (!apiKey) {
    throw new Error('GOOGLE_GENERATIVE_AI_API_KEY or GEMINI_API_KEY is missing')
  }

  return createGoogleGenerativeAI({ apiKey })('gemini-2.5-flash')
}

function stripCodeFence(text: string) {
  const trimmed = text.trim()
  if (!trimmed.startsWith('```')) {
    return trimmed
  }

  return trimmed
    .replace(/^```json\s*/i, '')
    .replace(/^```\s*/i, '')
    .replace(/```$/i, '')
    .trim()
}

function safeParseAnalysis(text: string) {
  try {
    return JSON.parse(stripCodeFence(text)) as Record<string, unknown>
  } catch {
    return null
  }
}

async function analyzeSingleImage(image: ReferenceImageRow) {
  const imageResponse = await fetch(image.image_url)
  if (!imageResponse.ok) {
    throw new Error(`Failed to fetch image: ${image.file_name ?? image.id}`)
  }

  const imageBuffer = await imageResponse.arrayBuffer()
  const result = await generateText({
    model: getGoogleModel(),
    messages: [
      {
        role: 'user',
        content: [
          { type: 'text', text: ANALYSIS_PROMPT },
          {
            type: 'image',
            image: imageBuffer,
          },
        ],
      },
    ],
  })

  const analysisJson = safeParseAnalysis(result.text)

  return {
    analysisText: result.text,
    analysisJson,
  }
}

export async function POST(req: Request) {
  try {
    const body = (await req.json()) as { projectId?: string }
    const projectId = typeof body?.projectId === 'string' ? body.projectId : ''

    if (!projectId) {
      return NextResponse.json({ error: 'projectId is required' }, { status: 400 })
    }

    const supabase = getSupabaseAdmin()

    const { data: images, error: fetchError } = await supabase
      .from('project_reference_images')
      .select(
        'id, project_id, file_name, image_url, analysis_status, analysis_text, analysis_json'
      )
      .eq('project_id', projectId)
      .eq('analysis_status', 'pending')

    if (fetchError) {
      throw fetchError
    }

    const pendingImages = (images ?? []) as ReferenceImageRow[]

    if (pendingImages.length === 0) {
      return NextResponse.json({
        success: true,
        projectId,
        analyzedCount: 0,
        message: 'No pending reference images found',
      })
    }

    const results: Array<{
      id: string
      status: 'completed' | 'failed'
      fileName: string | null
      error?: string
    }> = []

    for (const image of pendingImages) {
      try {
        const analysis = await analyzeSingleImage(image)

        const mergedAnalysisJson = {
          ...(image.analysis_json ?? {}),
          analysis: analysis.analysisJson,
        }

        const { error: updateError } = await supabase
          .from('project_reference_images')
          .update({
            analysis_status: 'completed',
            analysis_text: analysis.analysisText,
            analysis_json: mergedAnalysisJson,
          })
          .eq('id', image.id)

        if (updateError) {
          throw updateError
        }

        results.push({
          id: image.id,
          status: 'completed',
          fileName: image.file_name,
        })
      } catch (error: unknown) {
        const errorMessage =
          error instanceof Error ? error.message : 'Unknown analysis error'

        await supabase
          .from('project_reference_images')
          .update({
            analysis_status: 'failed',
            analysis_text: errorMessage,
          })
          .eq('id', image.id)

        results.push({
          id: image.id,
          status: 'failed',
          fileName: image.file_name,
          error: errorMessage,
        })
      }
    }

    return NextResponse.json({
      success: true,
      projectId,
      analyzedCount: results.filter((item) => item.status === 'completed').length,
      failedCount: results.filter((item) => item.status === 'failed').length,
      results,
    })
  } catch (error: unknown) {
    console.error('Reference image analysis error:', error)
    return NextResponse.json(
      {
        success: false,
        error:
          error instanceof Error
            ? error.message
            : 'Failed to analyze reference images',
      },
      { status: 500 }
    )
  }
}
