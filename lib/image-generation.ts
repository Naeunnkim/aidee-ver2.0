export type GeneratedImageBlock = {
  images: string[]
  prompt: string
  model: string
  purpose?: 'persona' | 'style_reference' | 'moodboard' | 'design' | 'thumbnail'
}

const DEFAULT_NANO_BANANA_MODEL = 'gemini-2.5-flash-image'

const NANO_BANANA_MODEL_ALIASES: Record<string, string> = {
  'gemini-2.5-flash-image-preview': 'gemini-2.5-flash-image',
}

const NANO_BANANA_MODEL_FALLBACKS = [
  DEFAULT_NANO_BANANA_MODEL,
  'gemini-3.1-flash-image-preview',
  'gemini-2.0-flash-preview-image-generation',
]

function normalizeNanoBananaModel(model: string) {
  return NANO_BANANA_MODEL_ALIASES[model] ?? model
}

function getGeminiApiKey() {
  const apiKey =
    process.env.GEMINI_API_KEY || process.env.GOOGLE_GENERATIVE_AI_API_KEY

  if (!apiKey) {
    throw new Error(
      'GEMINI_API_KEY or GOOGLE_GENERATIVE_AI_API_KEY is missing'
    )
  }

  return apiKey
}

function buildNanobananaPrompt({
  prompt,
  count,
  index,
}: {
  prompt: string
  count: number
  index: number
}) {
  const variationFocus =
    count <= 1
      ? 'Create the single best standalone image for this request.'
      : [
          'Variation focus: mood and emotional tone.',
          'Variation focus: material, texture, and finish.',
          'Variation focus: shape language, proportion, and detail.',
          'Variation focus: balanced final design direction.',
        ][Math.min(index, 3)]

  return [
    prompt,
    '',
    `Generate exactly one standalone image. This is variation ${index + 1} of ${count}.`,
    variationFocus,
    'Do not create a collage, grid, triptych, contact sheet, storyboard, or multi-panel image.',
    'Do not include multiple reference images inside the same output image.',
    'Return one image output, not an explanation-only answer.',
  ].join('\n')
}

async function generateSingleNanobananaImage({
  prompt,
  model,
  apiKey,
}: {
  prompt: string
  model: string
  apiKey: string
}) {
  const response = await fetch(
    `https://generativelanguage.googleapis.com/v1beta/models/${model}:generateContent`,
    {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json',
        'x-goog-api-key': apiKey,
      },
      body: JSON.stringify({
        contents: [
          {
            parts: [{ text: prompt }],
          },
        ],
        generationConfig: {
          responseModalities: ['TEXT', 'IMAGE'],
        },
      }),
    }
  )

  const rawText = await response.text()
  if (!response.ok) {
    console.error('[nanobanana] generation failed', response.status, rawText)
    throw new Error(`Nano Banana image generation failed (${response.status})`)
  }

  let payload: unknown
  try {
    payload = JSON.parse(rawText) as unknown
  } catch {
    throw new Error('Nano Banana returned invalid JSON')
  }

  const candidateParts =
    typeof payload === 'object' &&
    payload &&
    'candidates' in payload &&
    Array.isArray(payload.candidates) &&
    payload.candidates[0] &&
    typeof payload.candidates[0] === 'object' &&
    payload.candidates[0] &&
    'content' in payload.candidates[0] &&
    typeof payload.candidates[0].content === 'object' &&
    payload.candidates[0].content &&
    'parts' in payload.candidates[0].content &&
    Array.isArray(payload.candidates[0].content.parts)
      ? payload.candidates[0].content.parts
      : []

  const images = candidateParts
    .map((part: unknown) => {
      const inlineData =
        typeof part === 'object' && part && 'inlineData' in part
          ? part.inlineData
          : typeof part === 'object' && part && 'inline_data' in part
            ? part.inline_data
            : null

      if (
        typeof inlineData === 'object' &&
        inlineData &&
        'data' in inlineData &&
        typeof inlineData.data === 'string'
      ) {
        const mimeType =
          'mimeType' in inlineData && typeof inlineData.mimeType === 'string'
            ? inlineData.mimeType
            : 'mime_type' in inlineData &&
                typeof inlineData.mime_type === 'string'
              ? inlineData.mime_type
            : 'image/png'

        return `data:${mimeType};base64,${inlineData.data}`
      }

      return null
    })
    .filter((item: string | null): item is string => Boolean(item))

  if (images.length === 0) {
    console.error('[nanobanana] no image parts returned', rawText)
    throw new Error('Nano Banana returned no images')
  }

  return images[0]
}

export async function generateNanoBananaImages({
  prompt,
  count = 1,
  model = DEFAULT_NANO_BANANA_MODEL,
  apiKey,
}: {
  prompt: string
  count?: number
  model?: string
  apiKey?: string
}): Promise<GeneratedImageBlock> {
  const resolvedApiKey = apiKey ?? getGeminiApiKey()
  const preferredModel = normalizeNanoBananaModel(model)
  const modelCandidates = [
    preferredModel,
    ...NANO_BANANA_MODEL_FALLBACKS.filter((candidate) => candidate !== preferredModel),
  ]

  let lastError: unknown = null
  let bestPartialResult: GeneratedImageBlock | null = null

  for (const resolvedModel of modelCandidates) {
    const images: string[] = []

    for (let index = 0; index < count; index += 1) {
      try {
        const image = await generateSingleNanobananaImage({
          prompt: buildNanobananaPrompt({ prompt, count, index }),
          model: resolvedModel,
          apiKey: resolvedApiKey,
        })
        images.push(image)
      } catch (error) {
        lastError = error
        console.error('[nanobanana] single image attempt failed', {
          model: resolvedModel,
          index,
          error,
        })
      }
    }

    if (images.length >= count || (count === 1 && images.length > 0)) {
      return {
        images,
        prompt,
        model: resolvedModel,
      } satisfies GeneratedImageBlock
    }

    if (
      images.length > 0 &&
      (!bestPartialResult || images.length > bestPartialResult.images.length)
    ) {
      bestPartialResult = {
        images,
        prompt,
        model: resolvedModel,
      }
    }
  }

  if (bestPartialResult) {
    return bestPartialResult
  }

  throw new Error(
    lastError instanceof Error
      ? `Nano Banana returned no generated images: ${lastError.message}`
      : 'Nano Banana returned no generated images'
  )
}

export function appendGeneratedImagesBlock({
  text,
  payload,
}: {
  text: string
  payload: GeneratedImageBlock
}) {
  return `${text}

<<AIDEE_IMAGES>>
${JSON.stringify(payload)}
<</AIDEE_IMAGES>>`
}

export function extractGeneratedImagesBlock(text: string) {
  const match = text.match(
    /<<\s*AIDEE[-_ ]?IMAGES\s*>>[\s\n]*([\s\S]*?)[\s\n]*<<\s*\/\s*AIDEE[-_ ]?IMAGES\s*>>/i
  )

  if (!match) {
    const fallbackImages = text
      .match(/data:image\/[a-zA-Z0-9.+-]+;base64,[A-Za-z0-9+/=]+/g)
      ?.filter((value) => typeof value === 'string') ?? []

    if (fallbackImages.length > 0) {
      return {
        cleanedText: text.trim(),
        imageBlock: {
          images: fallbackImages,
          prompt: '',
          model: 'fallback-inline-data',
        } satisfies GeneratedImageBlock,
      }
    }

    return {
      cleanedText: text.trim(),
      imageBlock: null as GeneratedImageBlock | null,
    }
  }

  const cleanedText = text.replace(
    /\n?<<\s*AIDEE[-_ ]?IMAGES\s*>>[\s\S]*?<<\s*\/\s*AIDEE[-_ ]?IMAGES\s*>>\s*$/i,
    ''
  )

  try {
    const parsed = JSON.parse(match[1]) as GeneratedImageBlock
    const hasImages =
      Array.isArray(parsed.images) &&
      parsed.images.every((item) => typeof item === 'string')

    if (!hasImages) {
      return {
        cleanedText: cleanedText.trim(),
        imageBlock: null as GeneratedImageBlock | null,
      }
    }

    return {
      cleanedText: cleanedText.trim(),
      imageBlock: parsed,
    }
  } catch {
    return {
      cleanedText: cleanedText.trim(),
      imageBlock: null as GeneratedImageBlock | null,
    }
  }
}
