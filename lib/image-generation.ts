export type GeneratedImageBlock = {
  images: string[]
  prompt: string
  model: string
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
}: {
  prompt: string
  model: string
}) {
  const apiKey = getGeminiApiKey()
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
      if (
        typeof part === 'object' &&
        part &&
        'inlineData' in part &&
        typeof part.inlineData === 'object' &&
        part.inlineData &&
        'data' in part.inlineData &&
        typeof part.inlineData.data === 'string'
      ) {
        const mimeType =
          'mimeType' in part.inlineData && typeof part.inlineData.mimeType === 'string'
            ? part.inlineData.mimeType
            : 'image/png'

        return `data:${mimeType};base64,${part.inlineData.data}`
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
  model = 'gemini-3.1-flash-image-preview',
}: {
  prompt: string
  count?: number
  model?: string
}) {
  const images: string[] = []

  for (let index = 0; index < count; index += 1) {
    const image = await generateSingleNanobananaImage({
      prompt: buildNanobananaPrompt({ prompt, count, index }),
      model,
    })
    images.push(image)
  }

  return {
    images,
    prompt,
    model,
  } satisfies GeneratedImageBlock
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
    /<<AIDEE_IMAGES>>[\s\n]*([\s\S]*?)[\s\n]*<<\/AIDEE_IMAGES>>/
  )

  if (!match) {
    return {
      cleanedText: text.trim(),
      imageBlock: null as GeneratedImageBlock | null,
    }
  }

  const cleanedText = text.replace(
    /\n?<<AIDEE_IMAGES>>[\s\S]*?<<\/AIDEE_IMAGES>>\s*$/,
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
