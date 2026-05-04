import type { SupabaseClient } from '@supabase/supabase-js'

import type { GeneratedImageBlock } from '@/lib/image-generation'

type ProjectRequirements = Record<string, unknown> & {
  thumbnail_image_url?: string
  thumbnail_image_source?: string
  thumbnail_image_prompt?: string | null
}

function getImageExtension(mimeType: string) {
  if (mimeType.includes('webp')) {
    return 'webp'
  }

  if (mimeType.includes('jpeg') || mimeType.includes('jpg')) {
    return 'jpg'
  }

  return 'png'
}

function normalizeRequirements(value: unknown): ProjectRequirements {
  if (value && typeof value === 'object' && !Array.isArray(value)) {
    return value as ProjectRequirements
  }

  return {}
}

export async function saveGeneratedProjectThumbnail({
  supabase,
  projectId,
  imageBlock,
  overwrite = true,
}: {
  supabase: SupabaseClient
  projectId: string
  imageBlock: GeneratedImageBlock
  overwrite?: boolean
}) {
  const firstImage = imageBlock.images[0]

  if (!firstImage) {
    return null
  }

  const { data: project, error: fetchError } = await supabase
    .from('projects')
    .select('requirements')
    .eq('id', projectId)
    .single()

  if (fetchError) {
    throw fetchError
  }

  const requirements = normalizeRequirements(project?.requirements)
  const existingThumbnail = requirements.thumbnail_image_url

  if (!overwrite && typeof existingThumbnail === 'string' && existingThumbnail) {
    return existingThumbnail
  }

  const imageResponse = await fetch(firstImage)

  if (!imageResponse.ok) {
    throw new Error('Generated image data could not be read')
  }

  const imageBlob = await imageResponse.blob()
  const mimeType = imageBlob.type || 'image/png'
  const extension = getImageExtension(mimeType)
  const filePath = `projects/${projectId}/generated/thumbnails/${crypto.randomUUID()}.${extension}`

  const { error: uploadError } = await supabase.storage
    .from('project-reference-images')
    .upload(filePath, imageBlob, {
      contentType: mimeType,
      upsert: true,
    })

  if (uploadError) {
    throw uploadError
  }

  const { data: publicUrlData } = supabase.storage
    .from('project-reference-images')
    .getPublicUrl(filePath)

  const nextRequirements: ProjectRequirements = {
    ...requirements,
    thumbnail_image_url: publicUrlData.publicUrl,
    thumbnail_image_source: 'generated',
    thumbnail_image_prompt: imageBlock.prompt,
  }

  const { error: updateError } = await supabase
    .from('projects')
    .update({ requirements: nextRequirements })
    .eq('id', projectId)

  if (updateError) {
    throw updateError
  }

  return publicUrlData.publicUrl
}
