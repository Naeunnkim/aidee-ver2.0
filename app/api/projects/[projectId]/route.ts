import { createClient as createSupabaseClient } from '@supabase/supabase-js'
import { NextResponse } from 'next/server'

import { createClient } from '@/lib/supabase/server'
import { getSupabasePublishableKey, getSupabaseUrl } from '@/lib/supabase/env'

type ProjectRow = {
  id: string
  user_id: string
  title: string
}

type StorageRow = {
  name: string
  metadata: { size?: number } | null
}

function getSupabaseAdmin() {
  const supabaseUrl = getSupabaseUrl()
  const serviceRoleKey = process.env.SUPABASE_SERVICE_ROLE_KEY
  const publishableKey = getSupabasePublishableKey()

  return createSupabaseClient(
    supabaseUrl,
    serviceRoleKey || publishableKey,
    {
      auth: {
        persistSession: false,
        autoRefreshToken: false,
      },
    }
  )
}

function getStoragePrefixPaths(prefix: string, items: StorageRow[]) {
  return items.map((item) => `${prefix}/${item.name}`)
}

async function collectStoragePaths(
  supabase: ReturnType<typeof getSupabaseAdmin>,
  prefix: string
) {
  const paths: string[] = []
  const stack = [prefix]

  while (stack.length > 0) {
    const currentPrefix = stack.pop()
    if (!currentPrefix) {
      continue
    }

    const { data, error } = await supabase.storage
      .from('project-reference-images')
      .list(currentPrefix, { limit: 1000 })

    if (error) {
      throw error
    }

    const items = (data ?? []) as StorageRow[]
    const fileItems = items.filter((item) => item.metadata)
    const folderItems = items.filter((item) => !item.metadata)

    paths.push(...getStoragePrefixPaths(currentPrefix, fileItems))
    stack.push(...getStoragePrefixPaths(currentPrefix, folderItems))
  }

  return paths
}

async function deleteProjectStorageAssets(
  supabase: ReturnType<typeof getSupabaseAdmin>,
  projectId: string
) {
  const rootPrefix = `projects/${projectId}`
  const storagePaths = await collectStoragePaths(supabase, rootPrefix)

  if (storagePaths.length === 0) {
    return
  }

  const { error } = await supabase.storage
    .from('project-reference-images')
    .remove(storagePaths)

  if (error) {
    throw error
  }
}

export async function PATCH(
  request: Request,
  context: { params: Promise<{ projectId: string }> }
) {
  const { projectId } = await context.params
  const body = (await request.json().catch(() => null)) as
    | { title?: string }
    | null
  const nextTitle = typeof body?.title === 'string' ? body.title.trim() : ''

  if (!projectId) {
    return NextResponse.json({ error: 'projectId is required' }, { status: 400 })
  }

  if (!nextTitle) {
    return NextResponse.json({ error: 'title is required' }, { status: 400 })
  }

  const supabase = await createClient()
  const {
    data: { user },
  } = await supabase.auth.getUser()

  if (!user) {
    return NextResponse.json({ error: 'Unauthorized' }, { status: 401 })
  }

  const { data: project, error: projectError } = await supabase
    .from('projects')
    .select('id, user_id, title')
    .eq('id', projectId)
    .eq('user_id', user.id)
    .single<ProjectRow>()

  if (projectError || !project) {
    return NextResponse.json({ error: 'Project not found' }, { status: 404 })
  }

  const { error: updateError } = await supabase
    .from('projects')
    .update({ title: nextTitle })
    .eq('id', projectId)

  if (updateError) {
    return NextResponse.json(
      { error: updateError.message || 'Failed to update project' },
      { status: 500 }
    )
  }

  return NextResponse.json({
    success: true,
    project: {
      ...project,
      title: nextTitle,
    },
  })
}

export async function DELETE(
  request: Request,
  context: { params: Promise<{ projectId: string }> }
) {
  const { projectId } = await context.params

  if (!projectId) {
    return NextResponse.json({ error: 'projectId is required' }, { status: 400 })
  }

  const supabase = await createClient()
  const {
    data: { user },
  } = await supabase.auth.getUser()

  if (!user) {
    return NextResponse.json({ error: 'Unauthorized' }, { status: 401 })
  }

  const { data: project, error: projectError } = await supabase
    .from('projects')
    .select('id, user_id, title')
    .eq('id', projectId)
    .eq('user_id', user.id)
    .single<ProjectRow>()

  if (projectError || !project) {
    return NextResponse.json({ error: 'Project not found' }, { status: 404 })
  }

  const admin = getSupabaseAdmin()

  const { error: referenceDeleteError } = await admin
    .from('project_reference_images')
    .delete()
    .eq('project_id', projectId)

  if (referenceDeleteError) {
    return NextResponse.json(
      { error: referenceDeleteError.message || 'Failed to delete reference images' },
      { status: 500 }
    )
  }

  const { error: messagesDeleteError } = await admin
    .from('messages')
    .delete()
    .eq('project_id', projectId)

  if (messagesDeleteError) {
    return NextResponse.json(
      { error: messagesDeleteError.message || 'Failed to delete messages' },
      { status: 500 }
    )
  }

  const { error: stagesDeleteError } = await admin
    .from('design_stages')
    .delete()
    .eq('project_id', projectId)

  if (stagesDeleteError) {
    return NextResponse.json(
      { error: stagesDeleteError.message || 'Failed to delete design stages' },
      { status: 500 }
    )
  }

  const { error: sessionsDeleteError } = await admin
    .from('study_sessions')
    .delete()
    .eq('project_id', projectId)

  if (sessionsDeleteError) {
    return NextResponse.json(
      { error: sessionsDeleteError.message || 'Failed to delete study sessions' },
      { status: 500 }
    )
  }

  try {
    await deleteProjectStorageAssets(admin, projectId)
  } catch (error) {
    const message =
      error instanceof Error ? error.message : 'Failed to delete project assets'
    return NextResponse.json({ error: message }, { status: 500 })
  }

  const { error: projectDeleteError } = await admin
    .from('projects')
    .delete()
    .eq('id', projectId)

  if (projectDeleteError) {
    return NextResponse.json(
      { error: projectDeleteError.message || 'Failed to delete project' },
      { status: 500 }
    )
  }

  return NextResponse.json({
    success: true,
    deletedProjectId: projectId,
  })
}
