import { NextResponse } from 'next/server'

import { createClient } from '@/lib/supabase/server'
import { type StageKey, isKnownStageKey } from '@/lib/study'

type ProjectRow = {
  user_id: string
}

type LastStageRow = {
  stage_key: string
}

export async function POST(request: Request) {
  const body = (await request.json().catch(() => null)) as
    | { projectId?: string }
    | null
  const projectId = body?.projectId

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

  const { data: project } = await supabase
    .from('projects')
    .select('user_id')
    .eq('id', projectId)
    .eq('user_id', user.id)
    .single<ProjectRow>()

  if (!project) {
    return NextResponse.json({ error: 'Project not found' }, { status: 404 })
  }

  const { data: lastStage } = await supabase
    .from('design_stages')
    .select('stage_key')
    .eq('project_id', projectId)
    .order('entered_at', { ascending: false })
    .limit(1)
    .maybeSingle<LastStageRow>()

  const initialStageKey: StageKey =
    lastStage?.stage_key && isKnownStageKey(lastStage.stage_key)
      ? lastStage.stage_key
      : 'step_0_start'

  const { data: session, error: sessionError } = await supabase
    .from('study_sessions')
    .insert({
      user_id: user.id,
      status: 'active',
    })
    .select('id, status, started_at')
    .single()

  if (sessionError || !session) {
    return NextResponse.json(
      { error: sessionError?.message || 'Failed to create session' },
      { status: 500 }
    )
  }

  const { error: stageError } = await supabase.from('design_stages').insert({
    session_id: session.id,
    project_id: projectId,
    stage_key: initialStageKey,
    stage_order: 1,
    entered_at: new Date().toISOString(),
  })

  if (stageError) {
    return NextResponse.json(
      { error: stageError.message || 'Failed to create initial stage' },
      { status: 500 }
    )
  }

  return NextResponse.json({
    sessionId: session.id,
    status: session.status,
    currentStageKey: initialStageKey,
    startedAt: session.started_at,
  })
}
