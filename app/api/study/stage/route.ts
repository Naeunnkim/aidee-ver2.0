import { NextResponse } from 'next/server'

import { createClient } from '@/lib/supabase/server'
import { isKnownStageKey, isSameOrNextStage } from '@/lib/study'

type OpenStageRow = {
  id: string
  stage_key: string
  stage_order: number
  entered_at: string
}

export async function POST(request: Request) {
  const body = (await request.json().catch(() => null)) as
    | {
        sessionId?: string
        projectId?: string
        nextStageKey?: string
        exitReason?: string
      }
    | null

  const sessionId = body?.sessionId
  const projectId = body?.projectId
  const nextStageKey = body?.nextStageKey
  const exitReason = body?.exitReason ?? 'transition'

  if (!sessionId || !projectId || !nextStageKey || !isKnownStageKey(nextStageKey)) {
    return NextResponse.json(
      { error: 'sessionId, projectId, valid nextStageKey are required' },
      { status: 400 }
    )
  }

  const supabase = await createClient()
  const {
    data: { user },
  } = await supabase.auth.getUser()

  if (!user) {
    return NextResponse.json({ error: 'Unauthorized' }, { status: 401 })
  }

  const { data: openStage } = await supabase
    .from('design_stages')
    .select('id, stage_key, stage_order, entered_at')
    .eq('session_id', sessionId)
    .eq('project_id', projectId)
    .is('exited_at', null)
    .order('stage_order', { ascending: false })
    .limit(1)
    .maybeSingle<OpenStageRow>()

  if (openStage?.stage_key === nextStageKey) {
    return NextResponse.json({
      currentStageKey: openStage.stage_key,
      transitioned: false,
    })
  }

  if (
    openStage?.stage_key &&
    isKnownStageKey(openStage.stage_key) &&
    !isSameOrNextStage(openStage.stage_key, nextStageKey)
  ) {
    return NextResponse.json(
      {
        error: 'Non-sequential stage transition is not allowed',
        currentStageKey: openStage.stage_key,
        requestedStageKey: nextStageKey,
      },
      { status: 409 }
    )
  }

  const now = new Date()

  if (openStage) {
    const enteredAt = new Date(openStage.entered_at)
    const durationMs = Math.max(now.getTime() - enteredAt.getTime(), 0)

    const { error: updateError } = await supabase
      .from('design_stages')
      .update({
        exited_at: now.toISOString(),
        duration_ms: durationMs,
        exit_reason: exitReason,
      })
      .eq('id', openStage.id)

    if (updateError) {
      return NextResponse.json({ error: updateError.message }, { status: 500 })
    }
  }

  const nextStageOrder = (openStage?.stage_order ?? 0) + 1

  const { error: insertError } = await supabase.from('design_stages').insert({
    session_id: sessionId,
    project_id: projectId,
    stage_key: nextStageKey,
    stage_order: nextStageOrder,
    entered_at: now.toISOString(),
  })

  if (insertError) {
    return NextResponse.json({ error: insertError.message }, { status: 500 })
  }

  return NextResponse.json({
    currentStageKey: nextStageKey,
    transitioned: true,
    stageOrder: nextStageOrder,
  })
}
