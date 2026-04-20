import { type NextRequest, NextResponse } from 'next/server'

import { createClient } from '@/lib/supabase/server'

function getSafeNextPath(next: string | null) {
  if (!next || !next.startsWith('/')) {
    return '/dashboard'
  }

  return next.startsWith('//') ? '/dashboard' : next
}

export async function GET(request: NextRequest) {
  const requestUrl = new URL(request.url)
  const code = requestUrl.searchParams.get('code')
  const next = getSafeNextPath(requestUrl.searchParams.get('next'))

  if (!code) {
    return NextResponse.redirect(
      new URL('/?authError=missing_code', requestUrl.origin)
    )
  }

  const supabase = await createClient()
  const { error } = await supabase.auth.exchangeCodeForSession(code)

  if (error) {
    return NextResponse.redirect(
      new URL('/?authError=oauth_callback_failed', requestUrl.origin)
    )
  }

  const {
    data: { user },
    error: userError,
  } = await supabase.auth.getUser()

  if (userError || !user) {
    return NextResponse.redirect(
      new URL('/?authError=user_fetch_failed', requestUrl.origin)
    )
  }

  const { error: profileError } = await supabase.from('profiles').upsert(
    {
      id: user.id,
      email: user.email ?? null,
      full_name:
        user.user_metadata.full_name ?? user.user_metadata.name ?? null,
      avatar_url: user.user_metadata.avatar_url ?? null,
      provider: user.app_metadata.provider ?? 'google',
    },
    {
      onConflict: 'id',
    }
  )

  if (profileError) {
    return NextResponse.redirect(
      new URL('/?authError=profile_sync_failed', requestUrl.origin)
    )
  }

  return NextResponse.redirect(new URL(next, requestUrl.origin))
}
